import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

# ----------------------------
# Define the target and base densities
# ----------------------------

def cauchy_pdf(x):
    """Standard Cauchy density."""
    return 1.0 / (np.pi * (1 + x**2))

def gaussian_pdf(x, mu, sigma):
    """Univariate Gaussian density."""
    return 1.0/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

def mixture_pdf(x, components):
    """
    Evaluate a mixture of Gaussians.
    Each component is a tuple (weight, mu, sigma).
    """
    pdf_val = 0.0
    for (w, mu, sigma) in components:
        pdf_val += w * gaussian_pdf(x, mu, sigma)
    return pdf_val

# ----------------------------
# Sampling from a mixture of Gaussians
# ----------------------------

def sample_mixture(components, n_samples):
    """
    Sample n_samples points from the mixture.
    """
    weights = np.array([comp[0] for comp in components])
    indices = np.random.choice(len(components), size=n_samples, p=weights)
    samples = np.zeros(n_samples)
    for i, idx in enumerate(indices):
        _, mu, sigma = components[idx]
        samples[i] = np.random.normal(mu, sigma)
    return samples

# ----------------------------
# Divergence Estimation (always measure FKL)
# ----------------------------

def compute_FKL(components, n_samples):
    """
    Compute the forward KL divergence using a self-normalized
    importance sampling (SNIS) estimator:
      FKL ≈ sum_s w_s [ log p(x_s) - log q(x_s) ]
    where samples x_s are drawn from the mixture q.
    """
    samples = sample_mixture(components, n_samples)
    q_vals = mixture_pdf(samples, components)
    p_vals = cauchy_pdf(samples)
    r = p_vals / (q_vals + 1e-10)
    w = r / np.sum(r)
    fkl = np.sum(w * (np.log(p_vals + 1e-10) - np.log(q_vals + 1e-10)))
    return fkl

# ----------------------------
# Boosting update step (FKL-based)
# ----------------------------

def boosting_update_FKL(components, n_steps, n_samples_grad, lr):
    """
    FKL-based boosting update: optimize new component parameters by 
    minimizing the SNIS estimator of the forward KL:
      obj = sum_{s} w_s [ log p(x_s) - log q_new(x_s) ],
    where samples are drawn from q_old.
    """
    def objective(params):
        mu, log_sigma, gamma_u = params
        gamma = 1.0 / (1.0 + np.exp(-gamma_u))
        sigma = np.exp(log_sigma)
        samples = sample_mixture(components, n_samples_grad)
        q_old_vals = mixture_pdf(samples, components)
        f_vals = gaussian_pdf(samples, mu, sigma)
        q_new_vals = (1 - gamma) * q_old_vals + gamma * f_vals
        p_vals = cauchy_pdf(samples)
        r = p_vals / (q_new_vals + 1e-10)
        w = r / np.sum(r)
        obj = np.sum(w * (np.log(p_vals + 1e-10) - np.log(q_new_vals + 1e-10)))
        return obj

    params = np.array([0.0, 0.0, 0.0])  # initial: mu=0, log_sigma=0 (sigma=1), gamma=0.5
    grad_obj = grad(objective)
    for i in range(n_steps):
        g = grad_obj(params)
        params = params - lr * g

    mu_opt, log_sigma_opt, gamma_u_opt = params
    sigma_opt = np.exp(log_sigma_opt)
    gamma_opt = 1.0 / (1.0 + np.exp(-gamma_u_opt))
    
    # Update mixture: re-scale old weights and add new component.
    new_components = [(w * (1 - gamma_opt), mu, sigma) for (w, mu, sigma) in components]
    new_components.append((gamma_opt, mu_opt, sigma_opt))
    total_weight = sum(comp[0] for comp in new_components)
    new_components = [(w / total_weight, mu, sigma) for (w, mu, sigma) in new_components]
    return new_components

# ----------------------------
# Boosting update step (RKL-based)
# ----------------------------

def boosting_update_RKL(components, n_steps, n_samples_grad, lr):
    """
    RKL-based boosting update: optimize new component parameters by 
    minimizing an importance-sampling estimator of the reverse KL:
      RKL ≈ sum_s w_s [ log q_new(x_s) - log p(x_s) ],
    where samples x_s are drawn from q_old and weighted by r = q_new / q_old.
    """
    def objective(params):
        mu, log_sigma, gamma_u = params
        gamma = 1.0 / (1.0 + np.exp(-gamma_u))
        sigma = np.exp(log_sigma)
        samples = sample_mixture(components, n_samples_grad)
        q_old_vals = mixture_pdf(samples, components)
        f_vals = gaussian_pdf(samples, mu, sigma)
        q_new_vals = (1 - gamma) * q_old_vals + gamma * f_vals
        # Importance weight to account for sampling from q_old rather than q_new:
        r = q_new_vals / (q_old_vals + 1e-10)
        w = r / np.sum(r)
        # Objective approximating E_{q_new}[log q_new - log p]:
        obj = np.sum(w * (np.log(q_new_vals + 1e-10) - np.log(cauchy_pdf(samples) + 1e-10)))
        return obj

    params = np.array([0.0, 0.0, 0.0])
    grad_obj = grad(objective)
    for i in range(n_steps):
        g = grad_obj(params)
        params = params - lr * g

    mu_opt, log_sigma_opt, gamma_u_opt = params
    sigma_opt = np.exp(log_sigma_opt)
    gamma_opt = 1.0 / (1.0 + np.exp(-gamma_u_opt))
    
    new_components = [(w * (1 - gamma_opt), mu, sigma) for (w, mu, sigma) in components]
    new_components.append((gamma_opt, mu_opt, sigma_opt))
    total_weight = sum(comp[0] for comp in new_components)
    new_components = [(w / total_weight, mu, sigma) for (w, mu, sigma) in new_components]
    return new_components

# ----------------------------
# Run the boosting procedure for a given method
# ----------------------------

def run_boosting_method(method, K, n_steps, n_samples_grad, lr):
    """
    Run boosting to build a mixture with K components.
    method: "FKL" or "RKL"
    """
    # Initial mixture: one diffuse Gaussian.
    components = [(1.0, 0.0, 5.0)]
    for i in range(K - 1):
        if method == "FKL":
            components = boosting_update_FKL(components, n_steps, n_samples_grad, lr)
        elif method == "RKL":
            components = boosting_update_RKL(components, n_steps, n_samples_grad, lr)
        else:
            raise ValueError("Unknown method: choose 'FKL' or 'RKL'")
    return components

# ----------------------------
# Experiment functions: repeat each experiment multiple times and average FKL results
# ----------------------------

def experiment_vary_components(method, components_range, n_steps, n_samples_grad, lr, n_samples_eval, n_runs=10):
    fkl_values = []
    for K in components_range:
        fkl_runs = []
        for run in range(n_runs):
            comps = run_boosting_method(method, K, n_steps, n_samples_grad, lr)
            fkl_runs.append(compute_FKL(comps, n_samples_eval))
        avg_fkl = np.mean(fkl_runs)
        fkl_values.append(avg_fkl)
        print(f"Method {method} - K = {K}, avg FKL = {avg_fkl}")
    return fkl_values

def experiment_vary_n_steps(method, n_steps_range, K, n_samples_grad, lr, n_samples_eval, n_runs=10):
    fkl_values = []
    for n_steps in n_steps_range:
        fkl_runs = []
        for run in range(n_runs):
            comps = run_boosting_method(method, K, n_steps, n_samples_grad, lr)
            fkl_runs.append(compute_FKL(comps, n_samples_eval))
        avg_fkl = np.mean(fkl_runs)
        fkl_values.append(avg_fkl)
        print(f"Method {method} - n_steps = {n_steps}, avg FKL = {avg_fkl}")
    return fkl_values

def experiment_vary_is_samples(method, n_samples_range, K, n_steps, n_samples_grad, lr, n_runs=10):
    fkl_values = []
    for n_samples in n_samples_range:
        fkl_runs = []
        for run in range(n_runs):
            comps = run_boosting_method(method, K, n_steps, n_samples_grad, lr)
            fkl_runs.append(compute_FKL(comps, n_samples))
        avg_fkl = np.mean(fkl_runs)
        fkl_values.append(avg_fkl)
        print(f"Method {method} - IS samples = {n_samples}, avg FKL = {avg_fkl}")
    return fkl_values

# ----------------------------
# Main procedure: run experiments and plot FKL evolution for both methods
# ----------------------------

def main():
    # Default hyperparameters:
    default_n_steps = 200       # VI iterations per boosting update
    default_n_samples_grad = 25  # samples for gradient estimation
    default_lr = 0.01           # learning rate
    default_n_samples_eval = 1000  # IS samples for evaluation
    n_runs = 10                 # number of repeated runs per parameter setting

    # Define hyperparameter ranges:
    components_range = range(1, 11)
    n_steps_range = range(200, 1001, 100)
    n_samples_range = range(25, 201, 25)

    # Experiment 1: Vary number of boosting components
    fkl_comp_FKL = experiment_vary_components("FKL", components_range, default_n_steps, default_n_samples_grad, default_lr, default_n_samples_eval, n_runs)
    fkl_comp_RKL = experiment_vary_components("RKL", components_range, default_n_steps, default_n_samples_grad, default_lr, default_n_samples_eval, n_runs)
    
    plt.figure()
    plt.plot(list(components_range), fkl_comp_FKL, marker='o', label='FKL-optimized')
    plt.plot(list(components_range), fkl_comp_RKL, marker='s', label='RKL-optimized')
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Forward KL Divergence")
    plt.title("Evolution of FKL vs. Boosting Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("n_boost.png")

    # Experiment 2: Vary number of VI iterations per boosting update (fix K)
    fixed_K = 1
    fkl_steps_FKL = experiment_vary_n_steps("FKL", n_steps_range, fixed_K, default_n_samples_grad, default_lr, default_n_samples_eval, n_runs)
    fkl_steps_RKL = experiment_vary_n_steps("RKL", n_steps_range, fixed_K, default_n_samples_grad, default_lr, default_n_samples_eval, n_runs)
    
    plt.figure()
    plt.plot(list(n_steps_range), fkl_steps_FKL, marker='o', label='FKL-optimized')
    plt.plot(list(n_steps_range), fkl_steps_RKL, marker='s', label='RKL-optimized')
    plt.xlabel("Number of VI Iterations")
    plt.ylabel("Forward KL Divergence")
    plt.title("Evolution of FKL vs. VI Iterations ")
    plt.legend()
    plt.grid(True)
    plt.savefig("n_VI.png")

    # Experiment 3: Vary number of IS samples used in evaluation (fix K and n_steps)
    fkl_is_FKL = experiment_vary_is_samples("FKL", n_samples_range, fixed_K, default_n_steps, default_n_samples_grad, default_lr, n_runs)
    fkl_is_RKL = experiment_vary_is_samples("RKL", n_samples_range, fixed_K, default_n_steps, default_n_samples_grad, default_lr, n_runs)
    
    plt.figure()
    plt.plot(list(n_samples_range), fkl_is_FKL, marker='o', label='FKL-optimized')
    plt.plot(list(n_samples_range), fkl_is_RKL, marker='s', label='RKL-optimized')
    plt.xlabel("Number of IS Samples")
    plt.ylabel("Forward KL Divergence")
    plt.title("Evolution of FKL vs. IS Samples ")
    plt.legend()
    plt.grid(True)
    plt.savefig("n_IS.png")

if __name__ == "__main__":
    main()
