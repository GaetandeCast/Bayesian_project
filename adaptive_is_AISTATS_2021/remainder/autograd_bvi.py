# coding=utf-8
# Copyright 2020 The Adaptive Is Aistats 2021 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import autograd.numpy as np
from adaptive_is_AISTATS_2021.remainder import autograd_utils
from adaptive_is_AISTATS_2021.remainder import optimizers as opts
from adaptive_is_AISTATS_2021.remainder import utils


def remainder_rev_kl_iteration(target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               mean,
                               cov_sqrt,
                               num_samples=1000,
                               mean_step_size=0.005,
                               cov_sqrt_step_size=0.005,
                               num_iters=100,
                               tol=1e-6,
                               eps=1e-6,
                               use_adam=True,
                               residual=False,
                               residual_regulariziation=0.,
                               stabilized_gradients=True,
                               clip=False,
                               stabilize_weights=False):

  def old_mixture_logprob(theta):
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  print("Initializing with mean = {0} and cov_sqrt = {1}".format(
      np.mean(mean), np.mean(cov_sqrt)))
  init_remainder_rev_kl = autograd_utils.remainder_reverse_kl(
      mean,
      cov_sqrt,
      target_logprob,
      old_mixture_logprob,
      stabilize_diff=True,
      num_samples=num_samples,
      eps=eps,
      clip=clip)
  print(">>>> remainder_rev_kl_iteration : init_remainder_rev_kl = {} ".format(
      init_remainder_rev_kl))

  if use_adam:
    mean_optimizer = opts.Adam(lr=mean_step_size)
    cov_sqrt_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  # Setup for comparisons
  objective = 1000
  delta = 1000
  for t in range(num_iters):
    if residual:
      mean_grad, cov_sqrt_grad = autograd_utils.residual_rev_kl_grad(
          mean,
          cov_sqrt,
          target_logprob,
          old_mixture_logprob,
          regularization=residual_regulariziation,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          clip=clip)
    else:
      mean_grad, cov_sqrt_grad = autograd_utils.remainder_rev_kl_grad(
          mean,
          cov_sqrt,
          target_logprob,
          old_mixture_logprob,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          clip=clip)

    if t % 100 == 0:
      print("Mean Grad norm = {0} and cov_sqrt Grad norm = {1}".format(
          np.linalg.norm(mean_grad, ord=1),
          np.linalg.norm(cov_sqrt_grad, ord=1)))

    if use_adam:
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      cov_sqrt_update = np.array(
          cov_sqrt_optimizer.get_update(cov_sqrt, cov_sqrt_grad))
    else:
      mean_update = mean - mean_step_size * mean_grad
      cov_sqrt_update = cov_sqrt - cov_sqrt_step_size * cov_sqrt_grad
    if residual:
      new_objective = autograd_utils.residual_reverse_kl(
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_mixture_logprob,
          regularization=residual_regulariziation,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          clip=clip)
    else:
      new_objective = autograd_utils.remainder_reverse_kl(
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_mixture_logprob,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          clip=clip)

    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    if t % 25 == 0:
      print(
          "New Remainder Reverse KL is {0} for mean = {1} and cov_sqrt = {2} for sigma = {3}"
          .format(new_objective, np.mean(mean_update), np.mean(cov_sqrt_update),
                  np.mean(np.square(cov_sqrt_update))))
    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))

  return mean, cov_sqrt


def mixture_rev_kl_iteration(target_logprob,
                             old_weights,
                             old_means,
                             old_cov_sqrts,
                             mean,
                             cov_sqrt,
                             gamma=0.51,
                             num_samples=1000,
                             gamma_step_size=0.005,
                             mean_step_size=0.005,
                             cov_sqrt_step_size=0.005,
                             num_iters=100,
                             tol=1e-6,
                             eps=1e-6,
                             use_adam=True,
                             alternative=False,
                             bug=False):

  print("old Sigma invs = {}".format(old_cov_sqrts))
  print("old means = {}".format(old_means))
  print("init mean 	 shape = {}".format(mean.shape))
  print("init cov_sqrt shape = {}".format(cov_sqrt.shape))
  print("Initializing with gamma = {0} mean = {1} and cov_sqrt = {2}".format(
      gamma, np.mean(mean), np.mean(cov_sqrt)))
  if alternative:
    init_mixture_rev_kl = autograd_utils.alt_single_mixture_reverse_kl(
        gamma,
        mean,
        cov_sqrt,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        num_samples=num_samples,
        bug=bug,
        eps=eps)
  else:
    init_mixture_rev_kl = autograd_utils.single_mixture_reverse_kl(
        gamma,
        mean,
        cov_sqrt,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        differentiable=True,
        num_samples=num_samples,
        eps=eps)
  print(">>>> mixture_rev_kl_iteration : Initial Reverse KL = {} ".format(
      init_mixture_rev_kl))

  if use_adam:
    gamma_optimizer = opts.Adam(lr=gamma_step_size)
    mean_optimizer = opts.Adam(lr=mean_step_size)
    cov_sqrt_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  objective = 1000
  delta = 1000
  for t in range(num_iters):
    if alternative:
      gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.alt_mixture_rev_kl_grad(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=num_samples,
          bug=bug,
          eps=eps)
    else:
      gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.mixture_rev_kl_grad(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          differentiable=True,
          num_samples=num_samples,
          eps=eps)

    if t % 100 == 0:
      print("mean_grad shape = {}".format(mean_grad.shape))
      print("cov_sqrt_grad shape = {}".format(cov_sqrt_grad.shape))
      print(
          "Mean Grad norm = {0} and cov_sqrt Grad norm = {1} and Gamma Grad = {2} "
          .format(
              np.linalg.norm(mean_grad, ord=1),
              np.linalg.norm(cov_sqrt_grad, ord=1), gamma_grad))

    if use_adam:
      gamma_update = gamma_optimizer.get_update(
          np.array([gamma]), np.array([gamma_grad]))[0]
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      cov_sqrt_update = np.array(
          cov_sqrt_optimizer.get_update(cov_sqrt, cov_sqrt_grad))
    else:
      gamma_update = gamma - gamma_step_size * gamma_grad
      mean_update = mean - mean_step_size * mean_grad
      cov_sqrt_update = cov_sqrt - cov_sqrt_step_size * cov_sqrt_grad
    gamma_update = max(min(gamma_update, 0.999), 0.001)
    if alternative:
      new_objective = autograd_utils.alt_single_mixture_reverse_kl(
          gamma_update,
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=num_samples,
          bug=bug,
          eps=eps)
    else:
      new_objective = autograd_utils.single_mixture_reverse_kl(
          gamma_update,
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          differentiable=True,
          num_samples=num_samples,
          eps=eps)

    if t % 25 == 0:
      print(
          "New Reverse KL is {0} for gamma = {1} mean = {2} and cov_sqrt = {3} for sigma = {4}"
          .format(new_objective, gamma_update, np.mean(mean_update),
                  np.mean(cov_sqrt_update),
                  np.mean(np.square(cov_sqrt_update))))

    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    gamma = gamma_update

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))

  return gamma, mean, cov_sqrt


def joint_opt_rev_kl_iteration(target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               mean,
                               cov_sqrt,
                               gamma=0.51,
                               num_samples=1000,
                               weights_step_size=0.005,
                               means_step_size=0.005,
                               cov_sqrt_step_size=0.005,
                               num_iters=100,
                               tol=1e-6,
                               use_adam=True,
                               eps=1e-6):

  print("Old Means = {0} Old cov_sqrts = {1}".format(
      np.mean(old_means), np.mean(old_cov_sqrts)))
  print("Init Mean = {0} Init cov_sqrt = {1}".format(
      np.mean(mean), np.mean(cov_sqrt)))
  weights, means, cov_sqrts = utils.update_mixture(old_weights, old_means,
                                                   old_cov_sqrts, gamma, mean,
                                                   cov_sqrt)
  print(
      "Initializing with weights = {0} means = {1} and cov_sqrts = {2}".format(
          weights, means, cov_sqrts))
  init_mixture_rev_kl = autograd_utils.naive_mixture_reverse_kl(
      weights,
      means,
      cov_sqrts,
      target_logprob,
      num_samples=num_samples,
      eps=eps)
  print(">>>> Initial Reverse KL = {} ".format(init_mixture_rev_kl))

  if use_adam:
    weights_optimizer = opts.Adam(lr=weights_step_size)
    means_optimizer = opts.Adam(lr=means_step_size)
    cov_sqrts_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  objective = 1000
  delta = 1000

  for t in range(num_iters):
    weights_grad, means_grad, cov_sqrts_grad = autograd_utils.full_mixture_rev_kl_grad(
        weights,
        means,
        cov_sqrts,
        target_logprob,
        num_samples=num_samples,
        eps=eps)

    if use_adam:
      weights_update = np.array(
          weights_optimizer.get_update(weights, weights_grad))
      means_update = np.array(means_optimizer.get_update(means, means_grad))
      cov_sqrts_update = np.array(
          cov_sqrts_optimizer.get_update(cov_sqrts, cov_sqrts_grad))
    else:
      weights_update = weights - weights_step_size * weights_grad
      means_update = means - means_step_size * means_grad
      cov_sqrts_update = cov_sqrts - cov_sqrt_step_size * cov_sqrts_grad
    if t % 100 == 0:
      print("Weight Grad = {0}, MEan Grad = {1} and Cov_Sqrt Grad = {2}".format(
          weights_grad, means_grad, cov_sqrts_grad))
      print("Weights BEFORE Simplex projection = {}".format(weights_update))
    weights_update = opts.euclidean_proj_simplex(np.clip(weights_update, 0, 1))
    if t % 100 == 0:
      print("Weights AFTER Simplex projection = {}".format(weights_update))
    new_objective = autograd_utils.naive_mixture_reverse_kl(
        weights_update,
        means_update,
        cov_sqrts_update,
        target_logprob,
        num_samples=num_samples,
        eps=eps)
    if t % 25 == 0:
      print(
          "New Reverse KL is {0} for weights = {1} means = {2} and cov_sqrts = {3} "
          .format(new_objective, utils.logistic_transform(weights_update),
                  np.mean(means_update), np.mean(cov_sqrts_update)))

    delta = np.abs(objective - new_objective)
    objective = new_objective
    weights = weights_update
    means = means_update
    cov_sqrts = cov_sqrts_update
    # if np.any(weights_update < 1e-3):
    # 	print("One of the weights reached zero : {}".format(weights_update))
    # 	break
    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))

  return weights, means, cov_sqrts


def remainder_for_kl_iteration(target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               mean,
                               cov_sqrt,
                               num_samples=1000,
                               mean_step_size=0.005,
                               cov_sqrt_step_size=0.005,
                               num_iters=100,
                               tol=1e-6,
                               eps=1e-10,
                               use_adam=True,
                               stabilized_gradients=False,
                               stabilize_weights=False):

  print("Initializing with mean = {0} and cov_sqrt = {1}".format(
      np.mean(mean), np.mean(cov_sqrt)))
  init_remainder_for_kl = autograd_utils.remainder_forward_kl(
      mean,
      cov_sqrt,
      target_logprob,
      old_weights,
      old_means,
      old_cov_sqrts,
      num_samples=num_samples,
      stabilize_diff=True,
      eps=eps,
      stabilize_weights=stabilize_weights)
  print(">>>> remainder_for_kl_iteration : init_remainder_for_kl = {} ".format(
      init_remainder_for_kl))

  if use_adam:
    mean_optimizer = opts.Adam(lr=mean_step_size)
    cov_sqrt_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  objective = 1000
  delta = 1000
  for t in range(num_iters):
    mean_grad, cov_sqrt_grad = autograd_utils.remainder_forward_kl_grad(
        mean,
        cov_sqrt,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        num_samples=num_samples,
        stabilize_diff=stabilized_gradients,
        eps=eps,
        stabilize_weights=stabilize_weights)

    if t % 100 == 0:
      print("Mean Grad norm = {0} and cov_sqrt Grad norm = {1}".format(
          np.linalg.norm(mean_grad, ord=1),
          np.linalg.norm(cov_sqrt_grad, ord=1)))

    if use_adam:
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      cov_sqrt_update = np.array(
          cov_sqrt_optimizer.get_update(cov_sqrt, cov_sqrt_grad))
    else:
      mean_update = mean - mean_step_size * mean_grad
      cov_sqrt_update = cov_sqrt - cov_sqrt_step_size * cov_sqrt_grad
    new_objective = autograd_utils.remainder_forward_kl(
        mean_update,
        cov_sqrt_update,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        num_samples=num_samples,
        stabilize_diff=True,
        eps=eps,
        stabilize_weights=stabilize_weights)

    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    if t % 25 == 0:
      print(
          "New Remainder Forward KL is {0} for mean = {1} and cov_sqrt = {2} for sigma = {3}"
          .format(new_objective, np.mean(mean_update), np.mean(cov_sqrt_update),
                  np.mean(np.square(cov_sqrt_update))))

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))
  return mean, cov_sqrt


def mixture_for_kl_iteration(target_logprob,
                             old_weights,
                             old_means,
                             old_cov_sqrts,
                             mean,
                             cov_sqrt,
                             gamma=0.51,
                             num_samples=1000,
                             gamma_step_size=0.005,
                             mean_step_size=0.005,
                             cov_sqrt_step_size=0.005,
                             num_iters=100,
                             tol=1e-6,
                             eps=1e-6,
                             use_adam=True,
                             alternative=False,
                             gaussian_sample=False,
                             stabilized_gradients=False,
                             old_mixture=False,
                             split_weights=False,
                             stabilize_weights=False):
  print("Initializing with gamma = {0} mean = {1} and cov_sqrt = {2}".format(
      gamma, np.mean(mean), np.mean(cov_sqrt)))
  if alternative:
    if old_mixture:  #alternative with old mixture sampling
      init_mixture_for_kl = autograd_utils.alt_alt_old_mog_forward_kl(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=num_samples,
          stabilize_diff=True,
          eps=eps,
          stabilize_weights=stabilize_weights)
    elif split_weights:  #where we changed the SNIS weights to be consistent across the split
      init_mixture_for_kl = autograd_utils.alt_new_mog_forward_kl(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=num_samples,
          stabilize_diff=True,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:  # oldest
      init_mixture_for_kl = autograd_utils.alt_single_mixture_forward_kl(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=num_samples,
          stabilize_diff=True,
          eps=eps,
          stabilize_weights=stabilize_weights)
  elif old_mixture:  # old mixture sampling: could be the final
    init_mixture_for_kl = autograd_utils.old_mixture_forward_kl(
        gamma,
        mean,
        cov_sqrt,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        num_samples=num_samples,
        stabilize_diff=True,
        eps=eps,
        stabilize_weights=stabilize_weights)
  else:
    init_mixture_for_kl = autograd_utils.single_mixture_forward_kl(
        gamma,
        mean,
        cov_sqrt,
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        gaussian_sample=gaussian_sample,
        num_samples=num_samples,
        stabilize_diff=True,
        eps=eps,
        stabilize_weights=stabilize_weights)
  print(
      ">>>> mixture_for_kl_iteration : Initial Forward KL = {} | stabilized_gradients = {}"
      .format(init_mixture_for_kl, stabilized_gradients))
  if use_adam:
    gamma_optimizer = opts.Adam(lr=gamma_step_size)
    mean_optimizer = opts.Adam(lr=mean_step_size)
    cov_sqrt_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  objective = 1000
  delta = 1000
  for t in range(num_iters):
    if alternative:
      if old_mixture:
        gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.alt_alt_old_mog_forward_kl_grad(
            gamma,
            mean,
            cov_sqrt,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            stabilize_diff=stabilized_gradients,
            num_samples=num_samples,
            eps=eps,
            stabilize_weights=stabilize_weights)
      elif split_weights:
        gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.alt_new_mog_forward_kl_grad(
            gamma,
            mean,
            cov_sqrt,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            stabilize_diff=stabilized_gradients,
            num_samples=num_samples,
            eps=eps,
            stabilize_weights=stabilize_weights)
      else:
        gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.alt_single_forward_kl_grad(
            gamma,
            mean,
            cov_sqrt,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            stabilize_diff=stabilized_gradients,
            num_samples=num_samples,
            eps=eps,
            stabilize_weights=stabilize_weights)
    elif old_mixture:
      gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.old_mixture_forward_kl_grad(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      gamma_grad, mean_grad, cov_sqrt_grad = autograd_utils.single_forward_kl_grad(
          gamma,
          mean,
          cov_sqrt,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          gaussian_sample=gaussian_sample,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    if t % 100 == 0:
      print("Mean Grad mean = {0} and cov_sqrt Grad mean = {1} ".format(
          np.mean(mean_grad), np.mean(cov_sqrt_grad)))
      print(
          "Mean Grad norm = {0} and cov_sqrt Grad norm = {1} and Gamma Grad  = {2} "
          .format(
              np.linalg.norm(mean_grad, ord=1),
              np.linalg.norm(cov_sqrt_grad, ord=1), gamma_grad))

    if use_adam:
      gamma_update = gamma_optimizer.get_update(
          np.array([gamma]), np.array([gamma_grad]))[0]
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      cov_sqrt_update = np.array(
          cov_sqrt_optimizer.get_update(cov_sqrt, cov_sqrt_grad))
    else:
      gamma_update = gamma - gamma_step_size * gamma_grad
      mean_update = mean - mean_step_size * mean_grad
      cov_sqrt_update = cov_sqrt - cov_sqrt_step_size * cov_sqrt_grad

  # if gamma_update >= 1 or gamma_update <= 0:
    gamma_update = max(min(gamma_update, 0.999), 0.001)

    if alternative:
      if old_mixture:  #alternative with old mixture sampling
        new_objective = autograd_utils.alt_alt_old_mog_forward_kl(
            gamma_update,
            mean_update,
            cov_sqrt_update,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            num_samples=num_samples,
            stabilize_diff=True,
            eps=eps,
            stabilize_weights=stabilize_weights)
      elif split_weights:  #where we changed the SNIS weights to be consistent across the split
        new_objective = autograd_utils.alt_new_mog_forward_kl(
            gamma_update,
            mean_update,
            cov_sqrt_update,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            num_samples=num_samples,
            stabilize_diff=True,
            eps=eps,
            stabilize_weights=stabilize_weights)
      else:  # old
        new_objective = autograd_utils.alt_single_mixture_forward_kl(
            gamma_update,
            mean_update,
            cov_sqrt_update,
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            num_samples=num_samples,
            stabilize_diff=True,
            eps=eps,
            stabilize_weights=stabilize_weights)
    elif old_mixture:  # THE MAIN ONE (hopefully)
      new_objective = autograd_utils.old_mixture_forward_kl(
          gamma_update,
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      new_objective = autograd_utils.single_mixture_forward_kl(
          gamma_update,
          mean_update,
          cov_sqrt_update,
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          gaussian_sample=gaussian_sample,
          num_samples=num_samples,
          stabilize_diff=True,
          eps=eps,
          stabilize_weights=stabilize_weights)

    if t % 25 == 0:
      print(
          "New Forward KL is {0} for gamma = {1} mean = {2} and cov_sqrt = {3} for sigma = {4}"
          .format(new_objective, gamma_update, np.mean(mean_update),
                  np.mean(cov_sqrt_update),
                  np.mean(np.square(cov_sqrt_update))))

    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    gamma = gamma_update

    # THIS should be temporary !
    # if gamma < 1e-3 or gamma > 1:
    # 	print("GAMMA HIT ZERO. THIS IS A RECIPE FOR NANs. THIS ITERATION FAILED !!!")
    # 	break

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))
  return gamma, mean, cov_sqrt


def joint_for_kl_iteration(target_logprob,
                           old_weights,
                           old_means,
                           old_cov_sqrts,
                           mean,
                           cov_sqrt,
                           gamma=0.51,
                           num_samples=1000,
                           weights_step_size=0.005,
                           means_step_size=0.005,
                           cov_sqrt_step_size=0.005,
                           num_iters=100,
                           tol=1e-6,
                           eps=1e-6,
                           use_adam=True,
                           gaussian_sample=False,
                           stabilized_gradients=False,
                           old_mixture=False,
                           stabilize_weights=False):

  weights, means, cov_sqrts = utils.update_mixture(old_weights, old_means,
                                                   old_cov_sqrts, gamma, mean,
                                                   cov_sqrt)
  print(
      "Initializing with weights = {0} means = {1} and cov_sqrts = {2}".format(
          weights, means, cov_sqrts))
  if old_mixture:
    init_mixture_for_kl = autograd_utils.joint_old_mixture_forward_kl(
        weights,
        means,
        cov_sqrts,
        target_logprob,
        stabilize_diff=True,
        num_samples=num_samples,
        eps=eps,
        stabilize_weights=stabilize_weights)
  elif gaussian_sample:
    init_mixture_for_kl = autograd_utils.joint_mixture_mvn_forward_kl(
        weights,
        means,
        cov_sqrts,
        target_logprob,
        stabilize_diff=True,
        num_samples=num_samples,
        eps=eps,
        stabilize_weights=stabilize_weights)
  else:
    init_mixture_for_kl = autograd_utils.naive_mixture_mog_forward_kl(
        weights,
        means,
        cov_sqrts,
        target_logprob,
        stabilize_diff=True,
        num_samples=num_samples,
        eps=eps,
        stabilize_weights=stabilize_weights)
  print(">>>> Initial Forward KL (without stabilized diff) = {} ".format(
      init_mixture_for_kl))

  if use_adam:
    weights_optimizer = opts.Adam(lr=weights_step_size)
    means_optimizer = opts.Adam(lr=means_step_size)
    cov_sqrts_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  objective = 1000
  delta = 1000

  for t in range(num_iters):
    if old_mixture:
      weights_grad, means_grad, cov_sqrts_grad = autograd_utils.joint_old_mixture_forward_kl_grad(
          weights,
          means,
          cov_sqrts,
          target_logprob,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    elif gaussian_sample:
      weights_grad, means_grad, cov_sqrts_grad = autograd_utils.joint_mixture_mvn_forward_kl_grad(
          weights,
          means,
          cov_sqrts,
          target_logprob,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      weights_grad, means_grad, cov_sqrts_grad = autograd_utils.joint_mixture_mog_forward_kl_grad(
          weights,
          means,
          cov_sqrts,
          target_logprob,
          stabilize_diff=stabilized_gradients,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)

    if use_adam:
      weights_update = np.array(
          weights_optimizer.get_update(weights, weights_grad))
      means_update = np.array(means_optimizer.get_update(means, means_grad))
      cov_sqrts_update = np.array(
          cov_sqrts_optimizer.get_update(cov_sqrts, cov_sqrts_grad))
    else:
      weights_update = weights - weights_step_size * weights_grad
      means_update = means - means_step_size * means_grad
      cov_sqrts_update = cov_sqrts - cov_sqrt_step_size * cov_sqrts_grad

    # weights_update = opts.euclidean_proj_simplex(np.clip(weights_update, 0, 1))

    if old_mixture:
      new_objective = autograd_utils.joint_old_mixture_forward_kl(
          weights_update,
          means_update,
          cov_sqrts_update,
          target_logprob,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    elif gaussian_sample:
      new_objective = autograd_utils.joint_mixture_mvn_forward_kl(
          weights_update,
          means_update,
          cov_sqrts_update,
          target_logprob,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      new_objective = autograd_utils.naive_mixture_mog_forward_kl(
          weights_update,
          means_update,
          cov_sqrts_update,
          target_logprob,
          stabilize_diff=True,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)

    if t % 100 == 0:
      print(
          "New Forward KL is {0} for weights = {1} means = {2} and cov_sqrts = {3} "
          .format(new_objective, utils.logistic_transform(weights_update),
                  np.mean(means_update), np.mean(cov_sqrts_update)))

    delta = np.abs(objective - new_objective)
    objective = new_objective
    weights = weights_update
    means = means_update
    cov_sqrts = cov_sqrts_update

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))

  return weights, means, cov_sqrts
