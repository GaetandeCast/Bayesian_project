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
from autograd.scipy.special import logsumexp
import autograd.scipy.special as scpm
from autograd.scipy.special import gammaln
import autograd.scipy.stats as scst
import autograd.scipy.stats.multivariate_normal as mvn


def mog_sample(num_samples, log_weights, means, cov_sqrts):
  num_comps = len(log_weights)
  if num_comps > 0:
    weights = logistic_transform(log_weights)
    weights /= np.sum(weights)

  d = means.shape[1]
  x = np.zeros((num_samples, d))

  try:
    comp_count = np.random.multinomial(num_samples, weights)
  except ValueError as err:
    print("ValueError = {0} for Weights = {1}".format(err, weights))
    print("Sum of weights = {}".format(np.sum(weights)))
    raise

  x = np.vstack([
      mvn_sample(comp_count[n], means[n], cov_sqrts[n])
      for n in range(num_comps)
  ])
  return x


def mvn_sample(num_samples, mean, cov_sqrt):
  d = cov_sqrt.shape[0]
  std_normal = np.random.multivariate_normal(
      np.zeros(d), np.eye(d), num_samples)
  cov_sqrt_matrix = cov_sqrt
  if len(np.squeeze(cov_sqrt_matrix).shape) < 2:
    cov_sqrt_matrix = np.diag(cov_sqrt_matrix)
  return np.matmul(std_normal, cov_sqrt_matrix) + mean


def mog_logprob(x, log_weights, means, cov_sqrts, eps=1e-6):
  if len(log_weights) > 0:
    weights = logistic_transform(log_weights)
    weights /= np.sum(weights)
  if len(means) == 0:
    if np.isscalar(x) or x.size == 1 or len(x) == 1:
      return np.array(0.0)
    return np.zeros((len(x)))

  cluster_lls = []
  for weight, mean, cov_sqrt in zip(weights, means, cov_sqrts):
    log_pdf = mvn_logprob(x, mean, cov_sqrt, eps=eps)
    cluster_lls.append(np.log(weight) + log_pdf)
  logprob = logsumexp(np.vstack(cluster_lls), axis=0)

  return logprob


def mvn_logprob(x, mean, cov_sqrt, eps=1e-6):
  if len(np.squeeze(cov_sqrt).shape) < 2:  # if diagonal
    cov = np.diag(np.square(cov_sqrt))
  else:
    cov = np.dot(cov_sqrt.T, cov_sqrt)
  cov += eps * np.eye(cov_sqrt.shape[0])

  try:
    logprob = mvn.logpdf(x, mean, cov)
  except Exception as err:
    print("> Mean = {}".format(mean))
    print(">> cov_sqrt = {}".format(cov_sqrt))
    print(">> len(np.squeeze(cov_sqrt).shape) = {}".format(
        len(np.squeeze(cov_sqrt).shape)))
    print(">> cov = {}".format(cov))
    raise

  return logprob


def update_mixture(old_weights,
                   old_means,
                   old_cov_sqrts,
                   gamma,
                   new_mean,
                   new_cov_sqrt,
                   eps=1e-10):
  if len(old_means) > 0:
    new_means = np.append(old_means, [new_mean], axis=0)
    new_cov_sqrts = np.append(old_cov_sqrts, [new_cov_sqrt], axis=0)
    log_weights = logistic_transform(old_weights)
    new_weights = log_weights * (1 - gamma)
    new_weights = np.append(new_weights, gamma).astype(np.float64)
    new_weights = logit_transform(new_weights, eps=eps)
  else:
    new_means = np.array([new_mean])
    new_cov_sqrts = np.array([new_cov_sqrt])
    new_weights = weight_init()
  return new_weights, new_means, new_cov_sqrts


def update_mixture_aslists(old_weights,
                           old_means,
                           old_cov_sqrts,
                           gamma,
                           new_mean,
                           new_cov_sqrt,
                           eps=1e-10):
  new_means = list(old_means)
  new_means.append(new_mean)
  new_cov_sqrts = list(old_cov_sqrts)
  new_cov_sqrts.append(new_cov_sqrt)
  log_weights = logistic_transform(old_weights)
  new_weights = log_weights * (1 - gamma)
  new_weights = np.append(new_weights, gamma).astype(np.float64)
  new_weights = logit_transform(new_weights, eps=eps)
  return new_weights, new_means, new_cov_sqrts


def get_new_weights(old_weights, gamma, eps=1e-6):
  log_weights = logistic_transform(old_weights)
  new_weights = log_weights * (1 - gamma)
  new_weights = np.append(new_weights, gamma).astype(np.float64)
  new_weights = logit_transform(new_weights, eps=eps)
  return new_weights


############### 			Analytical Gradients 				###############


# Note: these use the sigmainv parametrization instead of cov_sqrt
def mvn_ll_grad(theta, mean, sigmainv, natural_gradients=True):
  """
  Currently, theta is a single sample
  \nabla_{\mu} \log p(\theta) = \Sigma^{-1} (\theta - \mu)
  \nabla_{\Sigma} \log p(\theta) = -\frac{1}{2} () \Sigma^{-1} -
  \Sigma^{-1} (\theta - \mu) (\theta - \mu)' \Sigma^{-1})
  """
  mean_grad = np.dot(sigmainv, theta - mean)
  if natural_gradients:
    mean_natural_grad = np.matmul(sigmainv, mean_grad)
    sigma_inv_natural_grad = sigmainv - sigmainv * np.outer(
        theta - mean, theta - mean) * sigmainv
    return mean_natural_grad, sigma_inv_natural_grad
  else:
    # This is for SigmaInv gradients (directly) + sort-of assumes diagonal since it's
    # not correcting for the off-diagonals the way we do by default for Sigma grad above.
    sigma_inv_grad = 0.5 * (
        np.linalg.inv(sigmainv) - np.outer(theta - mean, theta - mean))
    return mean_grad, sigma_inv_grad


# This is the gradient wrt theta
def mvn_ll_xgrad(theta, mean, sigmainv, natural_gradients=True):
  x_grad = -np.dot(sigmainv, theta - mean)
  natural_x_grad = np.matmul(sigmainv, x_grad)
  return natural_x_grad


def mog_ll_xgrad(theta, weights, means, sigmainvs, natural_gradients=True):
  sum_grad = 0
  for i in range(len(weights)):
    sum_grad += weights[i] * mvn_ll_xgrad(
        theta, means[i], sigmainvs[i], natural_gradients=natural_gradients)
  return sum_grad


# This is the score function gradient of the KL: \nabla KL(q||p) = E_q[\nabla log q * (log q - log p)]
# This makes an assumption of the proposal being gaussian but no assumptions on the target.
def kl_score_grad(samples,
                  target_logprob,
                  proposal_logprob,
                  mean,
                  sigmainv,
                  eps=1e-8,
                  natural_gradients=True):
  q_log_prob = proposal_logprob(samples)  #+ eps
  p_log_prob = target_logprob(samples)  #+ eps
  diff = q_log_prob - p_log_prob
  sum_mean_grad = 0
  sum_sigmainv_grad = 0
  for n in range(len(samples)):
    mean_grad, sigmainv_grad = mvn_ll_grad(samples[n], mean, sigmainv,
                                           natural_gradients)
    sum_mean_grad = sum_mean_grad + (diff[n]) * mean_grad
    sum_sigmainv_grad = sum_sigmainv_grad + (diff[n]) * sigmainv_grad
  sum_mean_grad = sum_mean_grad / len(samples)
  sum_sigmainv_grad = sum_sigmainv_grad / len(samples)
  return sum_mean_grad, sum_sigmainv_grad


def sigmainv_transform(cov_sqrt):
  full_cov_sqrt = cov_sqrt
  if len(np.squeeze(full_cov_sqrt).shape) < 2:
    full_cov_sqrt = np.diag(full_cov_sqrt)
  cov = np.dot(full_cov_sqrt, full_cov_sqrt.T)
  sigmainv = np.linalg.inv(cov)
  return sigmainv


def cov_sqrt_transform(sigmainv, eps=1e-3):
  new_sigmainv = sigmainv
  cov = np.linalg.inv(new_sigmainv)
  try:
    cov_sqrt = np.linalg.cholesky(cov)
  except np.linalg.LinAlgError as err:
    print("The original SigmaInv = {}".format(sigmainv))
    print("The improved SigmaInv = {}".format(new_sigmainv))
    print("The Covariance = {}".format(cov))
    raise
  return cov_sqrt


###############            More distributions below!            ###############


def cauchy_logprob(theta):
  """Standard Cauchy."""
  return -np.log(np.pi) - np.log(1. + np.power(theta, 2))


def cauchy_sample(num_samples=100):
  u = np.random.multivariate_normal(0, 1, num_samples)
  v = np.random.multivariate_normal(0, 1, num_samples)
  return u / v


def t_logprob(theta, mu, sigmainv, df, lndet=None):
  if lndet is None:
    lndet = -np.linalg.slogdet(sigmainv)[1]
  p = sigmainv.shape[0]
  outer_product = np.sum(
      (theta - mu) * (np.dot(sigmainv, (theta - mu).T).T), axis=1)
  return -0.5 * (df + p) * np.log(1. + outer_product / df) + gammaln(
      0.5 *
      (df + p)) - gammaln(0.5 * df) - 0.5 * p * np.log(df * np.pi) - 0.5 * lndet


def banana_logprob(theta, b=0.1):
  x = theta[:, 0]
  y = theta[:, 1]
  return -x**2 / 200 - (y + b * x**2 - 100 * b)**2 / 2 - np.log(2 * np.pi * 10)


def banana_sample(num_samples=100, b=0.1):
  u = np.random.multivariate_normal(0, 100, num_samples)
  v = np.random.multivariate_normal(0, 1, num_samples)
  x = np.zeros((num_samples, 2))
  x[:, 0] = u
  x[:, 1] = v - b * np.power(u, 2) + 100 * b
  return x


def twenty_mog_logprob(theta, equal_variance=True):
  means = np.array([[2.18, 5.76], [8.67, 9.59], [4.24, 8.48], [8.41, 1.68],
                    [3.93, 8.82], [3.25, 3.47], [1.70, 0.5], [4.59, 5.60],
                    [6.91, 5.81], [6.87, 5.40], [5.41, 2.65], [2.70, 7.88],
                    [4.98, 3.70], [1.14, 2.39], [8.33, 9.50], [4.93, 1.50],
                    [1.83, 0.09], [2.26, 0.31], [5.54, 6.86], [1.69, 8.11]])
  if equal_variance:
    weights = 0.05 * np.ones(20)
    sigmainvs = np.array([100 * np.eye(2)] * 20)
  else:
    mu = np.array([5., 5.]).T
    weights = np.array([1. / np.linalg.norm(mean - mu) for mean in means])
    weights = weights / np.sum(weights)
    print("np.linalg.norm(mean - mu) = {}".format(
        np.linalg.norm(means[0] - mu)))
    sigmainvs = np.array(
        [20. * np.eye(2) / np.linalg.norm(mean - mu) for mean in means])
  return mog_logprob(theta, weights, means, sigmainvs)


def beta_logprob(theta, a, b, loc=0, scale=1):
  return scst.beta.logpdf(theta, a, b, loc, scale)


def beta_sample(num_samples, a, b, loc=0, scale=1):
  return scst.beta.rvs(a, b, loc, scale, num_samples)


########################################################################################


def logistic_transform(weights):
  weights = 1 / (1 + np.exp(-weights))
  weights = weights.astype(np.float64)
  return weights


def logit_transform(weights, eps=1e-6):
  weights = np.log((weights) / (1 - weights))
  weights = weights.astype(np.float64)
  return weights


def weight_init():
  return np.array([1000.])


def renormalize_logit_weights(logit_weights, eps=1e-6):
  weights = logistic_transform(logit_weights)
  weights /= np.sum(weights)
  return logit_transform(weights, eps=eps)
