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
from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import optimizers as opts
from adaptive_is_AISTATS_2021.remainder import utils


# f_i = argmin_f KL (f || q_0/r_{i-1}), if reaminder=True
# f_i = argmin_f KL (\gamma f + (1-\gamma) r_{i-1} || q_0), if remainder=False
def remainder_rev_kl_iteration(target_logprob,
                               weights,
                               means,
                               cov_sqrts,
                               init_mean,
                               init_cov_sqrt,
                               mean_step_size=0.001,
                               cov_sqrt_step_size=0.001,
                               num_samples=100,
                               num_iters=100,
                               tol=1e-10,
                               eps=1e-6,
                               natural_gradients=True,
                               use_adam=True):

  def old_mixture_logprob(theta):
    return utils.mog_logprob(theta, weights, means, cov_sqrts)  #+ eps

  def new_target_logprob(theta):
    return np.log((np.exp(target_logprob(theta)) + eps) /
                  (np.exp(old_mixture_logprob(theta)) + eps))

  mean = init_mean
  cov_sqrt = init_cov_sqrt

  objective = 1000
  delta = 1000

  print("initializing with mean = {0} and cov_sqrt = {1} ".format(
      mean, cov_sqrt))
  init_for_kl, init_rev_kl = metrics.eval_component(new_target_logprob, mean,
                                                    cov_sqrt)

  print("We're initializing the mean at {0} and the cov_sqrt at {1}".format(
      mean, cov_sqrt))
  print("The INITIAL Reverse KL is : {0} and Forward KL is : {1}".format(
      init_rev_kl, init_for_kl))

  if use_adam:
    mean_optimizer = opts.Adam(lr=mean_step_size)
    sigmainv_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  for t in range(num_iters):

    def proposal_logprob(xs):
      return utils.mvn_logprob(xs, mean, cov_sqrt)

    samples = utils.mvn_sample(num_samples, mean, cov_sqrt)

    sigmainv = utils.sigmainv_transform(cov_sqrt)
    mean_grad, sigmainv_grad = utils.kl_score_grad(
        samples,
        new_target_logprob,
        proposal_logprob,
        mean,
        sigmainv,
        eps=eps,
        natural_gradients=True)
    if t % 50 == 0:
      print("Mean Grad = {0} and SigmaInv Grad = {1}".format(
          mean_grad, sigmainv_grad))

    if use_adam:
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      sigmainv_update = np.array(
          sigmainv_optimizer.get_update(sigmainv, sigmainv_grad))
    else:
      mean_update = mean - mean_step_size * mean_grad
      sigmainv_update = sigmainv - cov_sqrt_step_size * sigmainv_grad
    cov_sqrt_update = utils.cov_sqrt_transform(sigmainv_update)

    # Assessing convergence
    def new_proposal_logprob(xs):
      return utils.mvn_logprob(xs, mean_update, cov_sqrt_update)

    new_objective = metrics.reverse_kl(samples, new_target_logprob,
                                       new_proposal_logprob)
    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    if t % 50 == 0:
      print(
          "New Reverse KL is {0} for mean = {1} and cov_sqrt = {2} for sigmainv = {3}"
          .format(new_objective, mean_update, cov_sqrt_update,
                  utils.sigmainv_transform(cov_sqrt_update)))
    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(">>> DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
          .format(delta, tol))

  for_kl, rev_kl = metrics.eval_component(new_target_logprob, mean, cov_sqrt)
  print("The new mean is {0} and cov_sqrt = {1}".format(mean, cov_sqrt))
  print("The final Reverse KL is : {0} and Forward KL is : {1}".format(
      rev_kl, for_kl))
  return mean, cov_sqrt


# f_i = argmin_f KL(q_0/r_{i-1} || f)
def remainder_forward_kl_iteration(target_logprob,
                                   weights,
                                   means,
                                   cov_sqrts,
                                   mean_q,
                                   cov_sqrt_q,
                                   mean_step_size=0.001,
                                   cov_sqrt_step_size=0.001,
                                   num_samples=100,
                                   num_iters=100,
                                   tol=1e-10,
                                   eps=1e-6,
                                   natural_gradients=True,
                                   use_adam=True):

  def old_mixture_logprob(theta):
    return utils.mog_logprob(theta, weights, means, cov_sqrts)  #+ eps

  def new_target_logprob(theta):
    return np.log((np.exp(target_logprob(theta)) + eps) /
                  (np.exp(old_mixture_logprob(theta)) + eps))

  mean = mean_q
  cov_sqrt = cov_sqrt_q

  init_for_kl, init_rev_kl = metrics.eval_component(new_target_logprob, mean,
                                                    cov_sqrt)
  print("The INITIAL Forward KL is : {0} and Forward KL is : {1}".format(
      init_rev_kl, init_for_kl))

  objective = 1000
  delta = 1000

  if use_adam:
    mean_optimizer = opts.Adam(lr=mean_step_size)
    sigmainv_optimizer = opts.Adam(lr=cov_sqrt_step_size)
  for t in range(num_iters):
    # Defining this function to compute the likelihood under the current params.
    def proposal_logprob(xs):
      return utils.mvn_logprob(xs, mean, cov_sqrt)

    samples = utils.mvn_sample(num_samples, mean, cov_sqrt)
    sampling_weights = metrics.snis_weights(samples, new_target_logprob,
                                            proposal_logprob)
    # print("sampling_weights shape = {0} at itr {1}".format(sampling_weights.shape, t))
    # Gradient Descent
    sum_mean_grad = 0
    sum_sigmainv_grad = 0
    # \nabla_{\mu} KL (p_{\mu, \Sigma} || q) = - \sum_i w(x_i) \nabla_{\mu} \log p(x_i)
    sigmainv = utils.sigmainv_transform(cov_sqrt)
    for n in range(num_samples):
      mean_grad, sigmainv_grad = utils.mvn_ll_grad(
          samples[n], mean, sigmainv, natural_gradients=True)
      sum_mean_grad -= sampling_weights[n] * mean_grad
      sum_sigmainv_grad -= sampling_weights[n] * sigmainv_grad

    if use_adam:
      mean_update = np.array(mean_optimizer.get_update(mean, mean_grad))
      sigmainv_update = np.array(
          sigmainv_optimizer.get_update(sigmainv, sigmainv_grad))
    else:
      mean_update = mean - mean_step_size * sum_mean_grad
      sigmainv_update = sigmainv - cov_sqrt_step_size * sum_sigmainv_grad

    cov_sqrt_update = utils.cov_sqrt_transform(sigmainv_update)

    # Assessing convergence
    def new_proposal_logprob(xs):
      return utils.mvn_logprob(xs, mean_update, cov_sqrt_update)

    new_samples = utils.mvn_sample(num_samples, mean_update, cov_sqrt_update)
    new_objective = metrics.forward_kl(new_samples, new_target_logprob,
                                       new_proposal_logprob)
    delta = np.abs(objective - new_objective)
    objective = new_objective
    mean = mean_update
    cov_sqrt = cov_sqrt_update
    if t % 50 == 0:
      print(
          "New Objective is {0} for mean = {1} and cov_sqrt = {2} for sigma = {3}"
          .format(new_objective, mean_update, cov_sqrt_update,
                  utils.sigmainv_transform(cov_sqrt_update)))

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break
    # Compute the Forward KL at this point :
  if t == num_iters and delta > tol:
    print("DID NOT CONVERGE as the final delta is {0} > {1} : tolerance".format(
        delta, tol))
  for_kl, rev_kl = metrics.eval_component(new_target_logprob, mean, cov_sqrt)
  print("The final Forward KL is : {0} and Forward KL is : {1}".format(
      rev_kl, for_kl))

  return mean, cov_sqrt
