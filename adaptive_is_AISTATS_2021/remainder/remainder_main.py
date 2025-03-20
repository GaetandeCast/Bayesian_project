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
import scipy.stats as scpst

from adaptive_is_AISTATS_2021.remainder import autograd_bvi
from adaptive_is_AISTATS_2021.remainder import autograd_utils
from adaptive_is_AISTATS_2021.remainder import bvi
from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import utils
from adaptive_is_AISTATS_2021.remainder import weight_search


def boosting_vi(target_logprob, d, num_boosting_iters, **kwargs):
  means = np.empty((0, d))
  cov_sqrts = np.empty((0, d, d))
  weights = np.empty((0, 1))
  for i in range(num_boosting_iters):
    new_means, new_cov_sqrts, new_weights, new_metrics = boosting_iteration(
        i,
        target_logprob,
        d,
        old_means=means,
        old_cov_sqrts=cov_sqrts,
        old_weights=weights,
        **kwargs)
    means = new_means
    cov_sqrts = new_cov_sqrts
    weights = new_weights
    print("Curent Weights = {0} Means = {1} Cov_Sqrts = {2} ".format(
        weights, means, cov_sqrts))
    print("Metrics = {}".format(new_metrics))
    print(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end of iteration : {}"
        .format(i))
  return weights, means, cov_sqrts, new_metrics


def boosting_iteration(itr,
                       target_logprob,
                       d,
                       old_weights,
                       old_means,
                       old_cov_sqrts,
                       diagonal=True,
                       init_mean=None,
                       init_cov_sqrt=None,
                       init_weight=None,
                       num_vi_iters=1000,
                       num_samples=1000,
                       mean_step_size=0.01,
                       cov_sqrt_step_size=0.01,
                       gamma_step_size=0.001,
                       use_adam=True,
                       weight_search_method="fully_corrective",
                       target_samples=None,
                       natural_gradients=False,
                       gaussian_sample=False,
                       stabilized_gradients=False,
                       residual_regulariziation=0.,
                       residual=False,
                       gradient_init=False,
                       tol=1e-6,
                       eps=1e-6,
                       analytical=True,
                       remainder=True,
                       reverse=True,
                       joint=False,
                       alternative=False,
                       old_mixture=False,
                       split_weights=False,
                       clip=False,
                       mixed=False,
                       stabilize_weights=False,
                       stabilize_remainder=False,
                       mcmc=False,
                       doubly_remainder=False,
                       target_logprob_tf=None):
  if gradient_init:
    print("Trying Gradient Descent to find the next best initial mean.")
    print(
        "Gradient Init Proposal : Old Weights = {0} Old Means = {1} and Old Cov Sqrts = {2}"
        .format(old_weights, old_means, old_cov_sqrts))

    def proposal_logprob(theta):
      return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

    if init_mean is None:
      init_theta = np.zeros((d,))
    else:
      init_theta = init_mean

    if init_cov_sqrt is None:
      init_stddev = 0.003 * np.eye(d)
    elif diagonal:
      init_stddev = 3. * np.diag(init_cov_sqrt)
    else:
      init_stddev = 3. * init_cov_sqrt

    # print("ADAM MAP : init_theta = {} and init_stddev = {}".format(init_theta, init_stddev))
    stabilize_remainder_value = stabilize_remainder
    if itr == 0:
      stabilize_remainder_value = False
    init_mean = autograd_utils.adam_map(
        target_logprob,
        proposal_logprob,
        d=d,
        init_theta=init_theta,
        init_stddev=init_stddev,
        lr=mean_step_size,
        max_iter=400,
        tol=tol,
        eps=eps,
        diagonal=diagonal,
        stabilize_remainder=stabilize_remainder_value)
    print("ADAM MAP init_mean = {}".format(np.mean(init_mean)))
  elif init_mean is None:
    init_mean = np.zeros(d)
  if init_cov_sqrt is None:
    if diagonal:
      init_cov_sqrt = np.ones(d)  # try 0.5
    else:
      init_cov_sqrt = np.eye(d)  # try 0.5
  elif diagonal and len(np.squeeze(init_cov_sqrt).shape) > 1:
    raise ValueError(
        "Shape of init_cov_sqrt = {} while diagonal is True".format(
            init_cov_sqrt.shape))

  if init_weight is None:
    init_weight = utils.weight_init()
  gamma = 0.51  # : 0.5+eps ?

  ########################### MAIN TRAINING LOOP ###############################
  if mixed and itr == 0:
    gamma, new_mean, new_cov_sqrt = autograd_bvi.mixture_rev_kl_iteration(
        target_logprob,
        old_weights,
        old_means,
        old_cov_sqrts,
        init_mean,
        init_cov_sqrt,
        gamma=gamma,
        num_samples=num_samples,
        gamma_step_size=gamma_step_size,
        mean_step_size=mean_step_size,
        cov_sqrt_step_size=cov_sqrt_step_size,
        num_iters=num_vi_iters,
        tol=tol,
        eps=eps,
        use_adam=use_adam,
        alternative=alternative)
  elif mcmc:
    assert target_logprob_tf is not None
    gamma, new_mean, new_cov_sqrt = autograd_bvi.mcmc_for_kl_iteration(
        target_logprob,
        target_logprob_tf,
        old_weights,
        old_means,
        old_cov_sqrts,
        init_mean,
        init_cov_sqrt,
        d=d,
        gamma=gamma,
        gamma_step_size=gamma_step_size,
        mean_step_size=mean_step_size,
        cov_sqrt_step_size=cov_sqrt_step_size,
        num_iters=num_vi_iters,
        num_samples=num_samples,
        tol=tol,
        eps=eps,
        use_adam=use_adam,
        stabilize_diff=stabilized_gradients,
        stabilize_weights=stabilize_weights,
        remainder=remainder,
        doubly_remainder=doubly_remainder)

  elif analytical:
    if reverse:
      new_mean, new_cov_sqrt = bvi.remainder_rev_kl_iteration(
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          init_mean,
          init_cov_sqrt,
          mean_step_size=mean_step_size,
          cov_sqrt_step_size=cov_sqrt_step_size,
          num_samples=num_samples,
          num_iters=num_vi_iters,
          tol=tol,
          eps=eps,
          natural_gradients=natural_gradients,
          use_adam=use_adam)
    else:
      new_mean, new_cov_sqrt = bvi.remainder_forward_kl_iteration(
          target_logprob,
          old_weights,
          old_means,
          old_cov_sqrts,
          init_mean,
          init_cov_sqrt,
          mean_step_size=mean_step_size,
          cov_sqrt_step_size=cov_sqrt_step_size,
          num_samples=num_samples,
          num_iters=num_vi_iters,
          tol=tol,
          eps=eps,
          natural_gradients=natural_gradients,
          use_adam=use_adam)
  else:  # AutoGrad
    if joint:
      if reverse:
        weights, means, cov_sqrts = autograd_bvi.joint_opt_rev_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            gamma=gamma,
            num_samples=num_samples,
            weights_step_size=gamma_step_size,
            means_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            use_adam=use_adam,
            eps=eps)
      else:
        weights, means, cov_sqrts = autograd_bvi.joint_for_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            gamma=gamma,
            num_samples=num_samples,
            weights_step_size=gamma_step_size,
            means_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            eps=eps,
            use_adam=use_adam,
            stabilized_gradients=stabilized_gradients,
            gaussian_sample=False,
            old_mixture=old_mixture,
            stabilize_weights=stabilize_weights)
    elif remainder:
      if reverse:
        new_mean, new_cov_sqrt = autograd_bvi.remainder_rev_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            num_samples=1000,
            mean_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            eps=eps,
            use_adam=use_adam,
            residual=residual,
            residual_regulariziation=residual_regulariziation,
            stabilized_gradients=stabilized_gradients,
            clip=clip)
      else:
        new_mean, new_cov_sqrt = autograd_bvi.remainder_for_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            num_samples=num_samples,
            mean_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            eps=eps,
            use_adam=use_adam,
            stabilized_gradients=stabilized_gradients,
            stabilize_weights=stabilize_weights)
    else:
      if reverse:
        gamma, new_mean, new_cov_sqrt = autograd_bvi.mixture_rev_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            gamma=gamma,
            num_samples=num_samples,
            gamma_step_size=gamma_step_size,
            mean_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            eps=eps,
            use_adam=use_adam,
            alternative=alternative)
      else:
        gamma, new_mean, new_cov_sqrt = autograd_bvi.mixture_for_kl_iteration(
            target_logprob,
            old_weights,
            old_means,
            old_cov_sqrts,
            init_mean,
            init_cov_sqrt,
            gamma=gamma,
            num_samples=num_samples,
            gamma_step_size=gamma_step_size,
            mean_step_size=mean_step_size,
            cov_sqrt_step_size=cov_sqrt_step_size,
            num_iters=num_vi_iters,
            tol=tol,
            eps=eps,
            use_adam=use_adam,
            gaussian_sample=False,
            stabilized_gradients=stabilized_gradients,
            alternative=alternative,
            old_mixture=old_mixture,
            split_weights=split_weights,
            stabilize_weights=stabilize_weights)

  # Mixture weight search
  if len(old_means) > 0 and not joint:
    print("Length of Means before addition = {}".format(len(old_means)))
    if weight_search_method == "simple":
      gamma = 2. / (itr + 2.)
      new_weights = utils.get_new_weights(old_weights, gamma)
    elif weight_search_method == "line_search":
      new_weights = weight_search.weights_line_search(
          target_logprob,
          new_mean,
          new_cov_sqrt,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=1000,
          tol=1e-6,
          num_iters=500,
          init_gamma=gamma)
    elif weight_search_method == "uniform":
      n = len(old_means) + 1
      new_weights = 1. / n * np.ones(n).astype(np.float64)
      new_weights = utils.logit_transform(new_weights, eps=eps)
    elif weight_search_method == "fully_corrective":
      new_weights = weight_search.fully_corrective_weights_gd(
          target_logprob,
          new_mean,
          new_cov_sqrt,
          old_weights,
          old_means,
          old_cov_sqrts,
          num_samples=1000,
          num_iters=500,
          tol=tol,
          step_size=gamma_step_size,
          init_gamma=gamma,
          forward_kl=(not reverse),
          eps=eps,
          clip=clip)
    else:
      print("WARNING : Did not finetune the weights !")
  elif not joint:
    new_weights = utils.weight_init()

  if not joint:
    if utils.logistic_transform(new_weights[-1]) < 1e-3:
      print("Not adding new comp with weight = {}: Logistc = {}".format(
          new_weights[-1], utils.logistic_transform(new_weights[-1])))
      weights = new_weights[:-1].astype(np.float64)
      means = old_means
      cov_sqrts = old_cov_sqrts
    else:
      if len(old_means) > 0:
        means = np.append(old_means, [new_mean], axis=0)
        cov_sqrts = np.append(old_cov_sqrts, [new_cov_sqrt], axis=0)
      else:
        means = np.array([new_mean])
        cov_sqrts = np.array([new_cov_sqrt])
      weights = new_weights.astype(np.float64)

  def proposal_logprob(theta):
    return utils.mog_logprob(theta, weights, means, cov_sqrts)

  max_moment = 10
  num_eval_samples = 5000
  samples = utils.mog_sample(num_eval_samples, weights, means, cov_sqrts)
  new_moments, new_snis_moments = metrics.empirical_moments(
      samples,
      proposal_logprob=proposal_logprob,
      target_logprob=target_logprob,
      max_moment=max_moment,
      eps=eps)
  moments_diff = None
  snis_diff = None
  true_for_kl = None
  if target_samples is not None:
    p_log_prob = target_logprob(target_samples)
    q_log_prob = proposal_logprob(target_samples)
    true_for_kl = np.mean(p_log_prob - q_log_prob)
    target_moments = np.array([
        scpst.moment(target_samples, moment=m)
        for m in range(1, max_moment + 1)
    ])
    print(">>>> The TRUE Forward KL = {}".format(true_for_kl))
    moments_diff = target_moments - new_moments
    snis_diff = target_moments - new_snis_moments
    print(">>> Difference of SNIS moments = {}".format(snis_diff))

  ess, cv = metrics.eval_importance_weights(
      samples,
      target_logprob,
      proposal_logprob,
      num_eval_samples,
      eps=eps,
      stabilize_weights=stabilize_weights)
  print(">>> Eval IS = {}".format([ess, cv]))

  metrics_dict = {}
  metrics_dict["moments"] = moments_diff
  metrics_dict["snis_moments"] = snis_diff
  metrics_dict["ESS"] = ess
  metrics_dict["cv"] = cv
  metrics_dict["kl"] = true_for_kl
  return means, cov_sqrts, weights, metrics_dict
