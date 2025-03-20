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
from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import optimizers as opts
from adaptive_is_AISTATS_2021.remainder import utils


def fully_corrective_weights_gd(target_logprob,
                                new_mean,
                                new_cov_sqrt,
                                weights,
                                means,
                                cov_sqrts,
                                num_samples=500,
                                num_iters=100,
                                tol=1e-6,
                                step_size=0.001,
                                init_gamma=0.6,
                                forward_kl=True,
                                eps=1e-6,
                                clip=False):
  """Inspired by the FW code. This minimizes the reverse kl."""
  assert len(means) == len(cov_sqrts) == len(weights)
  num_comps = len(weights) + 1
  print("Old Weights = {}".format(weights))
  print("Old Means = {}".format(means))
  print("Old cov_sqrts = {}".format(cov_sqrts))

  new_means = np.append(means, [new_mean], axis=0)
  new_cov_sqrts = np.append(cov_sqrts, [new_cov_sqrt], axis=0)

  samples = np.array([
      utils.mvn_sample(num_samples, mean, cov_sqrt)
      for (mean, cov_sqrt) in zip(new_means, new_cov_sqrts)
  ])
  print(
      "Shape of samples used by the Fully Corrective Weight search = {}".format(
          samples.shape))
  p_log_probs = np.array([target_logprob(samples[i]) for i in range(num_comps)])
  print("Weights before updating = {}".format(
      utils.logistic_transform(weights)))
  new_weights = utils.get_new_weights(weights, init_gamma)
  print(
      "Logits of Weights after adding gamma for init = {}".format(new_weights))
  new_weights = utils.logistic_transform(new_weights)
  print("Weights after adding gamma for init = {}".format(new_weights))

  for t in range(num_iters):
    grad = np.zeros(num_comps)
    for i in range(num_comps):
      q_log_probs = utils.mog_logprob(samples[i],
                                      utils.logit_transform(new_weights),
                                      new_means, new_cov_sqrts)
      if not forward_kl:
        diff = q_log_probs - p_log_probs[i]
      else:
        diff = -(np.exp(p_log_probs[i]) + eps) / (np.exp(q_log_probs) + eps)
      grad[i] = np.mean(diff, axis=0)  # take the expectation
    weights_update = new_weights - step_size * grad
    if t % 50 == 0:
      print("Grad at itr {0} is {1}".format(t, grad))
      print("Weights update before simplex projection = {}".format(
          weights_update))
    weights_update = opts.euclidean_proj_simplex(np.clip(weights_update, 0, 1))
    if t % 50 == 0:
      print(
          "Weights update AFTER simplex projection = {}".format(weights_update))
      new_for_kl, new_rev_kl = metrics.eval_mixture(
          target_logprob,
          utils.logit_transform(weights_update),
          new_means,
          new_cov_sqrts,
          eps=eps,
          clip=clip)
      print("New Forward KL = {0} and New Reverse KL = {1}".format(
          new_for_kl, new_rev_kl))
    delta = np.abs(new_weights - weights_update)
    if np.max(delta) < tol:
      print("Delta = {0} < {1} = tol".format(delta, tol))
      new_weights = weights_update.astype(np.float32)
      break
    new_weights = weights_update.astype(np.float32)
    if np.any(new_weights == 0):
      print("Stopping due to a ZERO weight : {}".format(new_weights))
      break
  if np.max(delta) > tol:
    print(
        "Reached maximum number of iterations {0} but the delta {1} > {2} : tol"
        .format(num_iters, np.max(delta), tol))
  print("Ended Fully Corrective with weights = {}".format(new_weights))
  if np.any(new_weights == 0):
    new_weights += eps
    new_weights /= new_weights.sum()
    print("Ended Fully Corrective with (corrected) weights = {}".format(
        new_weights))
  logit_weights = utils.logit_transform(new_weights)
  print("Ended Fully Corrective with logit_weights = {}".format(logit_weights))
  return logit_weights


def weights_line_search(target_logprob,
                        new_mean,
                        new_cov_sqrt,
                        weights,
                        means,
                        cov_sqrts,
                        num_samples=1000,
                        tol=1e-6,
                        num_iters=500,
                        init_gamma=0.5,
                        init_step_size=0.01):
  """Line search for the new component's mixture weight."""

  comp_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  old_mog_samples = utils.mog_sample(num_samples, weights, means, cov_sqrts)

  def old_mixture_logprob(theta):
    return utils.mog_logprob(theta, weights, means, cov_sqrts)

  gamma = init_gamma

  def get_new_weights(gamma):
    new_weights = weights * (1 - gamma)
    new_weights = np.append(new_weights, gamma).astype(np.float32)
    return new_weights

  new_weights, new_means, new_cov_sqrts = utils.update_mixture(
      weights, means, cov_sqrts, gamma, new_mean, new_cov_sqrt)
  init_for_kl, init_rev_kl = metrics.eval_mixture(target_logprob, new_weights,
                                                  new_means, new_cov_sqrts)
  print("The new Reverse KL is {0} and Forward KL is {1} for weights = {2}"
        .format(init_rev_kl, init_for_kl, new_weights))

  for t in range(num_iters):
    new_weights = get_new_weights(gamma)

    def new_mixture_logprob(samples):
      return utils.mog_logprob(samples, new_weights, new_means, new_cov_sqrts)

    new_comp_expectation = new_mixture_logprob(comp_samples) - target_logprob(
        comp_samples)
    old_mixture_expectation = new_mixture_logprob(
        old_mog_samples) - target_logprob(old_mog_samples)
    diff = new_comp_expectation - old_mixture_expectation
    # print("Shape of the difference in expectations = {}".format(diff.shape))
    grad = np.mean(diff)
    if t % 25 == 0:
      print("t", t, "gamma", gamma, "grad", grad)
      new_for_kl, new_rev_kl = metrics.eval_mixture(target_logprob, new_weights,
                                                    new_means, new_cov_sqrts)
      print("The new Reverse KL is {0} and Forward KL is {1}".format(
          new_rev_kl, new_for_kl))

    step_size = init_step_size / (t + 1)
    gamma_update = gamma - grad * step_size
    if gamma_update >= 1 or gamma_update <= 0:
      gamma_update = max(min(gamma_update, 1.), 0.)

    if np.abs(gamma - gamma_update) < tol:
      gamma = gamma_update
      break
    gamma = gamma_update

  if gamma < 1e-3:
    print(" |||| WARNING |||| : New component weight is insignificant ! ")
    gamma = 1e-3

  print("final t", t, "gamma", gamma, "grad", grad)
  return get_new_weights(gamma)


def autograd_fully_corrected(target_logprob,
                             new_mean,
                             new_cov_sqrt,
                             weights,
                             means,
                             cov_sqrts,
                             num_samples=500,
                             num_iters=100,
                             step_size=0.001,
                             init_gamma=0.6,
                             old_mixture=False,
                             use_adam=True,
                             stabilize_diff=True,
                             stabilize_weights=True,
                             eps=1e-6,
                             tol=1e-6):
  new_means = np.append(means, [new_mean], axis=0)
  new_cov_sqrts = np.append(cov_sqrts, [new_cov_sqrt], axis=0)
  new_weights = utils.get_new_weights(weights, init_gamma)

  if old_mixture:
    init_mixture_for_kl = autograd_utils.old_mixture_forward_kl_v2(
        new_weights,
        new_means,
        new_cov_sqrts,
        target_logprob,
        stabilize_diff=stabilize_diff,
        num_samples=num_samples,
        eps=eps,
        stabilize_weights=stabilize_weights)
  else:
    init_mixture_for_kl = autograd_utils.joint_mixture_mog_forward_kl(
        new_weights,
        new_means,
        new_cov_sqrts,
        target_logprob,
        stabilize_diff=stabilize_diff,
        num_samples=num_samples,
        eps=eps,
        stabilize_weights=stabilize_weights)

  if use_adam:
    weights_optimizer = opts.Adam(lr=step_size)

  objective = 1000
  delta = 1000
  for t in range(num_iters):
    if old_mixture:
      weights_grad = autograd_utils.old_mixture_forward_kl_v2_grad(
          new_weights,
          new_means,
          new_cov_sqrts,
          target_logprob,
          stabilize_diff=stabilize_diff,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      weights_grad = autograd_utils.joint_mixture_mog_forward_kl_weight_grad(
          new_weights,
          new_means,
          new_cov_sqrts,
          target_logprob,
          stabilize_diff=stabilize_diff,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)

    if use_adam:
      weights_update = np.array(
          weights_optimizer.get_update(new_weights, weights_grad))
    else:
      weights_update = new_weights - step_size * weights_grad

    if old_mixture:
      new_objective = autograd_utils.old_mixture_forward_kl_v2(
          new_weights,
          new_means,
          new_cov_sqrts,
          target_logprob,
          stabilize_diff=stabilize_diff,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    else:
      new_objective = autograd_utils.joint_mixture_mog_forward_kl(
          new_weights,
          new_means,
          new_cov_sqrts,
          target_logprob,
          stabilize_diff=stabilize_diff,
          num_samples=num_samples,
          eps=eps,
          stabilize_weights=stabilize_weights)
    if t % 100 == 0:
      print(
          "Autograd Weight Search: New Forward KL is {0} for weights = {1} means = {2} and cov_sqrts = {3} "
          .format(new_objective, utils.logistic_transform(weights_update)))

    delta = np.abs(objective - new_objective)
    objective = new_objective
    new_weights = weights_update

    if delta < tol:
      print("Delta = {0} < {1} : Tol for inner iteration {2}".format(
          delta, tol, t))
      break

  if delta > tol:
    print(
        ">>> Autograd Weight Search: DID NOT CONVERGE as the final delta is {0} > {1} : tolerance"
        .format(delta, tol))

  return new_weights
