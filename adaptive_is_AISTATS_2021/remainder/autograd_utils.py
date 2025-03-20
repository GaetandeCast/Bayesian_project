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

"""KL computation and AutoGrad gradients."""
from autograd import grad, hessian_vector_product
import autograd.numpy as np

from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import optimizers as opts
from adaptive_is_AISTATS_2021.remainder import utils


###############################    REVERSE KL    ###############################
# NOTE: do not use this unless certain of your finite differences.
def naive_mixture_reverse_kl(new_weights,
                             new_means,
                             new_cov_sqrts,
                             target_logprob,
                             num_samples=5000,
                             eps=1e-6):
  # WARNING : MOG_SAMPLE is not differentiable unlike the above appraoch : finite-differences could still work.
  samples = utils.mog_sample(num_samples, new_weights, new_means, new_cov_sqrts)
  q_log_probs = utils.mog_logprob(samples, new_weights, new_means,
                                  new_cov_sqrts)
  p_log_probs = target_logprob(samples)
  return np.mean(q_log_probs - p_log_probs)


# NOTE : Use this to differnetiate through any of ALL weights / Means / cov_sqrts.
def differentiable_mixture_reverse_kl(new_weights,
                                      new_means,
                                      new_cov_sqrts,
                                      target_logprob,
                                      num_samples=5000,
                                      eps=1e-6):
  samples = np.stack([
      utils.mvn_sample(num_samples, mean, cov_sqrt)
      for (mean, cov_sqrt) in zip(new_means, new_cov_sqrts)
  ],
                     axis=0)

  def proposal_logprob(theta):
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  p_log_probs = np.array([
      utils.logistic_transform(weight) * np.mean(target_logprob(sample))
      for (weight, sample) in zip(new_weights, samples)
  ])
  q_log_probs = np.array([
      utils.logistic_transform(weight) * np.mean(proposal_logprob(sample))
      for (weight, sample) in zip(new_weights, samples)
  ])
  return np.sum(q_log_probs - p_log_probs)


# NOTE : Use this to differentiate through Gamma, new_mean, or new_cov_sqrt [thus: "Single"]
def single_mixture_reverse_kl(gamma,
                              new_mean,
                              new_cov_sqrt,
                              target_logprob,
                              old_weights,
                              old_means,
                              old_cov_sqrts,
                              differentiable=True,
                              num_samples=5000,
                              eps=1e-6):
  # r_i = \gamma * q + (1 - \gamma) r_{i-1}
  # KL(r_i || p) = E_{r_i} [\log(r_i) - \log(p)] = \sum_{i} \gamma_i E_{h_i} [\log(r_i) - \log(p)]
  # or (bad approx) : ~ \gamma E_{q} [\log(r_i) - \log(p)] :
  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

  if differentiable:
    return differentiable_mixture_reverse_kl(new_weights, new_means,
                                             new_cov_sqrts, target_logprob,
                                             num_samples)
  else:
    return naive_mixture_reverse_kl(new_weights, new_means, new_cov_sqrts,
                                    target_logprob, num_samples)


# NOTE : Use this to differentiate through Gamma, new_mean, or new_cov_sqrt [thus: "Single"]
def alt_single_mixture_reverse_kl(gamma,
                                  new_mean,
                                  new_cov_sqrt,
                                  target_logprob,
                                  old_weights,
                                  old_means,
                                  old_cov_sqrts,
                                  num_samples=5000,
                                  bug=False,
                                  eps=1e-6):
  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

  # We don't care about differentiating through mog_sample in this case !
  if len(old_weights) > 0:
    old_mog_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                       old_cov_sqrts)
    if not bug:
      old_q_log_probs = utils.mog_logprob(old_mog_samples, new_weights,
                                          new_means, new_cov_sqrts)
    else:
      old_q_log_probs = utils.mog_logprob(old_mog_samples, old_weights,
                                          old_means, old_cov_sqrts)
    old_p_log_probs = target_logprob(old_mog_samples)
    old_diff = np.mean(old_q_log_probs - old_p_log_probs)
  else:
    old_diff = 0.

  comp_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  comp_q_log_probs = utils.mog_logprob(comp_samples, new_weights, new_means,
                                       new_cov_sqrts)
  comp_p_log_probs = target_logprob(comp_samples)
  comp_diff = np.mean(comp_q_log_probs - comp_p_log_probs)

  return gamma * comp_diff + (1 - gamma) * old_diff


# AutoGrad Remainder : this is the same as the RELBO (below) with regularization = 1
def remainder_reverse_kl(new_mean,
                         new_cov_sqrt,
                         target_logprob,
                         old_mixture_logprob,
                         stabilize_diff=True,
                         num_samples=5000,
                         eps=1e-6,
                         clip=False):
  samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  comp_log_probs = utils.mvn_logprob(samples, new_mean, new_cov_sqrt)
  q_log_probs = old_mixture_logprob(samples)  #+ eps
  p_log_probs = target_logprob(samples)  #+ eps
  residual_logprob = q_log_probs - p_log_probs
  if stabilize_diff:
    if clip:
      residual_logprob = -np.log(
          np.fmax(np.exp(p_log_probs), eps) / np.fmax(np.exp(q_log_probs), eps))
    else:
      residual_logprob = -np.log(
          (np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  return np.mean(comp_log_probs + residual_logprob)


# AutoGrad + Residual ELBO (+- regularization)
def residual_reverse_kl(new_mean,
                        new_cov_sqrt,
                        target_logprob,
                        proposal_logprob,
                        regularization=0.,
                        stabilize_diff=True,
                        num_samples=5000,
                        eps=1e-6,
                        clip=False):
  samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  q_log_probs = proposal_logprob(samples)
  p_log_probs = target_logprob(samples)
  residual_logprob = q_log_probs - p_log_probs
  if stabilize_diff:
    # residual_logprob = -np.log( (np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
    if clip:
      residual_logprob = -np.log(
          np.fmax(np.exp(p_log_probs), eps) / np.fmax(np.exp(q_log_probs), eps))
    else:
      residual_logprob = -np.log(
          (np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  if regularization != 0:
    residual_logprob = residual_logprob + regularization * utils.mvn_logprob(
        samples, new_mean, new_cov_sqrt)
  return np.mean(residual_logprob)


############################ REVERSE KL GRADIENTS ##############################
# (single) Gradient wrt new_mean / new_cov_sqrt for residual / remainder
def remainder_rev_kl_grad(new_mean,
                          new_cov_sqrt,
                          target_logprob,
                          old_mixture_logprob,
                          stabilize_diff=True,
                          num_samples=5000,
                          eps=1e-6,
                          clip=False):
  mean_grad_fn = grad(remainder_reverse_kl, 0)
  cov_sqrt_grad_fn = grad(remainder_reverse_kl, 1)
  mean_grad = mean_grad_fn(new_mean, new_cov_sqrt, target_logprob,
                           old_mixture_logprob, stabilize_diff, num_samples,
                           eps, clip)
  cov_sqrt_grad = cov_sqrt_grad_fn(new_mean, new_cov_sqrt, target_logprob,
                                   old_mixture_logprob, stabilize_diff,
                                   num_samples, eps, clip)
  return mean_grad, cov_sqrt_grad


def residual_rev_kl_grad(new_mean,
                         new_cov_sqrt,
                         target_logprob,
                         proposal_logprob,
                         regularization=0.,
                         stabilize_diff=True,
                         num_samples=5000,
                         eps=1e-6,
                         clip=False):
  mean_grad_fn = grad(residual_reverse_kl, 0)
  cov_sqrt_grad_fn = grad(residual_reverse_kl, 1)
  mean_grad = mean_grad_fn(new_mean, new_cov_sqrt, target_logprob,
                           proposal_logprob, regularization, stabilize_diff,
                           num_samples, eps, clip)
  cov_sqrt_grad = cov_sqrt_grad_fn(new_mean, new_cov_sqrt, target_logprob,
                                   proposal_logprob, regularization,
                                   stabilize_diff, num_samples, eps, clip)
  return mean_grad, cov_sqrt_grad


########
# (single) Gradient wrt gamma / new_cov_sqrt / new_mean
def mixture_rev_kl_grad(gamma,
                        new_mean,
                        new_cov_sqrt,
                        target_logprob,
                        old_weights,
                        old_means,
                        old_cov_sqrts,
                        differentiable=True,
                        num_samples=5000,
                        eps=1e-6):
  gamma_grad_fn = grad(single_mixture_reverse_kl, 0)
  mean_grad_fn = grad(single_mixture_reverse_kl, 1)
  cov_sqrt_grad_fn = grad(single_mixture_reverse_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             differentiable, num_samples)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           differentiable, num_samples)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, differentiable, num_samples)
  return gamma_grad, mean_grad, cov_sqrt_grad


def alt_mixture_rev_kl_grad(gamma,
                            new_mean,
                            new_cov_sqrt,
                            target_logprob,
                            old_weights,
                            old_means,
                            old_cov_sqrts,
                            num_samples=5000,
                            bug=False,
                            eps=1e-6):
  gamma_grad_fn = grad(alt_single_mixture_reverse_kl, 0)
  mean_grad_fn = grad(alt_single_mixture_reverse_kl, 1)
  cov_sqrt_grad_fn = grad(alt_single_mixture_reverse_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts, num_samples,
                             bug)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts, num_samples,
                           bug)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, num_samples, bug)
  return gamma_grad, mean_grad, cov_sqrt_grad


########
# (joint) Gradient wrt weights / cov_sqrts / mewns
def full_mixture_rev_kl_grad(new_weights,
                             new_means,
                             new_cov_sqrts,
                             target_logprob,
                             num_samples=5000,
                             eps=1e-6):
  weights_grad_fn = grad(differentiable_mixture_reverse_kl, 0)
  means_grad_fn = grad(differentiable_mixture_reverse_kl, 1)
  cov_sqrts_grad_fn = grad(differentiable_mixture_reverse_kl, 2)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, num_samples)
  means_grad = means_grad_fn(new_weights, new_means, new_cov_sqrts,
                             target_logprob, num_samples)
  cov_sqrts_grad = cov_sqrts_grad_fn(new_weights, new_means, new_cov_sqrts,
                                     target_logprob, num_samples)
  return weights_grad, means_grad, cov_sqrts_grad


################################  FORWARD KL  ##################################
# KL(p || r_i) = E_p [\log(p) - \log(r)]
# = E_q [(p/q) (\log(p) - \log(r))]
def joint_mixture_mvn_forward_kl(new_weights,
                                 new_means,
                                 new_cov_sqrts,
                                 target_logprob,
                                 stabilize_diff=True,
                                 num_samples=5000,
                                 eps=1e-6,
                                 stabilize_weights=False):
  new_mean = new_means[-1]
  new_cov_sqrt = new_cov_sqrts[-1]
  samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)

  def is_proposal_logprob(theta):  #q
    return utils.mvn_logprob(theta, new_mean, new_cov_sqrt)

  def mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  sampling_weights = metrics.snis_weights(
      samples,
      target_logprob,
      is_proposal_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  q_log_probs = mixture_logprob(
      samples)  #this is actually r in the above Latex notation
  p_log_probs = target_logprob(samples)
  diff = p_log_probs - q_log_probs

  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


# KL(p || r_i) = E_p [\log(p) - \log(r)]
# = E_r [(p/r) (\log(p) - \log(r))]
def joint_mixture_mog_forward_kl(new_weights,
                                 new_means,
                                 new_cov_sqrts,
                                 target_logprob,
                                 stabilize_diff=True,
                                 num_samples=5000,
                                 eps=1e-6,
                                 stabilize_weights=False):
  num_comps = len(new_weights)

  def mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  samples = np.stack([
      utils.mvn_sample(num_samples, new_means[i], new_cov_sqrts[i])
      for i in range(num_comps)
  ],
                     axis=0)
  p_log_probs = np.stack([
      utils.logistic_transform(new_weights[i]) * target_logprob(samples[i])
      for i in range(num_comps)
  ],
                         axis=0)
  q_log_probs = np.stack([
      utils.logistic_transform(new_weights[i]) * mixture_logprob(samples[i])
      for i in range(num_comps)
  ],
                         axis=0)
  sampling_weights = np.stack([
      metrics.snis_weights(
          samples[i],
          target_logprob,
          mixture_logprob,
          eps=eps,
          stabilize_weights=stabilize_weights) for i in range(num_comps)
  ],
                              axis=0)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  # return np.sum(integrand)
  return np.sum(np.mean(integrand, axis=-1))


# USE this : NEVER unless you're confident of differentiating through argmax ...
def naive_mixture_mog_forward_kl(new_weights,
                                 new_means,
                                 new_cov_sqrts,
                                 target_logprob,
                                 stabilize_diff=True,
                                 num_samples=5000,
                                 eps=1e-6,
                                 stabilize_weights=False):
  samples = utils.mog_sample(num_samples, new_weights, new_means, new_cov_sqrts)

  def mixture_logprob(theta):
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  sampling_weights = metrics.snis_weights(
      samples,
      target_logprob,
      mixture_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  q_log_probs = mixture_logprob(samples)
  p_log_probs = target_logprob(samples)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)

  return np.sum(integrand)


def single_mixture_forward_kl(gamma,
                              new_mean,
                              new_cov_sqrt,
                              target_logprob,
                              old_weights,
                              old_means,
                              old_cov_sqrts,
                              gaussian_sample=False,
                              stabilize_diff=True,
                              num_samples=5000,
                              eps=1e-6,
                              stabilize_weights=False):
  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)
  return joint_mixture_mog_forward_kl(new_weights, new_means, new_cov_sqrts,
                                      target_logprob, stabilize_diff,
                                      num_samples, eps, stabilize_weights)


########################### FORWARD GRADS ######################################


# joint gradients : new_weights, new_means, new_cov_sqrts
def joint_mixture_mvn_forward_kl_grad(new_weights,
                                      new_means,
                                      new_cov_sqrts,
                                      target_logprob,
                                      stabilize_diff=True,
                                      num_samples=5000,
                                      eps=1e-6,
                                      stabilize_weights=False):
  weights_grad_fn = grad(joint_mixture_mvn_forward_kl, 0)
  means_grad_fn = grad(joint_mixture_mvn_forward_kl, 1)
  cov_sqrts_grad_fn = grad(joint_mixture_mvn_forward_kl, 2)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  means_grad = means_grad_fn(new_weights, new_means, new_cov_sqrts,
                             target_logprob, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  cov_sqrts_grad = cov_sqrts_grad_fn(new_weights, new_means, new_cov_sqrts,
                                     target_logprob, stabilize_diff,
                                     num_samples, eps, stabilize_weights)
  return weights_grad, means_grad, cov_sqrts_grad


def joint_mixture_mog_forward_kl_grad(new_weights,
                                      new_means,
                                      new_cov_sqrts,
                                      target_logprob,
                                      stabilize_diff=True,
                                      num_samples=5000,
                                      eps=1e-6,
                                      stabilize_weights=False):
  weights_grad_fn = grad(joint_mixture_mog_forward_kl, 0)
  means_grad_fn = grad(joint_mixture_mog_forward_kl, 1)
  cov_sqrts_grad_fn = grad(joint_mixture_mog_forward_kl, 2)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  means_grad = means_grad_fn(new_weights, new_means, new_cov_sqrts,
                             target_logprob, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  cov_sqrts_grad = cov_sqrts_grad_fn(new_weights, new_means, new_cov_sqrts,
                                     target_logprob, stabilize_diff,
                                     num_samples, eps, stabilize_weights)
  return weights_grad, means_grad, cov_sqrts_grad


def naive_mixture_mog_forward_kl_grad(new_weights,
                                      new_means,
                                      new_cov_sqrts,
                                      target_logprob,
                                      num_samples=5000,
                                      stabilize_diff=True,
                                      eps=1e-6,
                                      stabilize_weights=False):
  weights_grad_fn = grad(naive_mixture_mog_forward_kl, 0)
  means_grad_fn = grad(naive_mixture_mog_forward_kl, 1)
  cov_sqrts_grad_fn = grad(naive_mixture_mog_forward_kl, 2)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  means_grad = means_grad_fn(new_weights, new_means, new_cov_sqrts,
                             target_logprob, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  cov_sqrts_grad = cov_sqrts_grad_fn(new_weights, new_means, new_cov_sqrts,
                                     target_logprob, stabilize_diff,
                                     num_samples, eps, stabilize_weights)
  return weights_grad, means_grad, cov_sqrts_grad


def joint_old_mixture_forward_kl(new_weights,
                                 new_means,
                                 new_cov_sqrts,
                                 target_logprob,
                                 stabilize_diff=True,
                                 num_samples=5000,
                                 eps=1e-6,
                                 stabilize_weights=False):
  old_weights = utils.renormalize_logit_weights(new_weights[:-1], eps=eps)
  old_means = new_means[:-1]
  old_cov_sqrts = new_cov_sqrts[:-1]
  num_old_comps = len(old_weights)

  def old_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  def new_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  old_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                 old_cov_sqrts)
  sampling_weights = metrics.snis_weights(
      old_samples,
      target_logprob,
      old_mixture_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  p_log_probs = target_logprob(old_samples)
  q_log_probs = new_mixture_logprob(old_samples)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


# This one samples from the old mixture : replaces JOINT Mixture Forward KL. This can also be used for Single !
def old_mixture_forward_kl(gamma,
                           new_mean,
                           new_cov_sqrt,
                           target_logprob,
                           old_weights,
                           old_means,
                           old_cov_sqrts,
                           stabilize_diff=True,
                           num_samples=5000,
                           eps=1e-6,
                           stabilize_weights=False):
  num_old_comps = len(old_weights)

  def old_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

  def new_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  if num_old_comps > 0:
    old_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                   old_cov_sqrts)
    sampling_weights = metrics.snis_weights(
        old_samples,
        target_logprob,
        old_mixture_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    p_log_probs = target_logprob(old_samples)
    q_log_probs = new_mixture_logprob(old_samples)
  else:
    old_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
    sampling_weights = metrics.snis_weights(
        old_samples,
        target_logprob,
        new_mixture_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    p_log_probs = target_logprob(old_samples)
    q_log_probs = new_mixture_logprob(old_samples)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


# This one samples from the old mixture too
def alt_old_mixture_forward_kl(gamma,
                               new_mean,
                               new_cov_sqrt,
                               target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               gaussian_sample=False,
                               stabilize_diff=True,
                               num_samples=5000,
                               eps=1e-6,
                               stabilize_weights=False):
  num_old_comps = len(old_weights)

  def old_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

  def new_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  samples = np.stack([
      utils.mvn_sample(num_samples, old_means[i], old_cov_sqrts[i])
      for i in range(num_old_comps)
  ],
                     axis=0)
  p_log_probs = np.stack([
      old_weights[i] * target_logprob(samples[i]) for i in range(num_old_comps)
  ],
                         axis=0)
  q_log_probs = np.stack([
      old_weights[i] * new_mixture_logprob(samples[i])
      for i in range(num_old_comps)
  ],
                         axis=0)
  sampling_weights = np.stack([
      metrics.snis_weights(
          samples[i],
          target_logprob,
          old_mixture_logprob,
          eps=eps,
          stabilize_weights=stabilize_weights) for i in range(num_old_comps)
  ],
                              axis=0)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


def alt_single_mixture_forward_kl(gamma,
                                  new_mean,
                                  new_cov_sqrt,
                                  target_logprob,
                                  old_weights,
                                  old_means,
                                  old_cov_sqrts,
                                  stabilize_diff=True,
                                  num_samples=5000,
                                  eps=1e-6,
                                  stabilize_weights=False):

  def new_comp_logprob(theta):
    return utils.mvn_logprob(theta, new_mean, new_cov_sqrt)

  def old_mog_logprob(theta):
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)
  if len(old_weights) > 0:
    old_mog_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                       old_cov_sqrts)
    old_q_log_probs = utils.mog_logprob(old_mog_samples, new_weights, new_means,
                                        new_cov_sqrts)
    old_p_log_probs = target_logprob(old_mog_samples)
    mog_sampling_weights = metrics.snis_weights(
        old_mog_samples,
        target_logprob,
        old_mog_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    diff = old_p_log_probs - old_q_log_probs
    if stabilize_diff:
      diff = np.log(
          (np.exp(old_p_log_probs) + eps) / (np.exp(old_q_log_probs) + eps))
    old_diff = np.multiply(mog_sampling_weights, diff)
  else:
    old_diff = 0.

  comp_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  comp_q_log_probs = utils.mog_logprob(comp_samples, new_weights, new_means,
                                       new_cov_sqrts)
  comp_p_log_probs = target_logprob(comp_samples)
  comp_sampling_weights = metrics.snis_weights(
      comp_samples,
      target_logprob,
      new_comp_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  diff = comp_p_log_probs - comp_q_log_probs
  if stabilize_diff:
    diff = np.log(
        (np.exp(comp_p_log_probs) + eps) / (np.exp(comp_q_log_probs) + eps))
  comp_diff = np.multiply(comp_sampling_weights, diff)
  return gamma * np.sum(comp_diff) + (1 - gamma) * np.sum(old_diff)


# This one is the same as the above but changes the SNIS weights
def alt_new_mog_forward_kl(gamma,
                           new_mean,
                           new_cov_sqrt,
                           target_logprob,
                           old_weights,
                           old_means,
                           old_cov_sqrts,
                           stabilize_diff=True,
                           num_samples=5000,
                           eps=1e-6,
                           stabilize_weights=False):

  def new_comp_logprob(theta):
    return utils.mvn_logprob(theta, new_mean, new_cov_sqrt)

  def old_mog_logprob(theta):
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
      old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

  def new_mog_logprob(theta):
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  if len(old_weights) > 0:
    old_mog_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                       old_cov_sqrts)
    old_q_log_probs = new_mog_logprob(old_mog_samples)
    old_p_log_probs = target_logprob(old_mog_samples)
    mog_sampling_weights = metrics.snis_weights(
        old_mog_samples,
        target_logprob,
        new_mog_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    diff = old_p_log_probs - old_q_log_probs
    if stabilize_diff:
      diff = np.log(
          (np.exp(old_p_log_probs) + eps) / (np.exp(old_q_log_probs) + eps))
    old_diff = np.multiply(mog_sampling_weights, diff)
  else:
    old_diff = 0.

  comp_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  comp_q_log_probs = new_mog_logprob(comp_samples)
  comp_p_log_probs = target_logprob(comp_samples)
  comp_sampling_weights = metrics.snis_weights(
      comp_samples,
      target_logprob,
      new_mog_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  diff = comp_p_log_probs - comp_q_log_probs
  if stabilize_diff:
    diff = np.log(
        (np.exp(comp_p_log_probs) + eps) / (np.exp(comp_q_log_probs) + eps))
  comp_diff = np.multiply(comp_sampling_weights, diff)
  return gamma * np.sum(comp_diff) + (1 - gamma) * np.sum(old_diff)


# This one samples from the old mixture instead of the new mixture
def alt_alt_old_mog_forward_kl(gamma,
                               new_mean,
                               new_cov_sqrt,
                               target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               stabilize_diff=True,
                               num_samples=5000,
                               eps=1e-6,
                               stabilize_weights=False):

  def new_comp_logprob(theta):
    return utils.mvn_logprob(theta, new_mean, new_cov_sqrt)

  def old_mog_logprob(theta):
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  if len(old_weights) > 0:
    new_weights, new_means, new_cov_sqrts = utils.update_mixture_aslists(
        old_weights, old_means, old_cov_sqrts, gamma, new_mean, new_cov_sqrt)

    def new_mog_logprob(theta):
      return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

    old_mog_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                       old_cov_sqrts)
    old_q_log_probs = new_mog_logprob(old_mog_samples)
    old_p_log_probs = target_logprob(old_mog_samples)
    mog_sampling_weights = metrics.snis_weights(
        old_mog_samples,
        target_logprob,
        old_mog_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    diff = old_p_log_probs - old_q_log_probs
    if stabilize_diff:
      diff = np.log(
          (np.exp(old_p_log_probs) + eps) / (np.exp(old_q_log_probs) + eps))
    return np.sum(np.multiply(mog_sampling_weights, diff))
  else:
    comp_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
    comp_q_log_probs = new_comp_logprob(comp_samples)
    comp_p_log_probs = target_logprob(comp_samples)
    comp_sampling_weights = metrics.snis_weights(
        comp_samples,
        target_logprob,
        new_comp_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    diff = comp_p_log_probs - comp_q_log_probs
    if stabilize_diff:
      diff = np.log(
          (np.exp(comp_p_log_probs) + eps) / (np.exp(comp_q_log_probs) + eps))
    return gamma * np.sum(np.multiply(comp_sampling_weights, diff))


def remainder_forward_kl(new_mean,
                         new_cov_sqrt,
                         target_logprob,
                         old_weights,
                         old_means,
                         old_cov_sqrts,
                         stabilize_diff=True,
                         num_samples=5000,
                         eps=1e-6,
                         stabilize_weights=False):

  def old_mixture_logprob(theta):
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  def new_target_logprob(theta):
    if stabilize_diff:
      return np.log((np.exp(target_logprob(theta)) + eps) /
                    (np.exp(old_mixture_logprob(theta)) + eps))
    else:
      return target_logprob(theta) - old_mixture_logprob(theta)

  def proposal_logprob(theta):
    return utils.mvn_logprob(theta, new_mean, new_cov_sqrt)

  samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
  sampling_weights = metrics.snis_weights(
      samples,
      new_target_logprob,
      proposal_logprob,
      eps=eps,
      stabilize_weights=stabilize_weights)
  q_log_probs = proposal_logprob(samples)
  p_log_probs = new_target_logprob(
      samples
  )  # remainder sort of target: should be just the target at the start !
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


###########################    FORWARD KL GRADS   ##############################
# single gradients : new_mean, new_cov_sqrt (residual)
def remainder_forward_kl_grad(new_mean,
                              new_cov_sqrt,
                              target_logprob,
                              old_weights,
                              old_means,
                              old_cov_sqrts,
                              num_samples=5000,
                              stabilize_diff=True,
                              eps=1e-6,
                              stabilize_weights=False):
  mean_grad_fn = grad(remainder_forward_kl, 0)
  cov_sqrt_grad_fn = grad(remainder_forward_kl, 1)
  mean_grad = mean_grad_fn(new_mean, new_cov_sqrt, target_logprob, old_weights,
                           old_means, old_cov_sqrts, stabilize_diff,
                           num_samples, eps, stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(new_mean, new_cov_sqrt, target_logprob,
                                   old_weights, old_means, old_cov_sqrts,
                                   stabilize_diff, num_samples, eps,
                                   stabilize_weights)
  return mean_grad, cov_sqrt_grad


# single gradients : gamma, new_mean, new_cov_sqrt
def single_forward_kl_grad(gamma,
                           new_mean,
                           new_cov_sqrt,
                           target_logprob,
                           old_weights,
                           old_means,
                           old_cov_sqrts,
                           gaussian_sample=False,
                           stabilize_diff=True,
                           num_samples=5000,
                           eps=1e-6,
                           stabilize_weights=False):
  gamma_grad_fn = grad(single_mixture_forward_kl, 0)
  mean_grad_fn = grad(single_mixture_forward_kl, 1)
  cov_sqrt_grad_fn = grad(single_mixture_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             gaussian_sample, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           gaussian_sample, stabilize_diff, num_samples, eps,
                           stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, gaussian_sample,
                                   stabilize_diff, num_samples, eps,
                                   stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def alt_single_forward_kl_grad(gamma,
                               new_mean,
                               new_cov_sqrt,
                               target_logprob,
                               old_weights,
                               old_means,
                               old_cov_sqrts,
                               stabilize_diff=True,
                               num_samples=5000,
                               eps=1e-6,
                               stabilize_weights=False):
  gamma_grad_fn = grad(alt_single_mixture_forward_kl, 0)
  mean_grad_fn = grad(alt_single_mixture_forward_kl, 1)
  cov_sqrt_grad_fn = grad(alt_single_mixture_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           stabilize_diff, num_samples, eps, stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, stabilize_diff, num_samples,
                                   eps, stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def old_mixture_forward_kl_grad(gamma,
                                new_mean,
                                new_cov_sqrt,
                                target_logprob,
                                old_weights,
                                old_means,
                                old_cov_sqrts,
                                stabilize_diff=True,
                                num_samples=5000,
                                eps=1e-6,
                                stabilize_weights=False):
  gamma_grad_fn = grad(old_mixture_forward_kl, 0)
  mean_grad_fn = grad(old_mixture_forward_kl, 1)
  cov_sqrt_grad_fn = grad(old_mixture_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           stabilize_diff, num_samples, eps, stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, stabilize_diff, num_samples,
                                   eps, stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def alt_old_mixture_forward_kl_grad(gamma,
                                    new_mean,
                                    new_cov_sqrt,
                                    target_logprob,
                                    old_weights,
                                    old_means,
                                    old_cov_sqrts,
                                    gaussian_sample=False,
                                    stabilize_diff=True,
                                    num_samples=5000,
                                    eps=1e-6,
                                    stabilize_weights=False):
  gamma_grad_fn = grad(alt_old_mixture_forward_kl, 0)
  mean_grad_fn = grad(alt_old_mixture_forward_kl, 1)
  cov_sqrt_grad_fn = grad(alt_old_mixture_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             gaussian_sample, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           gaussian_sample, stabilize_diff, num_samples, eps,
                           stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, gaussian_sample,
                                   stabilize_diff, num_samples, eps,
                                   stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def alt_new_mog_forward_kl_grad(gamma,
                                new_mean,
                                new_cov_sqrt,
                                target_logprob,
                                old_weights,
                                old_means,
                                old_cov_sqrts,
                                stabilize_diff=True,
                                num_samples=5000,
                                eps=1e-6,
                                stabilize_weights=False):
  gamma_grad_fn = grad(alt_new_mog_forward_kl, 0)
  mean_grad_fn = grad(alt_new_mog_forward_kl, 1)
  cov_sqrt_grad_fn = grad(alt_new_mog_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           stabilize_diff, num_samples, eps, stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, stabilize_diff, num_samples,
                                   eps, stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def alt_alt_old_mog_forward_kl_grad(gamma,
                                    new_mean,
                                    new_cov_sqrt,
                                    target_logprob,
                                    old_weights,
                                    old_means,
                                    old_cov_sqrts,
                                    stabilize_diff=True,
                                    num_samples=5000,
                                    eps=1e-6,
                                    stabilize_weights=False):
  gamma_grad_fn = grad(alt_alt_old_mog_forward_kl, 0)
  mean_grad_fn = grad(alt_alt_old_mog_forward_kl, 1)
  cov_sqrt_grad_fn = grad(alt_alt_old_mog_forward_kl, 2)
  gamma_grad = gamma_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                             old_weights, old_means, old_cov_sqrts,
                             stabilize_diff, num_samples, eps,
                             stabilize_weights)
  mean_grad = mean_grad_fn(gamma, new_mean, new_cov_sqrt, target_logprob,
                           old_weights, old_means, old_cov_sqrts,
                           stabilize_diff, num_samples, eps, stabilize_weights)
  cov_sqrt_grad = cov_sqrt_grad_fn(gamma, new_mean, new_cov_sqrt,
                                   target_logprob, old_weights, old_means,
                                   old_cov_sqrts, stabilize_diff, num_samples,
                                   eps, stabilize_weights)
  return gamma_grad, mean_grad, cov_sqrt_grad


def joint_old_mixture_forward_kl_grad(new_weights,
                                      new_means,
                                      new_cov_sqrts,
                                      target_logprob,
                                      stabilize_diff=True,
                                      num_samples=5000,
                                      eps=1e-6,
                                      stabilize_weights=False):
  weights_grad_fn = grad(joint_old_mixture_forward_kl, 0)
  means_grad_fn = grad(joint_old_mixture_forward_kl, 1)
  cov_sqrts_grad_fn = grad(joint_old_mixture_forward_kl, 2)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  means_grad = means_grad_fn(new_weights, new_means, new_cov_sqrts,
                             target_logprob, stabilize_diff, num_samples, eps,
                             stabilize_weights)
  cov_sqrts_grad = cov_sqrts_grad_fn(new_weights, new_means, new_cov_sqrts,
                                     target_logprob, stabilize_diff,
                                     num_samples, eps, stabilize_weights)
  return weights_grad, means_grad, cov_sqrts_grad


#######################  Weight Search Utils ###################################


def joint_mixture_mog_forward_kl_weight_grad(new_weights,
                                             new_means,
                                             new_cov_sqrts,
                                             target_logprob,
                                             stabilize_diff=True,
                                             num_samples=5000,
                                             eps=1e-6,
                                             stabilize_weights=False):
  weights_grad_fn = grad(joint_mixture_mog_forward_kl, 0)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  return weights_grad


def old_mixture_forward_kl_v2(new_weights,
                              new_means,
                              new_cov_sqrts,
                              target_logprob,
                              stabilize_diff=True,
                              num_samples=5000,
                              eps=1e-6,
                              stabilize_weights=False):
  old_weights = utils.renormalize_logit_weights(new_weights[:-1], eps=eps)
  old_means = new_means[:-1]
  old_cov_sqrts = new_cov_sqrts[:-1]
  num_old_comps = len(old_weights)
  new_mean = new_means[-1]
  new_cov_sqrt = new_cov_sqrts[-1]

  def old_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, old_weights, old_means, old_cov_sqrts)

  def new_mixture_logprob(theta):  #r
    return utils.mog_logprob(theta, new_weights, new_means, new_cov_sqrts)

  if num_old_comps > 0:
    old_samples = utils.mog_sample(num_samples, old_weights, old_means,
                                   old_cov_sqrts)
    sampling_weights = metrics.snis_weights(
        old_samples,
        target_logprob,
        old_mixture_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    p_log_probs = target_logprob(old_samples)
    q_log_probs = new_mixture_logprob(old_samples)
  else:
    old_samples = utils.mvn_sample(num_samples, new_mean, new_cov_sqrt)
    sampling_weights = metrics.snis_weights(
        old_samples,
        target_logprob,
        new_mixture_logprob,
        eps=eps,
        stabilize_weights=stabilize_weights)
    p_log_probs = target_logprob(old_samples)
    q_log_probs = new_mixture_logprob(old_samples)
  diff = p_log_probs - q_log_probs
  if stabilize_diff:
    diff = np.log((np.exp(p_log_probs) + eps) / (np.exp(q_log_probs) + eps))
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


def old_mixture_forward_kl_v2_grad(new_weights,
                                   new_means,
                                   new_cov_sqrts,
                                   target_logprob,
                                   stabilize_diff=True,
                                   num_samples=5000,
                                   eps=1e-6,
                                   stabilize_weights=False):
  weights_grad_fn = grad(old_mixture_forward_kl_v2, 0)
  weights_grad = weights_grad_fn(new_weights, new_means, new_cov_sqrts,
                                 target_logprob, stabilize_diff, num_samples,
                                 eps, stabilize_weights)
  return weights_grad


def negative_remainder(theta,
                       target_logprob,
                       proposal_logprob,
                       eps=1e-6,
                       stabilize_remainder=False):
  theta = np.expand_dims(theta, 0)
  q = proposal_logprob(theta)
  p = target_logprob(theta)
  # print('>>> q = {} | p = {}, q-p = {}'.format(q, p, q - p))
  if stabilize_remainder:
    target_pdf = np.exp(p) + eps
    proposal_pdf = np.exp(q) + eps
    return -np.log(target_pdf / proposal_pdf)
  return q - p


def neg_remainder_grad(theta,
                       target_logprob,
                       proposal_logprob,
                       eps=1e-6,
                       stabilize_remainder=False):
  remainder_grad_fn = grad(negative_remainder, 0)
  return remainder_grad_fn(theta, target_logprob, proposal_logprob, eps,
                           stabilize_remainder)


def adam_map(target_logprob,
             proposal_logprob,
             d=1,
             init_theta=0.,
             init_stddev=0.3,
             lr=0.01,
             max_iter=500,
             tol=1e-10,
             eps=1e-6,
             diagonal=False,
             stabilize_remainder=True):
  if d == 1:
    theta_list = np.linspace(-10, 10, num=max_iter)
    q = proposal_logprob(theta_list)
    p = target_logprob(theta_list)
    remainder = (np.exp(q) + eps) / (np.exp(p) + eps)
    min_residual_idx = np.argmin(remainder)
    min_theta = theta_list[min_residual_idx].reshape((d,))
    min_residual = remainder[min_residual_idx]
    print('Lowest residual is {} for theta = {}'.format(min_residual,
                                                        min_theta))
    return min_theta

  theta = np.random.multivariate_normal(init_theta, init_stddev)
  cur_remainder = negative_remainder(theta, target_logprob, proposal_logprob,
                                     eps)
  print('cur_remainder = {} for theta = {}'.format(cur_remainder, theta))
  optimizer = opts.Adam(lr=lr)
  delta = 100
  for t in range(max_iter):
    grad = neg_remainder_grad(theta, target_logprob, proposal_logprob, eps,
                              stabilize_remainder)
    new_theta = np.array(optimizer.get_update(theta, grad))
    new_remainder = negative_remainder(new_theta, target_logprob,
                                       proposal_logprob, eps,
                                       stabilize_remainder)
    theta = new_theta
    delta = np.abs(new_remainder - cur_remainder)
    cur_remainder = new_remainder
    if delta < tol and t > 100:
      print(
          'Delta = {} < {} : Tol for inner iteration {} and new_remainder = {}'
          .format(delta, tol, t, new_remainder))
      break
  if delta > tol:
    print(
        ' Might have not converged at itr {} to the mode since the final delta is {} > {} : tolerance'
        .format(t, delta, tol))
  return theta
