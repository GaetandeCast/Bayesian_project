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
import autograd.scipy as scipy
import scipy.stats as scpst

from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import utils


def reverse_kl(samples, target_logprob, proposal_logprob):
  q_log_prob = proposal_logprob(samples)
  p_log_prob = target_logprob(samples)
  return np.mean(q_log_prob - p_log_prob)


def snis_weights(samples,
                 target_logprob,
                 proposal_logprob,
                 eps=1e-6,
                 clip=False,
                 stabilize_weights=False,
                 truncated=False):
  q_log_prob = proposal_logprob(samples)
  p_log_prob = target_logprob(samples)
  if stabilize_weights:
    sampling_weights = (np.exp(p_log_prob) + eps) / (np.exp(q_log_prob) + eps)
  elif truncated:
    original_weights = np.exp(p_log_prob - q_log_prob)
    average_weight = np.mean(original_weights)
    sampling_weights = np.minimum(original_weights,
                                  average_weight * np.sqrt(len(samples)))
  else:
    sampling_weights = np.exp(p_log_prob - q_log_prob) + eps

  sampling_weights /= np.sum(sampling_weights)
  return sampling_weights


# Computed with SNIS for KL (new component || remainder)
def forward_kl(samples,
               target_logprob,
               proposal_logprob,
               eps=1e-6,
               clip=False,
               stabilize_weights=False):
  q_log_prob = proposal_logprob(samples)
  p_log_prob = target_logprob(samples)
  sampling_weights = snis_weights(
      samples,
      target_logprob,
      proposal_logprob,
      eps=eps,
      clip=clip,
      stabilize_weights=stabilize_weights)
  diff = p_log_prob - q_log_prob
  integrand = np.multiply(sampling_weights, diff)
  return np.sum(integrand)


# KL (new_component || remainder) or KL (remainder || new_component)
def eval_component(target_logprob,
                   mean,
                   cov_sqrt,
                   num_samples=1000,
                   eps=1e-6,
                   clip=False,
                   stabilize_weights=False):
  print('Eval Component with cov_sqrt = {0} and Mean = {1}'.format(
      cov_sqrt, mean))
  samples = utils.mvn_sample(num_samples, mean, cov_sqrt)

  def proposal_logprob(x):
    return utils.mvn_logprob(x, mean, cov_sqrt)

  for_kl = forward_kl(
      samples,
      target_logprob,
      proposal_logprob,
      eps=eps,
      clip=clip,
      stabilize_weights=stabilize_weights)
  rev_kl = reverse_kl(samples, target_logprob, proposal_logprob)
  return for_kl, rev_kl


def eval_mixture(target_logprob,
                 weights,
                 means,
                 cov_sqrts,
                 num_samples=5000,
                 eps=1e-6,
                 clip=False,
                 stabilize_weights=False):
  samples = utils.mog_sample(num_samples, weights, means, cov_sqrts)

  def proposal_logprob(x):
    return utils.mog_logprob(x, weights, means, cov_sqrts)

  for_kl = forward_kl(
      samples,
      target_logprob,
      proposal_logprob,
      eps=eps,
      clip=clip,
      stabilize_weights=stabilize_weights)
  rev_kl = reverse_kl(samples, target_logprob, proposal_logprob)
  return for_kl, rev_kl


def eval_importance_weights(samples,
                            target_logprob,
                            proposal_logprob,
                            n=5000,
                            eps=1e-6,
                            clip=False,
                            stabilize_weights=False):
  is_w = snis_weights(
      samples,
      target_logprob,
      proposal_logprob,
      eps=eps,
      clip=clip,
      stabilize_weights=stabilize_weights)
  ess = np.power(np.sum(is_w), 2) / np.sum(np.power(is_w, 2))
  avg_w = np.mean(is_w)
  cv = np.sqrt(np.sum(np.power(is_w - avg_w, 2)) / (n - 1.)) / avg_w
  return ess, cv


def empirical_moments(samples,
                      target_logprob=None,
                      proposal_logprob=None,
                      max_moment=2,
                      eps=1e-6):
  sample_mean = np.mean(samples, axis=0)
  centered_moments = np.array([
      np.mean(np.power(samples - sample_mean, m), axis=0)
      for m in range(1, max_moment + 1)
  ])
  snis_weighted_moments = None
  if target_logprob is not None:
    assert proposal_logprob is not None
    self_normalized_weights = snis_weights(
        samples, target_logprob, proposal_logprob, eps=eps)
    snis_weighted_moments = np.array([
        np.mean(
            np.power(samples - sample_mean, m) *
            np.expand_dims(self_normalized_weights, -1),
            axis=0) for m in range(1, max_moment + 1)
    ])
  return centered_moments, snis_weighted_moments


def rmse(predictions, targets):
  rmse_loss = np.sqrt(np.mean(np.power(predictions - targets, 2), axis=-1))
  return np.mean(rmse_loss)


def mc_expectations(samples,
                    target_logprob,
                    proposal_logprob,
                    test_functions,
                    eps=1e-10,
                    stabilize_weights=False,
                    truncated=False):
  assert len(test_functions) > 0
  sample_mean = np.mean(samples, axis=0)
  #     centered_powers = np.array([f_test(samples - sample_mean) for f_test in test_functions])
  centered_powers = np.array([f_test(samples) for f_test in test_functions])
  print('centered_powers shape = {}'.format(
      centered_powers.shape))  # m+1 x num_samples x input_dim

  sampling_weights = metrics.snis_weights(
      samples,
      target_logprob,
      proposal_logprob,
      eps=eps,
      clip=False,
      stabilize_weights=stabilize_weights,
      truncated=truncated)
  print('sampling_weights shape = {}'.format(
      sampling_weights.shape))  # should be: num_samples x input_dim
  print(test_functions[0](samples - sample_mean).shape)
  reweighted_powers = np.array([
      np.reshape(sampling_weights, (-1, 1)) * f_test(samples)
      for f_test in test_functions
  ])
  print('reweighted_powers shape = {}'.format(
      sampling_weights.shape))  # should be: m+1 x num_samples x input_dim

  vb_moments = np.mean(centered_powers, axis=1)
  snis_moments = np.sum(reweighted_powers, axis=1)
  print('vb_moments shape = {}'.format(
      vb_moments.shape))  # should be m+1 x input_dim
  print('snis_moments shape = {}'.format(
      snis_moments.shape))  # should be m+1 x input_dim
  return vb_moments, snis_moments


def difference_of_moments(num_samples,
                          log_weights,
                          means,
                          cov_sqrts,
                          mog_target_logprob,
                          mog_target_samples,
                          test_functions,
                          eps=1e-12,
                          stabilize_weights=True):
  num_test_functions = len(test_functions)
  samples = utils.mog_sample(num_samples, log_weights, means, cov_sqrts)

  def proposal_logprob(theta):
    return utils.mog_logprob(theta, log_weights, means, cov_sqrts)

  vb_moments, snis_moments = mc_expectations(
      samples,
      mog_target_logprob,
      proposal_logprob,
      test_functions,
      eps=eps,
      stabilize_weights=stabilize_weights)

  mog_target_samples_mean = np.mean(mog_target_samples, axis=0)
  recentered_samples = mog_target_samples  # - mog_target_samples_mean
  target_moments = np.mean(
      [f_test(recentered_samples) for f_test in test_functions],
      axis=1)  # shape should be num_functions x input_dim

  vb_moments_difference = np.array([
      rmse(vb_moments[idx], target_moments[idx])
      for idx in range(num_test_functions)
  ])
  snis_moments_difference = np.array([
      rmse(snis_moments[idx], target_moments[idx])
      for idx in range(num_test_functions)
  ])
  print('[VB] Moments Difference = {}'.format(vb_moments_difference))
  print('[SNIS] Moments Difference = {}'.format(snis_moments_difference))
  return vb_moments_difference, snis_moments_difference
