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

"""Run FKL and RKL boosting."""

import time

from absl import app
from absl import flags
import autograd.numpy as np
import nn
import uci

from adaptive_is_AISTATS_2021.remainder import metrics
from adaptive_is_AISTATS_2021.remainder import remainder_main
from adaptive_is_AISTATS_2021.remainder import utils

flags.DEFINE_boolean(
    'alternate_value', False,
    'candidates from BLR_MCMC.ipynb: alternate_list = [False, True]')
flags.DEFINE_integer(
    'num_vi_iters_value', 200,
    'Number of gradient steps when fitting each boosting component.')
flags.DEFINE_integer(
    'num_samples_value', 25,
    'Number of Monte Carlo samples when computing each gradient.')
flags.DEFINE_float(
    'mean_step_size_value', 0.01,
    'Step size for the mean parameter for each boosting component.')
flags.DEFINE_float(
    'cov_sqrt_step_value', 0.1,
    'Step size for the covariance parameter for each boosting component.')
flags.DEFINE_float('init_cov_sqrt_scale', 0.001,
                   'Scale for initial covariance matrix.')
flags.DEFINE_boolean(
    'natural_gradients_value', True,
    'candidates from BLR_MCMC.ipynb: natural_gradients_list = [True, False]')
flags.DEFINE_boolean('remainder_value', False,
                     'If True, fit components using the remainder.')
flags.DEFINE_boolean('reverse_value', False,
                     'If True, use RKL. If False, use FKL.')
flags.DEFINE_boolean('stabilized_gradients_value', True,
                     'If True, stabilize gradients numerically.')
flags.DEFINE_integer('split_seed', 0, 'Seed for random dataset split.')
flags.DEFINE_string('dataset_name', 'wine',
                    'Dataset name. See uci.py for options.')
flags.DEFINE_boolean('split_weights_value', True, 'Stability parameter.')
flags.DEFINE_boolean('old_mixture_value', True, 'Stability parameter.')
flags.DEFINE_float('eps', 1e-10, 'eps value')
flags.DEFINE_boolean('stabilize_weights_value', True,
                     'If True, stabilize IS weights numerically.')
flags.DEFINE_boolean('mixed_value', True,
                     'If True, fit the first component using RKL.')
flags.DEFINE_boolean('stabilize_remainder_value', True, 'Stability parameter.')
flags.DEFINE_integer('num_boosting_iters', 3, 'Number of boosting iterations.')
flags.DEFINE_integer('num_comp_samples', 500,
                     'Number of samples in final log likelihood computation.')
flags.DEFINE_boolean('print_mog_only', True, 'If true, only print mog metrics.')
flags.DEFINE_boolean(
    'linear_model', True,
    'If true, performs Bayesian linear regression. Otherwise, uses a neural '
    'network with 1 hidden layer.')

FLAGS = flags.FLAGS

# Data Processing


def return_dataset(dataset_name, split_seed=0):
  data = uci.load_dataset(dataset_name, split_seed=split_seed)
  train = data[0]
  test = data[1]
  X_train = train[0]
  y_train = train[1]
  X_test = test[0]
  y_test = test[1]
  return X_train, y_train, X_test, y_test


def print_metric(metric_name, metric_value):
  """Prints metrics."""
  print('[metric] %s=%f' % (metric_name, metric_value))


def main(argv):
  #if len(argv) > 1:
  #  raise app.UsageError('Too many command-line arguments.')

  # Instantiate Data split and regression function
  split_seed = FLAGS.split_seed
  X_train, y_train, X_test, y_test = return_dataset(FLAGS.dataset_name,
                                                    split_seed)
  ydim = 1 if y_train.ndim == 1 else y_train.shape[1]
  layer_sizes = None
  if FLAGS.linear_model:
    layer_sizes = [X_train.shape[1], ydim]

  log_posterior, _, loglike, parser, _, _ = nn.make_nn_regression_funs(
      X_train, y_train, layer_sizes, obs_variance=None)
  print('Number of parameters = {}'.format(parser.N))

  def blr_log_posterior(thetas):
    return np.array([log_posterior(theta) for theta in thetas])

  def blr_loglike(thetas):
    return np.mean([
        np.mean([
            loglike(
                theta,
                X_test_sample.reshape((1, -1)),
                y_test_sample.reshape((1, 1)),
                unstandardized_data=True)
            for (X_test_sample, y_test_sample) in zip(X_test, y_test)
        ])
        for theta in thetas
    ])

  def is_blr_loglike(thetas, snis_weights):
    avg_loglike = np.zeros(len(thetas))
    for theta_idx in range(len(thetas)):
      theta = thetas[theta_idx]
      avg_loglike[theta_idx] = snis_weights[theta_idx] * np.mean([
          loglike(
              theta,
              X_test_sample.reshape((1, -1)),
              y_test_sample.reshape((1, 1)),
              unstandardized_data=True)
          for (X_test_sample, y_test_sample) in zip(X_test, y_test)
      ])
    return np.sum(avg_loglike)

  def print_mixture_loglike(log_weights,
                            means,
                            cov_sqrts,
                            num_comp_samples=2000,
                            eps=1e-6):
    mixture_samples = [
        np.random.multivariate_normal(mean, np.diag(cov_sqrt), num_comp_samples)
        for (mean, cov_sqrt) in zip(means, cov_sqrts)
    ]
    aggregate_samples = np.sum([
        utils.logistic_transform(log_weight) * component_sample
        for (log_weight, component_sample) in zip(log_weights, mixture_samples)
    ],
                               axis=0)
    average_theta = np.mean(aggregate_samples, 0)
    average_theta = np.reshape(average_theta, (1, -1))
    reweighted_means = np.reshape(
        utils.logistic_transform(log_weights), (-1, 1)) * means
    mean_theta = np.mean(reweighted_means, 0)
    mean_theta = np.reshape(mean_theta, (1, -1))
    num_samples = num_comp_samples * len(log_weights)
    mog_samples = utils.mog_sample(num_samples, log_weights, means, cov_sqrts)
    average_mog_theta = np.mean(aggregate_samples, 0)
    average_mog_theta = np.reshape(average_mog_theta, (1, -1))

    for component_sample in mixture_samples:
      print('component_sample.shape = {}'.format(component_sample.shape))
      print('> Componentwise LL : {}'.format(
          np.mean(blr_loglike(component_sample))))

    aggregate_samples_ll = blr_loglike(aggregate_samples)
    avg_agg_samples_ll = blr_loglike(average_theta)
    modal_ll = blr_loglike(mean_theta)
    mog_samples_ll = blr_loglike(mog_samples)
    avg_mog_samples_ll = blr_loglike(average_mog_theta)

    print_metric('VB-LL:aggregate_samples_ll', aggregate_samples_ll)
    print_metric('VB-LL:avg_agg_samples_ll', avg_agg_samples_ll)
    print_metric('VB-LL:modal_ll', modal_ll)
    print_metric('VB-LL:mog_samples_ll', mog_samples_ll)
    print_metric('VB-LL:avg_mog_samples_ll', avg_mog_samples_ll)

    # Now computing IS LogLikelihood
    target_logprob = blr_log_posterior

    def proposal_logprob(x):
      return utils.mog_logprob(x, log_weights, means, cov_sqrts, eps=eps)

    aggregate_snis = metrics.snis_weights(
        aggregate_samples,
        target_logprob,
        proposal_logprob,
        eps=eps,
        clip=False)
    avg_agg_snis = metrics.snis_weights(
        average_theta, target_logprob, proposal_logprob, eps=eps, clip=False)
    modal_snis = metrics.snis_weights(
        mean_theta, target_logprob, proposal_logprob, eps=eps, clip=False)
    mog_snis = metrics.snis_weights(
        mog_samples, target_logprob, proposal_logprob, eps=eps, clip=False)
    avg_mog_snis = metrics.snis_weights(
        average_mog_theta,
        target_logprob,
        proposal_logprob,
        eps=eps,
        clip=False)

    print('aggregate_snis shape = {}'.format(aggregate_snis.shape))
    print('avg_agg_snis shape = {}'.format(avg_agg_snis.shape))
    print('modal_snis shape = {}'.format(modal_snis.shape))
    print('mog_snis shape = {}'.format(mog_snis.shape))
    print('avg_mog_snis shape = {}'.format(avg_mog_snis.shape))

    aggregate_samples_iwll = is_blr_loglike(aggregate_samples, aggregate_snis)
    avg_agg_samples_iwll = is_blr_loglike(average_theta, avg_agg_snis)
    modal_iwll = is_blr_loglike(mean_theta, modal_snis)
    mog_samples_iwll = is_blr_loglike(mog_samples, mog_snis)
    avg_mog_samples_iwll = is_blr_loglike(average_mog_theta, avg_mog_snis)

    print_metric('IW-LL:aggregate_samples_iwll', aggregate_samples_iwll)
    print_metric('IW-LL:avg_agg_samples_iwll', avg_agg_samples_iwll)
    print_metric('IW-LL:modal_iwll', modal_iwll)
    print_metric('IW-LL:mog_samples_iwll', mog_samples_iwll)
    print_metric('IW-LL:avg_mog_samples_iwll', avg_mog_samples_iwll)

  def print_mixture_loglike_mog(log_weights,
                                means,
                                cov_sqrts,
                                num_comp_samples=2000,
                                eps=1e-6,
                                prefix=''):
    mixture_samples = [
        np.random.multivariate_normal(mean, np.diag(cov_sqrt), num_comp_samples)
        for (mean, cov_sqrt) in zip(means, cov_sqrts)
    ]

    num_samples = num_comp_samples * len(log_weights)
    mog_samples = utils.mog_sample(num_samples, log_weights, means, cov_sqrts)

    for component_sample in mixture_samples:
      print('component_sample.shape = {}'.format(component_sample.shape))
      print('> Componentwise LL : {}'.format(
          np.mean(blr_loglike(component_sample))))

    mog_samples_ll = blr_loglike(mog_samples)

    print_metric(prefix + 'VB-LL:mog_samples_ll', mog_samples_ll)

    # Now computing IS LogLikelihood
    target_logprob = blr_log_posterior

    def proposal_logprob(x):
      return utils.mog_logprob(x, log_weights, means, cov_sqrts, eps=eps)

    mog_snis = metrics.snis_weights(
        mog_samples,
        target_logprob,
        proposal_logprob,
        stabilize_weights=True,
        eps=eps,
        clip=False)

    mog_samples_iwll = is_blr_loglike(mog_samples, mog_snis)

    print_metric(prefix + 'IW-LL:mog_samples_iwll', mog_samples_iwll)
    return mog_samples_iwll

  # Default values for parameters we're not searching over.
  init_mean_value = np.zeros((parser.N))
  init_cov_sqrt_value = FLAGS.init_cov_sqrt_scale * np.ones(parser.N)

  # Set flags for parameters we are searching over.
  print('FLAGS.split_weights_value = {}'.format(FLAGS.split_weights_value))
  print('FLAGS.old_mixture_value = {}'.format(FLAGS.old_mixture_value))
  print('FLAGS.alternate_value = {}'.format(FLAGS.alternate_value))
  print('FLAGS.num_vi_iters_value = {}'.format(FLAGS.num_vi_iters_value))
  print('FLAGS.num_samples_value = {}'.format(FLAGS.num_samples_value))
  print('FLAGS.mean_step_size_value = {}'.format(FLAGS.mean_step_size_value))
  print('FLAGS.cov_sqrt_step_value = {}'.format(FLAGS.cov_sqrt_step_value))
  print('FLAGS.natural_gradients_value = {}'.format(
      FLAGS.natural_gradients_value))
  print('FLAGS.remainder_value = {}'.format(FLAGS.remainder_value))
  print('FLAGS.reverse_value = {}'.format(FLAGS.reverse_value))
  print('FLAGS.stabilized_gradients_value = {}'.format(
      FLAGS.stabilized_gradients_value))
  print('FLAGS.stabilize_weights_value = {}'.format(
      FLAGS.stabilize_weights_value))
  print('FLAGS.stabilize_remainder_value = {}'.format(
      FLAGS.stabilize_remainder_value))
  print('FLAGS.stabilize_remainder_value = {}'.format(
      FLAGS.stabilize_remainder_value))
  print('FLAGS.mixed_value = {}'.format(FLAGS.mixed_value))
  print('FLAGS.eps = {}'.format(FLAGS.eps))
  np.random.seed(0)
  start_time = time.time()

  final_weights, final_means, final_cov_sqrts, final_metrics = remainder_main.boosting_vi(
      blr_log_posterior,
      d=parser.N,
      num_boosting_iters=FLAGS.num_boosting_iters,
      num_vi_iters=FLAGS.num_vi_iters_value,
      num_samples=FLAGS.num_samples_value,
      mean_step_size=FLAGS.mean_step_size_value,
      cov_sqrt_step_size=FLAGS.cov_sqrt_step_value,
      gamma_step_size=0.005,
      init_mean=init_mean_value,
      init_cov_sqrt=init_cov_sqrt_value,
      init_weight=utils.weight_init(),
      analytical=False,
      remainder=FLAGS.remainder_value,
      reverse=FLAGS.reverse_value,
      joint=False,
      weight_search_method='fully_corrective',
      natural_gradients=FLAGS.natural_gradients_value,
      mixed=FLAGS.mixed_value,
      stabilized_gradients=FLAGS.stabilized_gradients_value,
      stabilize_weights=FLAGS.stabilize_weights_value,
      stabilize_remainder=FLAGS.stabilize_remainder_value,
      residual_regulariziation=0.,
      residual=False,
      gradient_init=True,
      alternative=FLAGS.alternate_value,
      old_mixture=FLAGS.old_mixture_value,
      split_weights=FLAGS.split_weights_value,
      diagonal=True,
      clip=False,
      tol=1e-8,
      eps=FLAGS.eps)

  print('> Running time = {}'.format(time.time() - start_time))

  res = []
  if FLAGS.print_mog_only:
    for i in range(len(final_weights)):
      res.append(print_mixture_loglike_mog(
          final_weights[:i + 1],
          final_means[:i + 1],
          final_cov_sqrts[:i + 1],
          num_comp_samples=FLAGS.num_comp_samples,
          prefix=('%d:' % (i + 1))))
  else:
    print_mixture_loglike(
        final_weights,
        final_means,
        final_cov_sqrts,
        num_comp_samples=FLAGS.num_comp_samples)

  print('------------------------------------------------------------------')

  return res


if __name__ == '__main__':
  app.run(main)
