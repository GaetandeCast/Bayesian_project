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


class Adam:
  """Adam optimizer.

  Default parameters follow those provided in the original paper.
  # Arguments
                  lr: float >= 0. Learning rate.
                  beta_1: float, 0 < beta < 1. Generally close to
                  1.
                  beta_2: float, 0 < beta < 1. Generally close to
                  1.
                  epsilon: float >= 0. Fuzz factor.
                  decay: float >= 0. Learning rate decay over each
                  update.
  # References
                  - [Adam - A Method for Stochastic
                  Optimization](http://arxiv.org/abs/1412.6980v8)
  """

  def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               decay=0.,
               **kwargs):

    allowed_kwargs = {'clipnorm', 'clipvalue'}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError('Unexpected keyword argument '
                        'passed to optimizer: ' + str(k))
    self.__dict__.update(kwargs)
    self.iterations = 0
    self.lr = lr
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.decay = decay
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_update(self, params, grads, callback=None):
    """ params and grads are list of numpy arrays"""
    original_shapes = [x.shape for x in params]
    # print("Original Shapes = {}".format(original_shapes))
    params = [x.flatten() for x in params]
    # print("params = {}".format(params))
    grads = [x.flatten() for x in grads]
    # print("grads = {}".format(grads))

    lr = self.lr
    if self.initial_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))

    t = self.iterations + 1
    lr_t = lr * (
        np.sqrt(1. - np.power(self.beta_2, t)) /
        (1. - np.power(self.beta_1, t)))

    if not hasattr(self, 'ms'):
      self.ms = [np.zeros(p.shape) for p in params]
      self.vs = [np.zeros(p.shape) for p in params]

    ret = [None] * len(params)
    for i, p, g, m, v in zip(
        range(len(params)), params, grads, self.ms, self.vs):
      m_t = ((self.beta_1 * m) + (1. - self.beta_1) * g)
      v_t = ((self.beta_2 * v) + (1. - self.beta_2) * np.square(g))
      # Bias Correction
      m_hat_t = m_t / (1 - np.power(self.beta_1, t))
      v_hat_t = v_t / (1 - np.power(self.beta_2, t))
      # Update Rule
      p_t = p - lr_t * m_hat_t / (np.sqrt(v_hat_t) + self.epsilon)
      self.ms[i] = m_t
      self.vs[i] = v_t
      ret[i] = p_t

    self.iterations += 1

    for i in range(len(ret)):
      ret[i] = ret[i].reshape(original_shapes[i])

    return ret


class SGD:
  """SGD optimizer.

                # Arguments
                                lr: float >= 0. Learning rate.
                """

  def __init__(self, lr=0.001, **kwargs):

    allowed_kwargs = {'clipnorm', 'clipvalue'}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError('Unexpected keyword argument '
                        'passed to optimizer: ' + str(k))
    self.__dict__.update(kwargs)
    self.lr = lr

  def get_update(self, params, grads):
    """ params and grads are list of numpy arrays"""
    original_shapes = [x.shape for x in params]
    params = [x.flatten() for x in params]
    grads = [x.flatten() for x in grads]
    ret = [None] * len(params)
    for i, p, g in zip(range(len(params)), params, grads):
      ret[i] = p - self.lr * g

    for i in range(len(ret)):
      ret[i] = ret[i].reshape(original_shapes[i])

    return ret


def euclidean_proj_simplex(v, s=1):
  assert s > 0, 'Radius s must be strictly positive (%d <= 0)' % s
  n, = v.shape  # will raise ValueError if v is not 1-D
  # check if we are already on the simplex
  if v.sum() == s and np.alltrue(v >= 0):
    return v
  # get the array of cumulative sums of a sorted (decreasing) copy of v
  u = np.sort(v)[::-1]
  cum_sum = np.cumsum(u)
  # get the number of > 0 components of the optimal solution
  nonz = np.nonzero(u * np.arange(1, n + 1) > (cum_sum - s))
  # print("nonz = {}".format(nonz))
  lam = nonz[0][-1]
  # compute the Lagrange multiplier associated to the simplex constraint
  theta = (cum_sum[lam] - s) / (lam + 1.0)
  w = (v - theta).clip(min=0)
  return w
