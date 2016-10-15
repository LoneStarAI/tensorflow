# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

FLAGS = None

def hinge_loss_cap(logits, target, cap=10.0, scope=None):
  """Method that returns the loss tensor for hinge loss.

  Args:
    logits: The logits, a float tensor.
    target: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A `Tensor` of same shape as logits and target representing the loss values
      across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `target` don't match.
  """
  with ops.name_scope(scope) as scope:
    logits.get_shape().assert_is_compatible_with(target.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    target = math_ops.to_float(target)
    all_ones = array_ops.ones_like(target)
    # convert labels into {1, -1} matrix
    labels = math_ops.sub(2 * target, all_ones)
    cross_prod = math_ops.mul(labels, logits)
    losses = nn_ops.relu(math_ops.sub(all_ones, cross_prod))
    losses_cap = -nn_ops.relu(math_ops.sub(cap, losses))
    return losses_cap

    # sigmoid loss
    #losses_left = math_ops.sigmoid(-10*cross_prod)
    #losses_right = math_ops.sigmoid(-0.01*cross_prod)
    #losses_soft = math_ops.minimum(losses_left, losses_right)

def train_with_cap(dataset, cap=10.0):
  """ Train model with different para of "cap" for hinge loss
  Args:
    dataset: tf.contrib.learn.python.learn.datasets.mnist.Dataset type
  """

  mnist = dataset
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

  cross_entropy = tf.reduce_mean(hinge_loss_cap(y, y_, cap=cap))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # introduce noise to labels
    np.random.shuffle(batch_ys[0:20, :])

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accu_val = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  print accu_val


def main(_):
  caps = [5.0, 10.0, 50.0, 100.0, 200.0]
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  for cap in caps:
    train_with_cap(mnist, cap=cap)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
