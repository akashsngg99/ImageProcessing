#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import net.lenet as lenet
import cv2
from utils.copy import get_file_list

from tensorflow.python.platform import gfile

data_params = {
  'model_root': 'data',
  'data_dir': 'MNIST_data/mnist_test',
  'data_format': 'channels_last',
  'num_classes': 10,
  'image_size': [28, 28]
}

model_params = {
  'lenet': {
    "net": lenet.mnist_model,
    'model_dir': os.path.join(
      data_params['model_root'],
      'lenet'
    ),
    'checkpoint_index': 72000
  }
}

init_params = {
  'data_params': data_params,
  'net_params': model_params['lenet'],
}

graph_params = {
  "source_checkpoint": os.path.join(
    init_params['net_params']['model_dir'],"model.ckpt-{}"
      .format(init_params['net_params']["checkpoint_index"])
  ),
  "inference_graph": os.path.join(
    init_params['net_params']["model_dir"],"inference_graph_{}.pb"
      .format(init_params['net_params']["checkpoint_index"])
  ),
  "input_nodes": "input",
  "output_nodes": "output"
}

model_fn = init_params['net_params']['net']


def inference_fn(images):
  logits = model_fn(images, mode=tf.estimator.ModeKeys.PREDICT, data_format=None)
  output = tf.identity(logits, name=graph_params["output_nodes"])
  return output

def export_inference_graph():
  """
  Need inference graph fn as inference_fn,
  placeholder must feed by real numpy matrix.
  """
  output_file=graph_params["inference_graph"]

  with tf.Graph().as_default() as graph:
    network_fn = inference_fn

    placeholder = tf.placeholder(name=graph_params["input_nodes"], dtype=tf.float32,
                                shape=[None,init_params['data_params']['image_size'][0],
                                 init_params['data_params']['image_size'][1], 1])
    ones = np.ones([1, init_params['data_params']['image_size'][0],
                    init_params['data_params']['image_size'][1],1])

    print(ones.shape)

    sess = tf.Session()
    output = network_fn(placeholder)
    sess.run(tf.global_variables_initializer())
    output_v = sess.run([output], feed_dict={placeholder:ones})
    graph_def = graph.as_graph_def()
    with gfile.GFile(output_file, 'wb') as f:
      f.write(graph_def.SerializeToString())


def predict(image_path):
  x = tf.placeholder(name=graph_params["input_nodes"], dtype=tf.float32,
                               shape=[None, 28, 28, 1])

  image = cv2.imread(image_path, 0)

  image = np.reshape(image, [1, 28, 28, 1])
  with tf.Session() as sess:
    y = inference_fn(x)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, graph_params["source_checkpoint"])
    print("Model restored.")
    result = sess.run([y], feed_dict={x: image})
    return result

def predict_batch(images_dir, phase='png'):
  file_list = get_file_list(images_dir)
  images = []
  for file in file_list:
    if not file.split('.')[-1] == phase:
      continue
    print(file)
    images.append(cv2.imread(os.path.join(images_dir, file), 0))

  xs = tf.placeholder(name=graph_params["input_nodes"], dtype=tf.float32,
                     shape=[None, 28, 28, 1])
  images = np.asarray(images)
  images = np.reshape(images, [len(images), 28, 28, 1])

  with tf.Session() as sess:
    ys = inference_fn(xs)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, graph_params["source_checkpoint"])
    print("Model restored.")
    results = sess.run(ys, feed_dict={xs: images})
    return results


if __name__ == "__main__":
  # image_path = "MNIST_data/mnist_train/0/mnist_train_34.png"
  # print(predict(image_path))
  images_path = "MNIST_data/fine_tuning"
  results = predict_batch(images_path, 'jpg')
  results = np.asarray(results)
  print(np.asarray(results).shape)
  for result in results:
    print(result)

