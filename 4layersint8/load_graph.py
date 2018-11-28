# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import argparse
import sys

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


def load_data(inference_input_file):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  return inference_data


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]
  
  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

if __name__ == "__main__":
  file_name = "./data/grace_hopper.jpg"
  model_file = \
    "./gnmt_infermodel.pbtxt"

  input_layer = "Inference/Placeholder"
  output_layer = "Inference/add"

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_layer0", help="name of input layer 0")
  parser.add_argument("--input_layer1", help="name of input layer 1")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  args = parser.parse_args()

  if args.input_layer0:
    input_layer0 = args.input_layer0
  if args.input_layer1:
    input_layer1 = args.input_layer1
  if args.output_layer:
    output_layer = args.output_layer
  if args.inference_input_file:
    inference_input_file = args.inference_input_file

  # Read data
  infer_data = load_data(inference_input_file)
  batch_size = 32

  graph = load_graph(model_file)
  print(model_file)
  print(infer_data)
  input_name0 = "import/" + input_layer0
  input_name1 = "import/" + input_layer1
  output_name = "import/" + output_layer
  output_operation = graph.get_operation_by_name(output_name);
  input_operation0 = graph.get_operation_by_name(input_name0);
  input_operation1 = graph.get_operation_by_name(input_name1);
  dataset_init_op = graph.get_operation_by_name('dataset_init')


  config = tf.ConfigProto()
  config.inter_op_parallelism_threads = 1 

  t = 0.2
  with tf.Session(graph=graph, config=config) as sess:
#    results = sess.run(output_operation.outputs[0],
    results = sess.run(dataset_init_op,
                      feed_dict={input_operation0.outputs[0]: infer_data,
                      input_operation1.outputs[0]: batch_size})
    print(results)
