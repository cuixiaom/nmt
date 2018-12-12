"""Tests for inference.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import argparse
import sys

import numpy as np
from google.protobuf import text_format

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import iterator_utils
import vocab_utils

def load_data(inference_input_file):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  return inference_data

hparams = tf.contrib.training.HParams(
    random_seed=3,
    eos="eos",
    sos="sos",
    src_vocab_file="/mnt/nrvlab_300G_work01/cuixiaom/Data/Nmt/Ger-Eng/Training/wmt16/vocab.bpe.32000.de",
    tgt_vocab_file="/mnt/nrvlab_300G_work01/cuixiaom/Data/Nmt/Ger-Eng/Training/wmt16/vocab.bpe.32000.en",
    share_vocab=False,
    src_max_len_infer=50,
    use_char_encode=False)

src_vocab_file = hparams.src_vocab_file
tgt_vocab_file = hparams.tgt_vocab_file

src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
    src_vocab_file, tgt_vocab_file, hparams.share_vocab)
reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
    tgt_vocab_file, default_value=vocab_utils.UNK)

src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

src_dataset = tf.data.Dataset.from_tensor_slices(
    src_placeholder)
iterator = iterator_utils.get_infer_iterator(
        src_dataset,
        src_vocab_table,
        batch_size=batch_size_placeholder,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer,
        use_char_encode=hparams.use_char_encode)

table_initializer = tf.tables_initializer()
source = iterator.source
seq_len = iterator.source_sequence_length
inference_input_file="newstest2014.tok.bpe.32000_100w.de"
infer_data = load_data(inference_input_file)
batch_size = 1

with tf.Session() as sess:
  sess.run(table_initializer)
  sess.run(
    iterator.initializer,
    feed_dict={
        src_placeholder: infer_data,
        batch_size_placeholder: batch_size
    })
  while  True:
    try:
      print("a new batch!")
      (source_v, seq_len_v) = sess.run((source, seq_len))
      print(source_v)
      print(seq_len_v)
    except tf.errors.OutOfRangeError:
      print("Done!")
      break

