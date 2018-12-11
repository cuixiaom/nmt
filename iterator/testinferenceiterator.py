"""Tests for inference.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

import iterator_utils





src_vocab_table = lookup_ops.index_table_from_tensor(
    tf.constant(["a", "b", "c", "eos", "sos"]))
src_dataset = tf.data.Dataset.from_tensor_slices(
    tf.constant(["c c a", "c a", "d", "f e a g"]))
hparams = tf.contrib.training.HParams(
    random_seed=3,
    eos="eos",
    sos="sos")
batch_size = 2
src_max_len = 3
iterator = iterator_utils.get_infer_iterator(
    src_dataset=src_dataset,
    src_vocab_table=src_vocab_table,
    batch_size=batch_size,
    eos=hparams.eos,
    src_max_len=src_max_len)
table_initializer = tf.tables_initializer()
source = iterator.source
seq_len = iterator.source_sequence_length
#assertEqual([None, None], source.shape.as_list())
#assertEqual([None], seq_len.shape.as_list())
with tf.Session() as sess:
  sess.run(table_initializer)
  sess.run(iterator.initializer)

  while  True:
    try:
      print("a new batch!")
      (source_v, seq_len_v) = sess.run((source, seq_len))
      print(source_v)
      print(seq_len_v)
    except tf.errors.OutOfRangeError:
      print("Done!")
      break

