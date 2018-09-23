import tensorflow as tf
import random
from collections import Counter

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Miscel
# --------------------------------------------------------------------------

def min_none(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# --------------------------------------------------------------------------
# TFRecord functions
# --------------------------------------------------------------------------

# Why it's so awkward to write a record I do not know

def int32_feature(value):
    return tf.train.Feature(int32_list=tf.train.Int32List(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def string_feature(value):
    return conv_bytes_feature(value)


def conv_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


# --------------------------------------------------------------------------
# TF helpers
# --------------------------------------------------------------------------

def tf_startswith(tensor, prefix, axis=None):
    return tf.reduce_all(tf.equal(tf.substr(tensor, 0, len(prefix)), prefix), axis=axis)


# --------------------------------------------------------------------------
# File readers and writers
# --------------------------------------------------------------------------

class Partitioner(object):

    def __init__(self, args):
        self.args = args
        self.written = 0
        self.answer_classes = Counter()
        self.answer_classes_types = Counter()

        #TODO: change these to use numpy arrays - much faster
        self.train_query_set = set()
        self.eval_query_set = set()

    def __enter__(self, *vargs):
        self.files = {
            i: tf.python_io.TFRecordWriter(self.args[f"{i}_input_path"])
            for i in self.args['modes']
        }

        return self

    def write(self, doc, record):
        r = random.random()
        query_as_str = str(doc["query"])

        if r < self.args["eval_holdback"]:
            mode = "eval"
            self.eval_query_set.add(query_as_str)
            if query_as_str in self.train_query_set:
                # Dont add to train because it's in eval
                logger.warning("Skipping adding record to eval set because it's already in the train set")
                return
        elif r < self.args["eval_holdback"] + self.args["predict_holdback"]:
            mode = "predict"
        else:
            mode = "train"
            self.train_query_set.add(query_as_str)
            if query_as_str in self.eval_query_set:
                # Dont add to eval because it's in train
                logger.warning("Skipping adding record to train set because it's already in the eval set")
                return

        key = (str(doc["answer"]), doc["question_type"])

        self.files[mode].write(record)
        self.answer_classes[str(doc["answer"])] += 1
        self.answer_classes_types[key] += 1
        self.written += 1

    def __exit__(self, *vargs):
        for i in self.files.values():
            i.close()

        self.files = None


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def StringDataset(s):
    def generator():
        yield s

    return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]))
