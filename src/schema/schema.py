import tensorflow as tf
import pathlib
from collections import Counter

from src.build_data.util import *


def generate_record(args, doc):
    label = doc["answer"]
    query = doc["query"]
    kb = doc["list"]
    feature = {
        "query": tf.train.Feature(float_list=tf.train.FloatList(value=query)),
        "query_len": int64_feature(len(query)),
        "kb": tf.train.Feature(float_list=tf.train.FloatList(value=kb.flatten())),
        "kb_len": int64_feature(len(kb)),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        "label_length": int64_feature(len(label)),
        "type_string": string_feature(doc["question_type"]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_single_example(sample):
    return lambda i: tf.parse_example(
        i,
        features={
            'query': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'kb': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'label': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        })


def reconstitute_single_example(sample):
    print(sample)
    query_len_ = sample["query_len"]
    kb_len = sample["kb_len"]
    label_length = sample["label_length"]
    return lambda features: ({
                'query': tf.reshape(tf.cast(features["query"], tf.float32), [-1, query_len_]),
                'kb': tf.reshape(tf.cast(features["kb"], tf.float32), [-1, kb_len, query_len_]),
            }, tf.reshape(tf.cast(features["label"], tf.float32), [-1, label_length]))
