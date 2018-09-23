import tensorflow as tf
import numpy as np

from .utils import *

def attention_fn(question_batch):
    scale_factor = tf.get_variable("attention_scalar",
                                   trainable=True,
                                             shape=[1], dtype=tf.float32)

    #scale_factor = tf.assign(scale_factor, tf.clip_by_value(scale_factor,-5,5,name=None))

    return question_batch

def actual_model(question_batch, list_batch, n_output_classes):
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        attn_query = attention_fn(question_batch)

        print(attn_query)
        print(list_batch)
        attn_scores = tf.einsum("jl,jkl -> jk", attn_query, list_batch)
        attn_distribution = tf.nn.softmax(attn_scores)

        attn_output = tf.einsum("jk,jkl -> jl", attn_distribution, list_batch)

        attention_concat = tf.concat([question_batch, attn_output], axis=1)
        focus = tf.reduce_sum(attn_scores, axis=1, keep_dims=True)
        attention_concat = tf.concat([attention_concat, focus], axis=1)

        magic = deeep(attention_concat, width=question_batch.shape[-1], activation=tf.nn.tanh)
        decision = tf.layers.dense(magic, n_output_classes, activation=tf.nn.sigmoid)

        # Not using softmax because we do softmax in the loss
        return decision

def model_fn(features, labels, mode, params):

    # --------------------------------------------------------------------------
    # Setup input
    # --------------------------------------------------------------------------

    args = params

    # EstimatorSpec slots
    loss = None
    train_op = None
    eval_metric_ops = None
    predictions = None
    eval_hooks = None

    # --------------------------------------------------------------------------

    # Model for realz

    # --------------------------------------------------------------------------
    print(features)
    logits = actual_model(features["query"], features["kb"], n_output_classes=labels.shape[-1])


    # --------------------------------------------------------------------------
    # Calc loss
    # --------------------------------------------------------------------------

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,-1), logits=logits)
        loss = tf.reduce_sum(crossent) / tf.to_float(args["batch_size"])


    # --------------------------------------------------------------------------
    # Optimize
    # --------------------------------------------------------------------------
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
            1E-06,
            global_step,
            decay_steps=1000,
            decay_rate=1.1
        )

        var_all = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op, gradients = minimize_clipped(optimizer, loss, args["max_gradient_norm"], var_all)


    # --------------------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------------------

    if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        predicted_labels = tf.nn.softmax(logits)
        actual_labels = labels
        label_class = tf.argmax(actual_labels, -1)
        predicted_class = tf.argmax(predicted_labels, -1)
        predictions = {
            "predicted_label": predicted_labels,
            "actual_label": actual_labels,
        }

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            label_class = tf.Print(label_class, [predicted_labels, logits, predicted_class], message="predicted labels", summarize=10)
            label_class = tf.Print(label_class, [actual_labels, labels, label_class], message="actual labels", summarize=10)
            #label_class = tf.Print(label_class, [tf.get_variable("attention_scalar")], message="attention_scalar")
        # For diagnostic visualisation
        predictions.update(features)

        # --------------------------------------------------------------------------
        # Eval metrics
        # --------------------------------------------------------------------------

        if mode == tf.estimator.ModeKeys.EVAL:

            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=label_class, predictions=predicted_class),
            }
            weights = tf.equal(label_class, 0)
            eval_metric_ops["class_accuracy_0"] = tf.metrics.accuracy(
                labels=label_class,
                predictions=predicted_class,
                weights=weights)

            weights = tf.equal(label_class, 1)
            eval_metric_ops["class_accuracy_1"] = tf.metrics.accuracy(
                labels=label_class,
                predictions=predicted_class,
                weights=weights)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops,
        export_outputs=None,training_chief_hooks=None, training_hooks=None, scaffold=None,
                                      evaluation_hooks=eval_hooks, prediction_hooks=None)
