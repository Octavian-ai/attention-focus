import tensorflow as tf
from ..train.input import input_fn
from ..train.estimator import get_estimator
from ..args import get_args

def train():
    args=get_args()

    hooks=None

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(args,"train"),
        max_steps=args["max_steps"] * 1000 if args["max_steps"] is not None else 200000,
        hooks=hooks)

    eval_spec = tf.estimator.EvalSpec(
        steps=10,
        start_delay_secs=5,
        input_fn=lambda: input_fn(args, "eval"),
        throttle_secs=30)

    estimator = get_estimator(args)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



# Run a Session
with tf.Session(graph=graph) as session:
    output_d = session.run(scores)
    output_d2 = session.run(scores2)
    output_softmax = session.run(attention_distribution)
    output_attention = session.run(attention)
    output_focus = session.run(focus)
    output_concat = session.run(attention_concat)
    print("\nsoftmax input")
    print(output_d)
    print(output_softmax)
    print(output_attention)
    print(output_focus)
    print(output_concat)
    print("\nsoftmax input2")
    print(output_d2)

    train()