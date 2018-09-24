import tensorflow as tf
from .input import input_fn
from .estimator import get_estimator

def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

graph = tf.Graph()
with graph.as_default():

    query = convert_to_tensor([[1, 0, 0],[0, 0, 2]])

    list = convert_to_tensor([[[1, 0, 0], [0, 0, 2]],[[1, 0, 0], [0, 0, 1]]])
    list2 = convert_to_tensor([[[1, 0, 0], [0, 0, 1], [0, 0, 1]],[[1, 0, 0], [0, 0, 1], [0, 0, 1]]])

    scores = tf.einsum("jl,jkl -> jk", query, list)
    scores2 = tf.einsum("jl,jkl -> jk", query, list2)

    attention_distribution = tf.nn.softmax(scores)

    attention = tf.einsum("jk,jkl -> jl", attention_distribution, list)

    focus = tf.reduce_sum(scores, axis=1, keep_dims=True)

    attention_concat = tf.concat([query, attention, focus], axis=1)


def train():
    args={
        "batch_size": 64,
        "max_steps": None,
        "model_dir": None,
        "warm_start_dir": None,
        "max_gradient_norm": 4.0,
        "input_dir": "experiments",
        "train_input_path": "experiments/train",
        "eval_input_path": "experiments/eval",
        "limit": 200000,
        "use_summary_scalar": True,
        "use_lr_finder": True,
        "use_lr_decay": False,
        "learning_rate": 1.5e-04,
        "use_attention_focus": True,
    }

    hooks=None

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(args,"train"),
        max_steps=args["max_steps"] if args["max_steps"] is not None else 200000,
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