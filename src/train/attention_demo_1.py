import tensorflow as tf

def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

graph = tf.Graph()
with graph.as_default():

    query = convert_to_tensor([1, 0, 0])

    list = convert_to_tensor([[[1, 0, 0], [0, 0, 1]]])
    list2 = convert_to_tensor([[[1, 0, 0], [0, 0, 1], [0, 0, 1]]])

    scores = tf.einsum("l,jkl -> jk", query, list)
    scores2 = tf.einsum("l,jkl -> jk", query, list2)

    attention_distribution = tf.nn.softmax(scores)

    attention = tf.multiply(list, tf.transpose([attention_distribution]))


def train():
    train_spec = tf.estimator.TrainSpec(

        input_fn=gen_input_fn(args, "train"),

        max_steps=args["max_steps"] * 1000 if args["max_steps"] is not None else None,

        hooks=hooks)

    eval_spec = tf.estimator.EvalSpec(

        input_fn=gen_input_fn(args, "eval"),

        throttle_secs=300)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



# Run a Session
with tf.Session(graph=graph) as session:
    output_d = session.run(scores)
    output_d2 = session.run(scores2)
    output_softmax = session.run(attention_distribution)
    output_attention = session.run(attention)
    print("\nsoftmax input")
    print(output_d)
    print(output_d2)