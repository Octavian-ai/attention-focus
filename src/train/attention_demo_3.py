import tensorflow as tf
from .model import attention
def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

graph = tf.Graph()
with graph.as_default():
    question_batch = convert_to_tensor([[0,1,1],[0,1,1]])
    list_batch = convert_to_tensor([[[0,0,0],[0,1,1]],[[0,1,0],[0,0,1]]])

    debug = True
    use_focus=True
    attention_concat = attention(question_batch, list_batch, question_batch, focus_fn="sum", score="euclidean", use_focus=use_focus, debug=debug)

# Run a Session
with tf.Session(graph=graph) as session:
    output_d = session.run(attention_concat)
    print(output_d)