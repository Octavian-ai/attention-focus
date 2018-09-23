import tensorflow as tf

def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg


r = []
graph = tf.Graph()
with graph.as_default():

    # Start with a simple example
    a = convert_to_tensor([2,0])
    b = convert_to_tensor([0,1])
    def dot_to_scalar(a, b):
        return tf.einsum("i,i ->", a, b)

    def dot_to_vector(a, b):
        return tf.einsum("i,i -> i", a, b)

    def something(a, b):
        # x_j = Sum_i(a_i.b_j)
        return tf.einsum("i,j -> j", a, b)

    r = tf.Print(r, [
        dot_to_scalar(a, b),
        dot_to_scalar(b, b),
        dot_to_vector(a, b),
        dot_to_vector(b, b),
        something(a, b),
        something(b, b),
    ], message="1:")

    # Now a harder example
    a = convert_to_tensor([[0,1]])
    b = convert_to_tensor([[1,0],[1,1],[0,3]])
    def dot_to_scalar(a, b):
        return tf.einsum("ik,jk -> j ", a, b)

    r = tf.Print(r, [
        dot_to_scalar(a, b),
    ], message="2:", summarize=6)

# Run a Session
with tf.Session(graph=graph) as session:
    output_d = session.run(r)