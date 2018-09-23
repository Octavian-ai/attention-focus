import tensorflow as tf
import math

def minimize_clipped(optimizer, value, max_gradient_norm, var=None):
    global_step = tf.train.get_global_step()
    if var is None:
        var = tf.trainable_variables()
    gradients = tf.gradients(value, var)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    grad_dict = dict(zip(var, clipped_gradients))
    op = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)
    return op, grad_dict


def deeep(tensor, width, depth=2, residual_depth=3, activation=tf.nn.tanh):
    """
    Quick 'n' dirty "let's slap on some layers" function.

    Implements residual connections and applies them when it can. Uses this schematic:
    https://blog.waya.ai/deep-residual-learning-9610bb62c355
    """

    with tf.name_scope("deeep"):
        if residual_depth is not None:
            for i in range(math.floor(depth / residual_depth)):
                tensor_in = tensor
                for j in range(residual_depth - 1):
                    tensor = tf.layers.dense(tensor, width, activation=activation)

                tensor = tf.layers.dense(tensor, width)

                if tensor_in.shape[-1] == width:
                    tensor += tensor_in

                tensor = activation(tensor)

            remaining = depth % residual_depth

        else:
            remaining = depth

        for i in range(remaining):
            tensor = tf.layers.dense(tensor, width, activation=activation)

        return tensor