import argparse
import os.path
import yaml
import pathlib
import tensorflow as tf

global_args = {}


def get_args(extend=lambda parser: None):
    parser = argparse.ArgumentParser()
    extend(parser)

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--output-dir', type=str, default="./output")
    parser.add_argument('--input-dir', type=str, default="./input_data/processed/default")
    parser.add_argument('--model-dir', type=str, default="./output/model/default")

    # Used in train / predict / build
    parser.add_argument('--limit', type=int, default=None, help="How many rows of input data to read")

    # --------------------------------------------------------------------------
    # Data build
    # --------------------------------------------------------------------------

    parser.add_argument('--eval-holdback', type=float, default=0.1)

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    parser.add_argument('--warm-start-dir', type=str, default=None,
                        help="Load model initial weights from previous checkpoints")

    parser.add_argument('--batch-size', type=int, default=32, help="Number of items in a full batch")
    parser.add_argument('--max-steps', type=int, default=None)

    parser.add_argument('--max-gradient-norm', type=float, default=4)
    parser.add_argument('--learning-rate', type=float, default=1E-2)

    # --------------------------------------------------------------------------
    # Network topology
    # --------------------------------------------------------------------------


    parser.add_argument('--output-activation', type=str, default="mi")
    parser.add_argument('--output-layers', type=int, default=2)

    parser.add_argument('--enable-lr-finder', action='store_true', dest="use_lr_finder")
    parser.add_argument('--enable-lr-decay', action='store_true', dest="use_lr_decay")

    parser.add_argument('--enable-tf-debug', action='store_true', dest="use_tf_debug")
    parser.add_argument('--enable-comet', action='store_true', dest="use_comet")

    args = vars(parser.parse_args())

    args["modes"] = ["eval", "train"]

    for i in [*args["modes"], "all"]:
        args[i + "_input_path"] = os.path.join(args["input_dir"], i + "_input.tfrecords")

    global_args.clear()
    global_args.update(args)

    # Expand activation args to callables
    act = {
        "tanh": tf.tanh,
        "relu": tf.nn.relu,
        "sigmoid": tf.nn.sigmoid,
    }

    for i in ["output_activation", "read_activation"]:
        args[i] = act[args[i].lower()]

    return args


def save_args(args):
    pathlib.Path(args["model_dir"]).mkdir(parents=True, exist_ok=True)
    with tf.gfile.GFile(os.path.join(args["config_path"]), "w") as file:
        yaml.dump(args, file)