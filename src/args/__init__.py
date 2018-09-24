import argparse
import os.path
import yaml
import pathlib
import tensorflow as tf
import subprocess
import time

from ..build_data.vectors import vector_type_fns

def get_git_hash():
    result = subprocess.run(
        ['git', '--no-pager', 'log', "--pretty=format:%h", '-n', '1'],
        stdout=subprocess.PIPE,
        check=True,
        universal_newlines=True
    )
    return result.stdout


def get_args(extend=lambda parser: None):
    git_hash = get_git_hash()
    print("git hash", git_hash)
    parser = argparse.ArgumentParser()
    extend(parser)

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------
    default_dir=f"./experiments/{git_hash}"
    timestamp_in_minutes = int(time.time()/60)
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--output-dir', type=str, default=f"{default_dir}/output")
    parser.add_argument('--input-dir', type=str, default=f"{default_dir}/input")
    parser.add_argument('--model-dir', type=str, default=f"{default_dir}/model/{timestamp_in_minutes}")

    # Used in train / predict / build
    parser.add_argument('--finder-initial-lr', type=float, default=1e-06, help="Initial learning rate for the learning rate finder")
    parser.add_argument('--limit', type=int, default=None, help="How many rows of input data to read")
    parser.add_argument('--use-summary-scalar', type=bool, default=False, help="Log training metrics for tensorboard")

    # --------------------------------------------------------------------------
    # Data build
    # --------------------------------------------------------------------------

    parser.add_argument('--eval-holdback', type=float, default=0.1)
    parser.add_argument('--predict-holdback', type=float, default=0)
    parser.add_argument('--balance-batch', type=int, default=20)
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--kb-vector-length', type=int, default=10,
                        help = f"The size of vectors used for the query and the kb list")
    parser.add_argument('--kb-list-size', type=int, default=4,
                        help=f"The number of elements in the kb list")
    parser.add_argument('--kb-vector-type', type=str, default='orthogonal',
                        help = f"What kind of vectors to use: {vector_type_fns.keys()}")

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    parser.add_argument('--warm-start-dir', type=str, default=None,
                        help="Load model initial weights from previous checkpoints")

    parser.add_argument('--batch-size', type=int, default=32, help="Number of items in a full batch")
    parser.add_argument('--max-steps', type=int, default=None)

    parser.add_argument('--max-gradient-norm', type=float, default=8)

    # --------------------------------------------------------------------------
    # Network topology
    # --------------------------------------------------------------------------

    parser.add_argument('--use-attention-focus', type=bool, default=True)

    parser.add_argument('--output-activation', type=str, default="tanh")
    parser.add_argument('--output-layers', type=int, default=2)

    parser.add_argument('--enable-lr-finder', action='store_true', dest="use_lr_finder")
    parser.add_argument('--enable-lr-decay', action='store_true', dest="use_lr_decay")

    parser.add_argument('--enable-tf-debug', action='store_true', dest="use_tf_debug")
    parser.add_argument('--enable-comet', action='store_true', dest="use_comet")

    args = vars(parser.parse_args())

    args["modes"] = ["eval", "train"]

    for i in [*args["modes"], "all"]:
        args[i + "_input_path"] = os.path.join(args["input_dir"], i + "_input.tfrecords")

    # Expand activation args to callables
    act = {
        "tanh": tf.tanh,
        "relu": tf.nn.relu,
        "sigmoid": tf.nn.sigmoid,
    }

    for i in ["output_activation"]:
        args[i] = act[args[i].lower()]

    return args


def save_args(args):
    pathlib.Path(args["model_dir"]).mkdir(parents=True, exist_ok=True)
    with tf.gfile.GFile(os.path.join(args["model_dir"], "config.yaml"), "w") as file:
        yaml.dump(args, file)