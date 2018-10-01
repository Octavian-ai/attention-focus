import tensorflow as tf
import logging
import pathlib
import math
from collections import Counter
from ..build_data import Partitioner, TwoLevelBalancer
from ..build_data import vectors
from ..schema import schema

from ..train.input import input_fn
from ..train.estimator import get_estimator
from ..args import get_args, save_args

logger = logging.getLogger(__name__)


def configure_logging(args):
    logging.basicConfig()
    logger.setLevel(args["log_level"])


def build(args):
    pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)

    if args["kb_vector_type"] in {'orthogonal', 'positive'}:
        vectors.init(args)

    question_types = Counter()
    output_classes = Counter()

    logger.info("Generate TFRecords")
    with Partitioner(args) as p:
        with TwoLevelBalancer(lambda d: str(d["answer"]), lambda d: d["question_type"], p,
                              args["balance_batch"]) as balancer:
            for i, doc in enumerate(vectors.gen_forever(args)):
                logger.debug("Generating #: %s (%s/%s)", i, p.written, args["number_of_questions"])
                record = schema.generate_record(args, doc)
                question_types[doc["question_type"]] += 1
                output_classes[str(doc["answer"])] += 1
                balancer.add(doc, record)
                if p.written >= args["number_of_questions"]:
                    break

        logger.info(f"Class distribution: {p.answer_classes}")

        logger.info(f"Wrote {p.written} TFRecords")


def find_learning_rate(args):
    args = dict(args)
    args.update({
        "use_lr_finder": True,
        "use_lr_decay": False,
        "use_summary_scalar": True,
        "max_steps": 20000,
    })
    i=0
    learning_rate = args["finder_initial_lr"]
    main_output_dir=args['output_dir']
    main_model_dir=args['model_dir']
    while True:
        i+=1
        args.update({
            "output_dir":main_output_dir+f'finder{i}',
            "model_dir":main_model_dir+f'finder{i}',
            "finder_initial_lr": learning_rate
        })
        train(args)
        rerun_q = input("Do you want to re-run the learning rate finder with a new starting rate? (1/0)")
        rerun = bool(int(rerun_q))
        if rerun:
            print("Re-running lr finder")
        else:
            print("Preparing for train/eval run")
        learning_rate = float(input("Enter the learning rate you want to use (e.g. 1e-4):"))
        if not rerun:
            return learning_rate

def train(args):

    hooks = None

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(args, "train"),
        max_steps=args["max_steps"] if args["max_steps"] is not None else 200000,
        hooks=hooks)

    eval_spec = tf.estimator.EvalSpec(
        steps=10,
        start_delay_secs=5,
        input_fn=lambda: input_fn(args, "eval"),
        throttle_secs=30)

    estimator = get_estimator(args)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    args = get_args()

    # TODO: don't regenerate data if it already exists
    build(args)

    args['learning_rate'] = find_learning_rate(args)
    args["use_lr_finder"] = False
    args["use_lr_decay"] = True
    save_args(args)

    train(args)
