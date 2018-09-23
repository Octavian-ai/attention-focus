import tensorflow as tf
import pathlib
from collections import Counter

from .util import *
from .balancer import TwoLevelBalancer
from .vectors import gen_forever
from .schema import generate_record

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------





# --------------------------------------------------------------------------
# Run the script
# --------------------------------------------------------------------------

if __name__ == "__main__":

    args = {
        "balance_batch": 20,
        "N":10000,
        "modes":{'train', 'eval'},
        "input_dir": "data",
        "train_input_path": "data/train",
        "eval_input_path": "data/eval",
        "eval_holdback": 0.5,
        "predict_holdback": 0,
        "log_level": "DEBUG",
    }

    logging.basicConfig()
    logger.setLevel(args["log_level"])
    logging.getLogger("input.util").setLevel(args["log_level"])

    try:
        pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    question_types = Counter()
    output_classes = Counter()

    logger.info("Generate TFRecords")
    with Partitioner(args) as p:
        with TwoLevelBalancer(lambda d: str(d["answer"]), lambda d: d["question_type"], p,
                              args["balance_batch"]) as balancer:
            for i, doc in enumerate(gen_forever(4)):
                logger.debug("Generating #: %s (%s/%s)", i, p.written, args["N"])
                record = generate_record(args, doc)
                question_types[doc["question_type"]] += 1
                output_classes[str(doc["answer"])] += 1
                balancer.add(doc, record)
                if p.written >= args["N"]:
                    break

        logger.info(f"Class distribution: {p.answer_classes}")

        logger.info(f"Wrote {p.written} TFRecords")






