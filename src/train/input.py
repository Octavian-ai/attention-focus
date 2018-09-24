import tensorflow as tf

from src.schema.schema import parse_single_example, reconstitute_single_example


def input_fn(args, mode):

    sample = {}
    for record in tf.python_io.tf_record_iterator(args[f"{mode}_input_path"]):
        example = tf.train.Example()
        example.ParseFromString(record)
        f = example.features.feature
        sample["kb_len"] = f['kb_len'].int64_list.value[0]
        sample["query_len"] = f['query_len'].int64_list.value[0]
        sample["label_length"] = f['label_length'].int64_list.value[0]
        break
        # TODO: consider using this and Dataset form generator to allow better deserialisation

    # --------------------------------------------------------------------------
    # Read TFRecords
    # --------------------------------------------------------------------------

    d = tf.data.TFRecordDataset([args[f"{mode}_input_path"]])
    if args["limit"] is not None:
        d = d.take(args["limit"])
    d = d.repeat()
    d = d.shuffle(args["batch_size"] * 1000)
    d = d.batch(args["batch_size"])
    #d = d.batch(1)
    d = d.map(parse_single_example(sample))

    # --------------------------------------------------------------------------
    # Layout input data
    # --------------------------------------------------------------------------

    d = d.map(reconstitute_single_example(sample))








    return d