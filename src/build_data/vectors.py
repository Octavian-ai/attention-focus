import itertools
import numpy as np
import tensorflow as tf
import math
import random
import timeit as t
from sympy.utilities.iterables import multiset_permutations

class vector_settings():
    pass


def init(args):
    vector_length = args["kb_vector_length"]

    vector_settings.VECTOR_LENGTH = vector_length
    vector_settings.identity_matrix = np.eye(vector_length, dtype=np.int64)
    vector_settings.ALL_PERMUTATION = get_bool_permutations()

    print("example vectors")
    print(random_permutation_vector())
    print(random_permutation_set(2, random_permutation_vector()))
    vector_time = t.timeit(random_permutation_vector, number=100)
    print(vector_time)
    set_time = t.timeit(lambda: random_permutation_set(2, random_permutation_vector()), number=100)
    print(set_time)
    # Assert data generation is not too slow
    assert set_time < 10


def get_bool_permutations():
    all_permutations = []
    for i in range(vector_settings.VECTOR_LENGTH):
        print(f"calculating permutations for {i}")
        values = np.concatenate(
            [np.ones(i, dtype=np.int64), np.zeros(vector_settings.VECTOR_LENGTH - i, dtype=np.int64)])
        all_permutations.extend(multiset_permutations(values))
    print(f"calculated {len(all_permutations)} permutations")
    print(f"getting on with life")
    as_array = np.array(all_permutations)
    assert len(np.unique(as_array, axis=0)) == len(all_permutations)
    return as_array


def random_one_hot_vector():
    return vector_settings.identity_matrix[np.random.choice(vector_settings.VECTOR_LENGTH, 1, replace=False)][0]


def random_one_hot_set(sz, query):
    return vector_settings.identity_matrix[np.random.choice(vector_settings.VECTOR_LENGTH, sz, replace=False)]


def query_in_list(list, query):
    assert len(query.shape) == 1
    return np.equal(list, query).all(1).any()


def random_permutation_vector():
    return vector_settings.ALL_PERMUTATION[np.random.choice(len(vector_settings.ALL_PERMUTATION), 1, replace=False)][0]


def random_permutation_set(sz, query):
    if bool(random.getrandbits(1)):
        perm_set = vector_settings.ALL_PERMUTATION[
            np.random.choice(len(vector_settings.ALL_PERMUTATION), sz, replace=False)]
        if not query_in_list(perm_set, query):
            perm_set[np.random.randint(0, sz)] = query
    else:
        perm_set_plus_one = vector_settings.ALL_PERMUTATION[
            np.random.choice(len(vector_settings.ALL_PERMUTATION), sz + 1, replace=False)]
        if query_in_list(perm_set_plus_one, query):
            perm_set = perm_set_plus_one[~np.equal(perm_set_plus_one, query).all(axis=1)]
        else:
            perm_set = perm_set_plus_one[:-1]

    assert len(perm_set) == sz
    return perm_set


vector_type_fns = {
    "orthogonal_query": random_one_hot_vector,
    "orthogonal_list": random_one_hot_set,
    "positive_query": random_permutation_vector,
    "positive_list": random_permutation_set,
}


def gen_forever(args):
    list_size = args["kb_list_size"]
    vector_type = args["kb_vector_type"]

    query_fn = vector_type_fns[f"{vector_type}_query"]
    list_fn = vector_type_fns[f"{vector_type}_list"]

    for i in itertools.count():
        query = query_fn()
        list = list_fn(list_size, query)

        answer = [0, 1] if query_in_list(list, query) else [1, 0]

        if i % (2000) == 0:
            print(i)
            print(query)
            print(len(list))
            print(answer)

        yield {'query': query, 'list': list, "answer": answer, "question_type": "existence"}
