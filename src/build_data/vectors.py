import itertools
import numpy as np
import tensorflow as tf
import math
import random
import timeit as t
from sympy.utilities.iterables import multiset_permutations

VECTOR_LENGTH=12
mid_point=int(math.floor(VECTOR_LENGTH/2))
RANGE=2
low_point=mid_point-RANGE
high_point=mid_point+RANGE
identity_matrix = np.eye(VECTOR_LENGTH, dtype=np.int64)

def get_bool_permutations():
    all_permutations = []
    for i in range(VECTOR_LENGTH):
        print(f"calculating permutations for {i}")
        values = np.concatenate([np.ones(i, dtype=np.int64), np.zeros(VECTOR_LENGTH-i, dtype=np.int64)])
        all_permutations.extend(multiset_permutations(values))
    print(f"calculated {len(all_permutations)} permutations")
    print(f"getting on with life")
    as_array = np.array(all_permutations)
    assert len(np.unique(as_array,axis=0)) == len(all_permutations)
    return as_array

ALL_PERMUTATION = get_bool_permutations()

def random_one_hot_vector():
    return identity_matrix[np.random.choice(VECTOR_LENGTH, 1, replace=False)].tolist()[0]

def random_one_hot_set(sz, query):
    return identity_matrix[np.random.choice(VECTOR_LENGTH, sz, replace=False)].tolist()


def query_in_list(list, query):
    assert len(query.shape) == 1
    return np.equal(list,query).all(1).any()

def random_permutation_vector():
    return ALL_PERMUTATION[np.random.choice(len(ALL_PERMUTATION), 1, replace=False)][0]

def random_permutation_set(sz, query):
    if bool(random.getrandbits(1)):
        perm_set = ALL_PERMUTATION[np.random.choice(len(ALL_PERMUTATION), sz, replace=False)]
        if not query_in_list(perm_set, query):
            perm_set[np.random.randint(0,sz)] = query
    else:
        perm_set_plus_one = ALL_PERMUTATION[np.random.choice(len(ALL_PERMUTATION), sz + 1, replace=False)]
        if query_in_list(perm_set_plus_one, query):
            perm_set = perm_set_plus_one[~np.equal(perm_set_plus_one, query).all(axis=1)]
        else:
            perm_set = perm_set_plus_one[:-1]

    assert len(perm_set) == sz
    return perm_set

print("example vectors")
print(random_permutation_vector())
print(random_permutation_set(2,random_permutation_vector()))
vector_time = t.timeit(random_permutation_vector, number=100)
print(vector_time)
set_time = t.timeit(lambda: random_permutation_set(2, random_permutation_vector()), number=100)
print(set_time)
# Assert data generation is not too slow
assert set_time < 10


def gen_forever(list_size):
  for i in itertools.count():
      query = random_permutation_vector()
      list = random_permutation_set(list_size, query)
      answer = [1,0] if query_in_list(list, query) else [0,1]

      if i % (2000) == 0:
          print(i)
          print(query)
          print(len(list))
          print(answer)
      yield {'query': query, 'list': list, "answer": answer, "question_type": "existence"}