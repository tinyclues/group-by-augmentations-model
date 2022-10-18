import os
import pickle

from pathlib import Path
from toolz import keyfilter
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))


def get_tensorflow_dataset(dataset: pd.DataFrame, item_features: List[str],
                           user_id_column: str, date_column: str,
                           cutoff: int = 40, num_oov_indices: int = 10,
                           ) -> Tuple[Dict[str, tf.Tensor],
                                      Dict[str, tf.keras.layers.StringLookup]]:
    tf_tensors = {user_id_column: tf.convert_to_tensor(dataset[user_id_column]),# tf.strings.as_string(
                  date_column: tf.cast(dataset[date_column].values.astype('datetime64[D]').astype(int), tf.int32)}
    inverse_lookups = {}
    for feature in item_features:
        print(f'Encoding {feature} column')
        column = dataset[feature].astype(str)
        # counting categories frequencies
        _counts = column.value_counts()
        _n_categories = len(_counts)
        vocabulary = _counts[_counts >= cutoff].index.values
        print(f'Reserving labels for {len(vocabulary)} categories out of {_n_categories}')
        # just a technical assert for safe cast of labels to int32
        assert len(vocabulary) < np.iinfo(np.int32).max
        lookup_kwargs = {'name': feature,
                         'num_oov_indices': num_oov_indices,
                         'vocabulary': tf.convert_to_tensor(vocabulary)}
        inverse_lookups[feature] = tf.keras.layers.StringLookup(invert=True, **lookup_kwargs)
        _lookup = tf.keras.layers.StringLookup(**lookup_kwargs)
        tf_tensors[feature] = tf.cast(_lookup(column), tf.int32)

    return tf_tensors, inverse_lookups


def save_inverse_lookups(inverse_lookups, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({name: (ll.get_vocabulary(), ll.num_oov_indices) for name, ll in inverse_lookups.items()}, f)
        
        
def load_inverse_lookups(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return {name: tf.keras.layers.StringLookup(vocabulary=vocabulary, invert=True, name=name, num_oov_indices=num_oov_indices)
            for name, (vocabulary, num_oov_indices) in obj.items()}


@tf.function(experimental_relax_shapes=True)
def gather_structure(structure, indices):
    return tf.nest.map_structure(lambda x: tf.gather(x, indices), structure)


def enforce_unique_values(dictionary: Dict[Any, str]):
    seen_values = set()
    for k, v in dictionary.items():
        while v in seen_values:
            v += '_'
        seen_values.add(v)
        dictionary[k] = v
    return dictionary


def get_user_sequences(datasets: Dict[str, Dict[str, tf.Tensor]], target: str, user_id_column: str):
    # how can we batch user_id indexing ?

    # first we label encode user indices using target event_type
    target_users = datasets[target][user_id_column]
    unique_users = tf.unique(target_users)[0]
    n_unique_users = tf.shape(unique_users)[0]
    user_indexer = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(unique_users, tf.range(n_unique_users)),
        -1)
    for event_type, tensors_dict in datasets.items():
        # now we apply label encoding on each event_type in dataset
        user_indices = user_indexer.lookup(tensors_dict[user_id_column])
        _defined_user_indices = tf.squeeze(tf.where(user_indices != -1), axis=1)
        tensors_dict = gather_structure(tensors_dict, _defined_user_indices)
        user_indices = tf.gather(user_indices, _defined_user_indices)

        # for each event_type we keep track where each user id appears
        # so by looking at tf.gather(tensors_dict[feature], tensors_dict['_user_index'])
        # we obtain all values of given feature in user's event sequence
        _line_numbers = tf.cast(tf.range(tf.shape(user_indices)[0]), tf.int64)
        tensors_dict['_user_index'] = tf.ragged.stack_dynamic_partitions(
            _line_numbers, user_indices, n_unique_users)

        datasets[event_type] = tensors_dict
    return datasets


def _reindex_users(user_index):
    lens = user_index.row_lengths()
    return tf.RaggedTensor.from_row_lengths(tf.range(tf.reduce_sum(lens)), lens)


def _restrict_to_user_index_subset(tensors_dict, user_index_subset):
    res = {'_user_index': _reindex_users(user_index_subset)}
    res.update(gather_structure(keyfilter(lambda x: x not in ['_user_index'], tensors_dict), user_index_subset.values))
    return res


def batch_by_user(tensors_dict_by_event: Dict[str, Dict[str, tf.Tensor]], target: str, batch_size: int, seed: Optional[int] = None):
    nb_users = tensors_dict_by_event[target]['_user_index'].bounding_shape()[0]
    permutation = tf.random.shuffle(tf.range(nb_users), seed=seed)
    dataset = None
    for i in range((nb_users // batch_size) + 1):
        local_permutation = permutation[i * batch_size: (i + 1) * batch_size]
        batch = {}
        for table, tensors_dict in tensors_dict_by_event.items():
            local_user_index = tf.gather(tensors_dict['_user_index'], local_permutation)
            batch[table] = _restrict_to_user_index_subset(tensors_dict, local_user_index)

        batch_ds = tf.data.Dataset.from_tensors(batch)
        dataset = batch_ds if i == 0 else dataset.concatenate(batch_ds)
    return dataset


@tf.function(input_signature=[(tf.TensorSpec(shape=None, dtype=tf.int32),
                               tf.TensorSpec(shape=None, dtype=tf.int32),
                               tf.TensorSpec(shape=None, dtype=tf.int32))],
             experimental_relax_shapes=True)
def indices_of_preceding_dates_one_user(args, max_elements=100):
    target_dates, dates_to_agg, user_index_to_agg = args
    dates_diff = tf.expand_dims(target_dates, axis=1) - tf.expand_dims(dates_to_agg, axis=0)
    mask = dates_diff > 0 # we can set maximal date diff here to drop old events as `tf.logical_and(dates_diff > 0, dates_diff < 360)`
    mask.set_shape([None, None])  # needed to make tf.ragged.boolean_mask guess the rank
    values = tf.repeat(tf.expand_dims(user_index_to_agg, 0), tf.shape(target_dates)[0], axis=0)
    full_indices = tf.ragged.boolean_mask(values, mask).with_row_splits_dtype(tf.int32)
    return full_indices[:,:max_elements]

    
@tf.function(experimental_relax_shapes=True)
def indices_of_preceding_dates(target_dates_by_user, dates_to_agg_by_user, user_index_to_agg_by_user):
    return tf.map_fn(indices_of_preceding_dates_one_user,
                     (target_dates_by_user, dates_to_agg_by_user, user_index_to_agg_by_user),
                     fn_output_signature=tf.RaggedTensorSpec(shape=[None, None],
                                                             dtype=tf.int32,
                                                             row_splits_dtype=tf.int32),
                     parallel_iterations=10).merge_dims(0, 1)

AGG_PREFIX = 'aggregated_'

def aggregate_preceding_events(batch, target, item_features, user_id_column, date_column):    
    res = {}
    target_dates_by_user = tf.gather(batch[target][date_column], batch[target]['_user_index'])
    for event, tensors_dict in batch.items():
        user_index_to_agg_by_user = tensors_dict['_user_index']
        indices = indices_of_preceding_dates(
            target_dates_by_user,
            tf.gather(tensors_dict[date_column], tensors_dict['_user_index']),
            user_index_to_agg_by_user)
        for feature in item_features:
            res[f'{AGG_PREFIX}{event}_{feature}'] = tf.RaggedTensor.from_row_splits(
                tf.gather(tensors_dict[feature], indices),
                batch[target]['_user_index'].row_splits)
    # we also add raw item features, userId, date for each line
    for feature in list(item_features) + [user_id_column, date_column]:
        res[feature] = tf.gather(batch[target][feature], batch[target]['_user_index'])
    return res
