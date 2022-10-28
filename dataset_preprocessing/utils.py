import os
import pickle

from pathlib import Path
from toolz import keyfilter
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

AGG_PREFIX = 'aggregated_'  # prefix to aggregated feature's name
USER_INDEX_KEY = '_user_index'


def get_tensorflow_dataset(dataset: pd.DataFrame, features: List[str],
                           date_column: str, keep_columns: Optional[List[str]] = None,
                           cutoff: int = 40, num_oov_indices: int = 10,
                           ) -> Tuple[Dict[str, tf.Tensor],
                                      Dict[str, tf.keras.layers.StringLookup]]:
    """
    Transform pandas.DataFrame `dataset` into dict of tensorflow.Tensor
    and apply tensorflow.StringLookup encoder on `features` columns
    :param dataset: initial pandas DataFrame
    :param features: list of features columns to transform using StringLookup
    :param date_column: mandatory name of date column, that will be casted in day integer
    :param keep_columns: list of columns' names to keep as is and just convert into Tensor
    :param cutoff: minimal number of occurrences for a feature value to be in StringLookup.vocabulary
    :param num_oov_indices: number of out-of-vocabulary indices for StringLookup
    :return: tuple with the resulting dict of Tensor and the dict with inverse transformations for each StringLookup
    """
    tf_tensors = {date_column: tf.cast(dataset[date_column].values.astype('datetime64[D]').astype(int), tf.int32)}
    for col in keep_columns:
        tf_tensors[col] = tf.convert_to_tensor(dataset[col])
    inverse_lookups = {}
    for feature in features:
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


def save_inverse_lookups(inverse_lookups: Dict[str, tf.keras.layers.StringLookup], path: str):
    """
    Save dict with inverse StringLookup transformations to a given path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({name: (ll.get_vocabulary(), ll.num_oov_indices) for name, ll in inverse_lookups.items()}, f)


def load_inverse_lookups(path: str) -> Dict[str, tf.keras.layers.StringLookup]:
    """
    Load dict with inverse StringLookup transformations saved using `save_inverse_lookups`
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return {name: tf.keras.layers.StringLookup(vocabulary=vocabulary, invert=True, name=name,
                                               num_oov_indices=num_oov_indices)
            for name, (vocabulary, num_oov_indices) in obj.items()}


@tf.function(experimental_relax_shapes=True)
def gather_structure(structure, indices):
    """
    Apply tf.gather through structure (typically dict) using map_structure
    >>> gather_structure({'a': tf.range(5)}, [3, 4])
    {'a': <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>}
    """
    return tf.nest.map_structure(lambda x: tf.gather(x, indices), structure)


@tf.function(experimental_relax_shapes=True)
def boolean_mask_structure(structure, mask):
    """
    Apply tf.boolean_mask through structure (typically dict) using map_structure
    >>> boolean_mask_structure({'a': tf.range(5)}, [True, False, False, True, False])
    {'a': <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 3], dtype=int32)>}
    """
    return tf.nest.map_structure(lambda x: tf.boolean_mask(x, mask), structure)


def enforce_unique_values(dictionary: Dict[Any, str]) -> dict:
    """
    Guarantee unique values in dict by adding _ symbols to repeated
    >>> enforce_unique_values({'a': 'v', 'b': 'v', 'c': 'v'})
    {'a': 'v', 'b': 'v_', 'c': 'v__'}
    """
    seen_values = set()
    for k, v in dictionary.items():
        while v in seen_values:
            v += '_'
        seen_values.add(v)
        dictionary[k] = v
    return dictionary


def get_user_sequences(tensors_dict_by_event: Dict[str, Dict[str, tf.Tensor]], target: str, user_id_column: str) \
        -> Dict[str, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
    """
    Add to a non partitioned dict `tensors_dict_by_event` additional RaggedTensor indexing events done by the same user
    :param tensors_dict_by_event: initial dictionary of the form {event_type -> {feature -> tensor}}
    :param target: name of target event that will be used to index users (usually the same event we want to predict)
    :param user_id_column: name of tensor containing user ids
    :return: dict with the same structure as initial `tensors_dict_by_event`,
    restricted to users present in tensors_dict_by_event[target]
    where to each event_type dict we add a new key _user_index with local indices of events done by the same user.
    One can obtain event sequences of a user with
    >>> tf.gather(result[event_type][feature], result[event_type]['_user_index'])
    """
    # TODO consider batch user_id indexing

    # first we label encode user indices using target event_type
    target_users = tensors_dict_by_event[target][user_id_column]
    unique_users = tf.unique(target_users)[0]
    n_unique_users = tf.shape(unique_users)[0]
    user_indexer = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(unique_users, tf.range(n_unique_users)),
        -1)
    result = {}
    for event_type, tensors_dict in tensors_dict_by_event.items():
        # now we apply label encoding on each event_type in dataset
        user_indices = user_indexer.lookup(tensors_dict[user_id_column])
        # we restric to users present in target event
        _defined_user_indices = tf.squeeze(tf.where(user_indices != -1), axis=1)
        tensors_dict = gather_structure(tensors_dict, _defined_user_indices)
        user_indices = tf.gather(user_indices, _defined_user_indices)

        # for each event_type we keep track where each user id appears
        # so by looking at tf.gather(tensors_dict[feature], tensors_dict[USER_INDEX_KEY])
        # we obtain all values of given feature in user's event sequence
        _line_numbers = tf.cast(tf.range(tf.shape(user_indices)[0]), tf.int64)
        tensors_dict[USER_INDEX_KEY] = tf.ragged.stack_dynamic_partitions(
            _line_numbers, user_indices, n_unique_users)

        result[event_type] = tensors_dict
    return result


def _reindex_users(user_index: tf.RaggedTensor) -> tf.RaggedTensor:
    """
    Given a subset of user_index replace line indexing to be between 0 and n events done by users' subset
    """
    lens = user_index.row_lengths()
    return tf.RaggedTensor.from_row_lengths(tf.range(tf.reduce_sum(lens)), lens)


def restrict_to_user_index_subset(tensors_dict: Dict[str, Union[tf.Tensor, tf.RaggedTensor]], indices: tf.Tensor) \
        -> Dict[str, Union[tf.Tensor, tf.RaggedTensor]]:
    """
    Restrict a dataset to a subset of users
    :param tensors_dict: dictionary {feature -> tf.Tensor} containing USER_INDEX_KEY
    :param indices: selection of users (between 0 and number of unique users)
    :return: `tensors_dict` limited to users from `indices` and with updated USER_INDEX_KEY
    """
    user_index_subset = tf.gather(tensors_dict[USER_INDEX_KEY], indices)
    result = {USER_INDEX_KEY: _reindex_users(user_index_subset)}
    result.update(
        gather_structure(keyfilter(lambda x: x not in [USER_INDEX_KEY], tensors_dict), user_index_subset.values))
    return result


def batch_by_user(tensors_dict_by_event: Dict[str, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]],
                  target: str, batch_size: int, seed: Optional[int] = None) -> tf.data.Dataset:
    """
    Batch a non partitioned dict `tensors_dict_by_event` by users
    :param tensors_dict_by_event: initial dictionary of the form {event_type -> {feature -> tensor}}
    :param target: name of target event that will be used to index users (usually the same event we want to predict)
    :param batch_size: number of unique users in batch
    :param seed: optional random seed
    :return: tf.data.Dataset where each batch has the same structure as `tensors_dict_by_event`,
    but limited to `batch_size` unique users. Note that last not-full batch is kept.
    """
    nb_users = tensors_dict_by_event[target][USER_INDEX_KEY].bounding_shape()[0]
    permutation = tf.random.shuffle(tf.range(nb_users), seed=seed)
    dataset = None
    for i in range((nb_users // batch_size) + 1):
        # indices of users chosen for current batch
        local_permutation = permutation[i * batch_size: (i + 1) * batch_size]
        batch = {}
        for table, tensors_dict in tensors_dict_by_event.items():
            batch[table] = restrict_to_user_index_subset(tensors_dict, local_permutation)

        batch_ds = tf.data.Dataset.from_tensors(batch)
        dataset = batch_ds if i == 0 else dataset.concatenate(batch_ds)
    return dataset


@tf.function(input_signature=[(tf.TensorSpec(shape=None, dtype=tf.int32),
                               tf.TensorSpec(shape=None, dtype=tf.int32),
                               tf.TensorSpec(shape=None, dtype=tf.int32))],
             experimental_relax_shapes=True)
def _indices_of_preceding_dates_one_user(args: tuple, max_elements: int = 100) -> tf.RaggedTensor:
    """
    For each unique user and for each date of target event find dates of actions with preceding dates
    :param args: tuple with (dates corresponding to user in target event,
                             dates corresponding to user in event we want to aggregate,
                             _user_index of event we want to aggregate)
    :param max_elements: maximal number of preceding dates to keep
    :return: ragged tensor with first dimension corresponding to actions in target event
        and ragged dimension containing list of indices from _user_index corresponding to preceding dates
        in event we want to aggregate

    In example here target dates are [1, 2, 2, 5], dates to aggegate [0, 1, 1, 3, 5] found on lines [0, 2, 4, 7, 8]
    so preceding dates to
    * 1 are [0], corresponding to indices [0]
    * 2 are [0, 1, 1], corresponding to indices [0, 2, 8]
    * 5 are [0, 1, 3, 1], corresponding to indices [0, 2, 7, 8]
    >>> _indices_of_preceding_dates_one_user(([1, 2, 2, 5], [0, 1, 5, 3, 1], [0, 2, 4, 7, 8]))
    <tf.RaggedTensor [[0], [0, 2, 8], [0, 2, 8], [0, 2, 7, 8]]>
    """
    target_dates, dates_to_agg, user_index_to_agg = args
    dates_diff = tf.expand_dims(target_dates, axis=1) - tf.expand_dims(dates_to_agg, axis=0)
    # we can set maximal date diff here to drop old events as `mask = tf.logical_and(dates_diff > 0, dates_diff < 360)`
    mask = dates_diff > 0
    mask.set_shape([None, None])  # needed to make tf.ragged.boolean_mask guess the rank
    values = tf.repeat(tf.expand_dims(user_index_to_agg, 0), tf.shape(target_dates)[0], axis=0)
    full_indices = tf.ragged.boolean_mask(values, mask).with_row_splits_dtype(tf.int32)
    return full_indices[:, :max_elements]


@tf.function(experimental_relax_shapes=True)
def _indices_of_preceding_dates(target_dates_by_user: tf.RaggedTensor,
                                dates_to_agg_by_user: tf.RaggedTensor,
                                user_index_to_agg_by_user: tf.RaggedTensor) -> tf.RaggedTensor:
    """
    Execute _indices_of_preceding_dates_one_user for each user separately using map_fn
    Arguments are the same as for _indices_of_preceding_dates_one_user, but put into ragged tensors
    with first dimension corresponding to unique users.
    :return: ragged tensor with first dimension corresponding to all target events (user dimension is flatten) and
        ragged dimension with list of indices of events we want to aggregate preceding chosen target event
    """
    return tf.map_fn(_indices_of_preceding_dates_one_user,
                     (target_dates_by_user, dates_to_agg_by_user, user_index_to_agg_by_user),
                     fn_output_signature=tf.RaggedTensorSpec(shape=[None, None],
                                                             dtype=tf.int32,
                                                             row_splits_dtype=tf.int32),
                     parallel_iterations=10).merge_dims(0, 1)


def aggregate_preceding_events(tensors_dict_by_event: Dict[str, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]],
                               target: str, features: List[str], user_id_column: str, date_column: str) \
        -> Dict[str, tf.RaggedTensor]:
    """
    Generate historical features for given target event:
        for each target event / event type to aggregate / feature
        we will get a list of feature's values corresponding to events to aggregate and happened before target event
    :param tensors_dict_by_event: initial dictionary of the form {event_type -> {feature -> tensor}}
    :param target: name of target event we want to generate features for
    :param features: list of features' names
    :param user_id_column: name of tensor containing user ids
    :param date_column: name of tensor with dates
    :return: dict with resulting features {feature -> ragged tensor}
        with tensor's first dim corresponding to unique users,
        second to target events done by user
        and third to historical events preceding a target event
    """
    result = {}
    target_dates_by_user = tf.gather(tensors_dict_by_event[target][date_column],
                                     tensors_dict_by_event[target][USER_INDEX_KEY])
    for event, tensors_dict in tensors_dict_by_event.items():
        user_index_to_agg_by_user = tensors_dict[USER_INDEX_KEY]
        indices = _indices_of_preceding_dates(
            target_dates_by_user,
            tf.gather(tensors_dict[date_column], tensors_dict[USER_INDEX_KEY]),
            user_index_to_agg_by_user)
        for feature in features:
            result[f'{AGG_PREFIX}{event}_{feature}'] = tf.RaggedTensor.from_row_splits(
                tf.gather(tensors_dict[feature], indices),
                tensors_dict_by_event[target][USER_INDEX_KEY].row_splits)
    # we also add non-aggregated features, user id and date for each line
    for feature in list(features) + [user_id_column, date_column]:
        result[feature] = tf.gather(tensors_dict_by_event[target][feature],
                                    tensors_dict_by_event[target][USER_INDEX_KEY])
    return result
