import os
import pickle

from functools import partial

from pathlib import Path
from toolz import keyfilter
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

AGG_PREFIX = 'aggregated_'  # prefix to aggregated feature's name
WEIGHT_SUFFIX = '_weight'  # suffix to weight columns used in aggregation
ADDITIONAL_NEGATIVES = 2  # generate first more negatives than asked and restrict after to avoid collisions


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
def merge_dims(outer_axis, inner_axis, structure):
    """
    TODO
    """
    return tf.nest.map_structure(lambda x: x.merge_dims(outer_axis, inner_axis), structure)


def add_equal_weights(batch, y, features):
    for col in features:
        weight_col = col + WEIGHT_SUFFIX
        # transform batch[col] into one-hot ragged
        batch[col] = tf.RaggedTensor.from_uniform_row_length(batch[col], 1)
        batch[weight_col] = batch[col].with_values(tf.ones(shape=(len(batch[col].values),), dtype=tf.float32))
    return batch, y


def _keep_nb_events_by_user_by_date(batch, date_column, nb, seed=None):
    # first we get all pairs user/date encoded as integers
    users = tf.cast(batch[date_column].value_rowids(), tf.int64)
    dates = tf.cast(batch[date_column].values, tf.int64)
    user_dates = dates + users * tf.reduce_max(dates)
    n_pairs = tf.shape(user_dates)[0]
    
    # we permute pairs in order to be able to get first nb pairs for each user below
    permutation = tf.random.shuffle(tf.range(n_pairs, dtype=tf.int32), seed=seed)
    
    # group by indices inside permutation by unique user/date pair
    unique_pairs, indices = tf.unique(tf.gather(user_dates, permutation))
    n_unique_pairs = tf.shape(unique_pairs)[0]
    grouped_indices = tf.ragged.stack_dynamic_partitions(tf.range(n_pairs, dtype=tf.int32), indices, n_unique_pairs)
    # keep first nb pairs in each group, then go back to permutation indices
    to_keep = tf.gather(permutation, grouped_indices[:,:nb].values)
    return gather_structure(merge_dims(0, 1, batch), to_keep)


def _rebatch_and_order_by_date(batch, batch_size, date_column, max_iter, seed):
    nb_events = tf.shape(batch[date_column])[0]
    permutation = tf.random.shuffle(tf.range(nb_events), seed=seed)
    dataset = tf.data.Dataset.from_tensors(gather_structure(batch, permutation[:0]))
    for i in range(max_iter):
        # taking a batch
        local_permutation = permutation[i * batch_size: (i + 1) * batch_size]
        # ordering by date_column - needed for following negatives generation
        local_permutation = tf.gather(local_permutation, 
                                      tf.argsort(tf.gather(batch[date_column], local_permutation)))
        dataset = dataset.concatenate(tf.data.Dataset.from_tensors(gather_structure(batch, local_permutation)))
    return dataset


def _keep_full_batches(batch, batch_size):
    any_column = list(batch.keys())[0]
    return tf.shape(batch[any_column])[0] == batch_size


def _add_constant_response(batch):
    any_column = list(batch.keys())[0]
    batch_size = tf.shape(batch[any_column])[0]
    return batch, tf.ones(shape=(batch_size,), dtype=tf.float32)
    

def rebatch_by_events(dataset, batch_size, date_column, nb_events_by_user_by_day=8, seed=None):
    # we also merge 0 (users) and 1 (events) axes inside _keep_nb_events_by_user_by_date
    dataset = dataset.map(partial(_keep_nb_events_by_user_by_date, date_column=date_column,
                                  nb=nb_events_by_user_by_day, seed=seed))

    # searching into how many event indexed batches we can split initial batch
    # we need it to be define statically to pass as arg into flat_map
    max_iter = 0
    for batch in dataset:
        max_iter = max(batch[date_column].shape[0] // batch_size + 1, max_iter)
    
    # shuffle batches in case of non-determinist call (for train)
    if not seed:
        dataset = dataset.shuffle(5)
    
    return dataset\
        .flat_map(partial(_rebatch_and_order_by_date, batch_size=batch_size, date_column=date_column,
                          max_iter=max_iter, seed=seed))\
        .filter(partial(_keep_full_batches, batch_size=batch_size))\
        .map(_add_constant_response)


def _response_negatives_in_minibatch(y_true, number_of_negatives):
    if not number_of_negatives:
        return y_true
    batch_size = tf.shape(y_true)[0]
    k = number_of_negatives + 1
    out_indices = tf.range(batch_size * k)
    row_i_in_minibatch = out_indices % k
    column_j_in_minibatch = out_indices % (k ** 2) // k
    return tf.reshape(tf.cast(row_i_in_minibatch == column_j_in_minibatch, tf.float32), (-1, 1))


class BroadcastLoss(tf.keras.losses.Loss):
    """
    Loss broadcasting ground truth to the output shape of the model
    needed in case when model output contains predictions for several augmentations
    >>> broadcast_loss = BroadcastLoss(tf.keras.metrics.mean_absolute_error)
    >>> broadcast_loss(tf.ones([3, 1]), tf.constant([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]))
    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
    """
    def __init__(self, loss_obj, number_of_negatives=None, **kwargs):
        self.loss_obj = loss_obj
        self.number_of_negatives = number_of_negatives
        super().__init__(reduction=self.loss_obj.reduction, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.broadcast_to(_response_negatives_in_minibatch(y_true, self.number_of_negatives), tf.shape(y_pred))
        return self.loss_obj(y_true, y_pred)
    
    
class BroadcastMetric(tf.keras.metrics.Metric):
    """
    Same as BroadcastLoss, but for Metric: broadcasting ground truth to the output shape of the model
    needed in case when model output contains predictions for several augmentations
    """
    def __init__(self, metric_obj, number_of_negatives=None, **kwargs):
        self.metric_obj = metric_obj
        self.number_of_negatives = number_of_negatives
        if 'name' not in kwargs:
            kwargs['name'] = self.metric_obj.name
        super().__init__(**kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.broadcast_to(_response_negatives_in_minibatch(y_true, self.number_of_negatives), tf.shape(y_pred))
        if sample_weight is not None:
            sample_weight = tf.broadcast_to(_response_negatives_in_minibatch(sample_weight, self.number_of_negatives),
                                            tf.shape(y_pred))
        return self.metric_obj.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return self.metric_obj.result()

    def reset_state(self):
        self.metric_obj.reset_state()
