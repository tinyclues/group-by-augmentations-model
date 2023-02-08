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


@tf.function(experimental_relax_shapes=True)
def _generate_negatives_one_batch(batch, batch_size, max_iter, number_of_negatives, offer_features,
                                  user_id_column, date_column, seed=None):
    nb_events = tf.shape(batch[user_id_column])[0]
    permutation = tf.random.shuffle(tf.range(nb_events), seed=seed)
    dataset = tf.data.Dataset.from_tensors(gather_structure({'response': tf.fill([0], 1), **batch}, permutation[:0]))
    for i in range(max_iter):
        local_permutation = permutation[i * batch_size: (i + 1) * batch_size]
        
        # making batch size divisible by number_of_negatives + 1 + ADDITIONAL_NEGATIVES
        # we need to do it for the last smaller batch
        effective_batch_size = tf.shape(local_permutation)[0]
        effective_batch_size -= effective_batch_size % (number_of_negatives + 1 + ADDITIONAL_NEGATIVES)
        
        # we order a batch by dates to get negatives from the same day
        order_dates_idx = tf.argsort(tf.gather(batch[date_column], local_permutation[:effective_batch_size]))
        local_permutation = tf.gather(local_permutation, order_dates_idx)
        
        # inside every minibatch we have number of negatives + 1 elements
        mini_batches = tf.reshape(local_permutation, (-1, number_of_negatives + 1 + ADDITIONAL_NEGATIVES))
        # we then will consider all pairs of indices from minibatch to say
        # if such (user, item) example is not positive it is negative
        n_combinations = (number_of_negatives + 1 + ADDITIONAL_NEGATIVES) ** 2
        negative_user_idx = tf.reshape(tf.repeat(mini_batches, number_of_negatives + 1 + ADDITIONAL_NEGATIVES, axis=0),
                                       (-1, n_combinations))
        negative_item_idx = tf.reshape(tf.repeat(mini_batches, number_of_negatives + 1 + ADDITIONAL_NEGATIVES, axis=1),
                                       (-1, n_combinations))
        true_negatives = tf.gather(batch[user_id_column], negative_user_idx) != tf.gather(batch[user_id_column], negative_item_idx)
        
        negative_user_idx = tf.boolean_mask(negative_user_idx, true_negatives)
        negative_item_idx = tf.boolean_mask(negative_item_idx, true_negatives)
        keep_exact_number = tf.random.shuffle(tf.range(tf.shape(negative_user_idx)[0]),
                                              seed=seed)[:effective_batch_size * number_of_negatives]

        user_idx = tf.concat([local_permutation, tf.gather(negative_user_idx, keep_exact_number)], axis=0)
        item_idx = tf.concat([local_permutation, tf.gather(negative_item_idx, keep_exact_number)], axis=0)
        response = tf.concat([tf.fill([effective_batch_size], 1), tf.zeros_like(keep_exact_number)], axis=0)
        
        new_batch = {'response': response}
        for feature in batch.keys():
            if feature in offer_features:
                new_batch[feature] = tf.gather(batch[feature], item_idx)
            else:
                new_batch[feature] = tf.gather(batch[feature], user_idx)
        
        dataset = dataset.concatenate(tf.data.Dataset.from_tensors(new_batch))
    return dataset


def _add_weights(batch, offer_features):
    for col in offer_features:
        weight_col = col + WEIGHT_SUFFIX
        # transform batch[col] into one-hot ragged
        batch[col] = tf.RaggedTensor.from_uniform_row_length(batch[col], 1)
        batch[weight_col] = batch[col].with_values(tf.ones(shape=(len(batch[col].values),), dtype=tf.float32))
    return batch


def _pop_response(batch):
    response = batch.pop('response')
    return batch, response


def generate_negatives_in_minibatches_and_rebatch(dataset, batch_size, number_of_negatives, offer_features,
                                                  user_id_column, date_column, deterministic,
                                                  seed=None):
    # in each batch we will have number_of_negatives negatives for 1 positive
    # so need to take less positives
    batch_size = batch_size // (number_of_negatives + 1)
    # we will try to generate slightly more negatives due to collisions
    # and we need batch size to be divisible by number of negatives we want + 1 to do it in minibatch
    batch_size -= batch_size % (number_of_negatives + 1 + ADDITIONAL_NEGATIVES)
    # find maximal max_iter we need to run through the whole batch of initial dataset (which can be big)
    max_iter = 0
    for batch in dataset:
        max_iter = max(batch[user_id_column].shape[0] // batch_size + 1, max_iter)
    if not deterministic:
        dataset = dataset.shuffle(5)
    # using flat_map instead of interleave to keep memory limited
    # we filter all batches with smaller batch size than asked
    return dataset\
        .flat_map(partial(_generate_negatives_one_batch, batch_size=batch_size,
                          max_iter=max_iter, number_of_negatives=number_of_negatives,
                          offer_features=offer_features, user_id_column=user_id_column,
                          date_column=date_column, seed=seed))\
        .filter(lambda batch_: tf.shape(batch_[user_id_column])[0] == batch_size * (number_of_negatives + 1))\
        .map(partial(_add_weights, offer_features=offer_features))\
        .map(_pop_response)


class BroadcastLoss(tf.keras.losses.Loss):
    """
    Loss broadcasting ground truth to the output shape of the model
    needed in case when model output contains predictions for several augmentations
    >>> broadcast_loss = BroadcastLoss(tf.keras.metrics.mean_absolute_error)
    >>> broadcast_loss(tf.ones([3, 1]), tf.constant([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]))
    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
    """
    def __init__(self, loss_obj, **kwargs):
        self.loss_obj = loss_obj
        super().__init__(reduction=self.loss_obj.reduction, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        return self.loss_obj(y_true, y_pred)
    
    
class BroadcastMetric(tf.keras.metrics.Metric):
    """
    Same as BroadcastLoss, but for Metric: broadcasting ground truth to the output shape of the model
    needed in case when model output contains predictions for several augmentations
    """
    def __init__(self, metric_obj, **kwargs):
        self.metric_obj = metric_obj
        if 'name' not in kwargs:
            kwargs['name'] = self.metric_obj.name
        super().__init__(**kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        if sample_weight is not None:
            sample_weight = tf.broadcast_to(sample_weight, tf.shape(y_pred))
        return self.metric_obj.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return self.metric_obj.result()

    def reset_states(self):
        self.metric_obj.reset_states()
