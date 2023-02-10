"""
For examples of usage of defined layers, see notebook
https://github.com/tinyclues/recsys-multi-atrribute-benchmark/blob/master/training/movielens%20bi-linear%20and%20augmentations.ipynb
"""
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp


def get_input_layer(name, dtype=tf.int32, row_splits_dtype=tf.int32):
    """
    Returns input layer for ragged input feature
    """
    return tf.keras.layers.Input(name=name,
                                 type_spec=tf.RaggedTensorSpec(shape=[None, None],
                                                               dtype=tf.int32,
                                                               row_splits_dtype=row_splits_dtype))


class WeightedEmbeddings(tf.keras.layers.Embedding):
    """
    Implementation of embeddings for ragged inputs, returning mean vector over ragged dimension.
    It can also return variance of vectors if initialized with calculate_variance=True.
    """
    
    def __init__(self, *args, **kwargs):
        self.calculate_variance = kwargs.pop('calculate_variance', False)
        super().__init__(*args, **kwargs)

    def call(self, input_tensor: tf.RaggedTensor, weights: Optional[tf.RaggedTensor] = None):
        """
        It is the same operation as
            tf.nn.safe_embedding_lookup_sparse(self.embeddings,
                                               sparse_ids=input_tensor.to_sparse(),
                                               sparse_weights=weights.to_sparse(),
                                               combiner='mean')
        but more memory efficient:
        we will first generate sparse matrix with indices of embeddings we want to combine
        and apply sparse_dense_matmul at the very end
        
        such approach also allows to easily calculate higher order moments, for example variance
        """
        nrows = input_tensor.nrows()
        row_indices = tf.repeat(tf.range(nrows, dtype=tf.int32), input_tensor.row_lengths())
        column_indices = input_tensor.values
        
        if weights is not None:
            # if weights are specified we suppose that weights are aligned with input_tensor
            # and each row of input_tensor contains unique labels
            pass
        else:
            projected_indices = row_indices * self.input_dim + column_indices
            unique_indices = tf.unique(projected_indices)[0]
            
            row_indices = unique_indices // self.input_dim
            column_indices = unique_indices % self.input_dim
            
            weights = tf.ones_like(row_indices, dtype=tf.keras.backend.floatx())
            weights = tf.RaggedTensor.from_value_rowids(weights, row_indices)
        
        weights /= tf.reduce_sum(weights, axis=1, keepdims=True)
        
        indices = tf.cast(tf.stack([row_indices, column_indices], axis=1), tf.int64)
        sparse_indicator = tf.sparse.SparseTensor(indices=indices,
                                                  values=tf.cast(weights.values, tf.float32),
                                                  dense_shape=(nrows, self.input_dim))
        if self.calculate_variance:
            first_second_moments = tf.sparse.sparse_dense_matmul(sparse_indicator,
                                                                 tf.concat([self.embeddings, self.embeddings ** 2], -1))
            mean, second = tf.split(first_second_moments, 2, -1)
            return mean, second - mean ** 2
        return tf.sparse.sparse_dense_matmul(sparse_indicator, self.embeddings)

    
def _split_into_smaller_groups(group_by_key: tf.Tensor) -> tf.Tensor:
    """
    Split the groups from group_by_key with large occurences into smaller sub_groups
    """
    _, idx_arr, count_per_group = tf.unique_with_counts(group_by_key)
    # taking a median value among group sizes, and consider larger groups
    # we will consider groups with at least 5 elements for edge-cases with a lot of small groups
    # to correct this addition in normal case we consider 40th percentile instead of exact median
    median = tf.cast(tfp.stats.percentile(tf.cast(count_per_group, tf.float32), 40), tf.int32) + 5
    
    nb_sub_groups_per_group = count_per_group // median + 1
    # allow to split each group into at most 50 smaller groups
    max_nb_sub_groups = min(tf.reduce_max(nb_sub_groups_per_group), 50)
    
    # we remap onto initial indices, so for each value in original group_by_key we know
    # into how many smaller groups it will be split
    sub_groups_range = tf.gather(nb_sub_groups_per_group, idx_arr)
    
    # in rand_sub_group tensor for each key value we got several indices between 0 and nb_sub_groups_per_group
    # showing into which smaller subgroup this value should go
    rand_sub_group = tf.random.uniform(shape=tf.shape(group_by_key),
                                       minval=0,
                                       maxval=max_nb_sub_groups,
                                       dtype=tf.int32) % sub_groups_range

    # reindexing local indices of subgroups into global ones
    return idx_arr * max_nb_sub_groups + rand_sub_group


def _collide_groups(group_by_key: tf.Tensor) -> tf.Tensor:
    """
    Pair groups and returns a new index where some of the groups will has the same index as another group.
    This operation is equivalent to redefinition of groups as "group=group1 OR group=group2", and
    we will use it to generate augmentations similar to offers having OR clause in their definition.
    """
    _, group_by_key = tf.unique(group_by_key)
    unique_groups_idx = tf.range(tf.reduce_max(group_by_key) + 1, dtype=tf.int32)
    number_of_groups_to_collide = 2 + tf.random.poisson(shape=(), lam=0.5, dtype=tf.int32)
    collided_groups_idx = unique_groups_idx // number_of_groups_to_collide
    # collided_groups_idx will contain `number_of_groups_to_collide` of each index
    # from 0 to a number of new groups: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    # by shuffling it we remap it to original groups -> groups with same index will be collided
    collided_groups_idx = tf.random.shuffle(collided_groups_idx)
    
    # going back to group_by_key indexing
    return tf.gather(collided_groups_idx, group_by_key)

    
class KeyGenerator(tf.keras.layers.Layer):
    """
    Layer to generate group by key from available offer attributes:
    we choose randomly some offer attributes and consider all values of those attributes
    it is equivalent to look at offer attr1='a' AND attr2='b' AND ...
    
    with probability 50% we also collide some groups to simulate OR-clause behaviour and to get offers of form
    (attr1='a' AND attr2='b' AND ...) OR (attr1='x' AND attr2='y' AND ...) OR ...
    """
    def __init__(self, number_of_offer_attributes, average_number_of_attributes_in_key, **kwargs):
        """
        :param number_of_offer_attributes: number of offer attributes in a model
        :param average_number_of_attributes_in_key: average number of offer attributes that will be used to generate a key
        """
        super().__init__(**kwargs)
        average_number_of_attributes_in_key = tf.cast(average_number_of_attributes_in_key, tf.float32)
        conservation_rate = average_number_of_attributes_in_key / tf.cast(number_of_offer_attributes, tf.float32)
        
        # we generate more blocks in advance in init to avoid generating non-interesting empty block
        n_to_generate = 30000
        proba_per_feature_per_block = tf.cast(tf.random.uniform((n_to_generate, number_of_offer_attributes)), tf.float32)
        blocks = tf.ragged.boolean_mask(tf.ragged.range(number_of_offer_attributes * tf.ones(n_to_generate, tf.int32)),
                                        proba_per_feature_per_block < conservation_rate)
        # by filtering empty blocks actually we violate demanded average_number_of_attributes_in_key
        # but it doesn't matter for the rest of the model, so we keep it like this for simplicity
        self.blocks = tf.ragged.boolean_mask(blocks, blocks.row_lengths() > 0)
        self._total_blocks = self.blocks.shape[0]
        
    def call(self, stacked_raw_attributes):
        """
        :param stacked_raw_attributes: tensor with raw values of offer attributes before embeddings
                                       shape (batch size, number of offer attributes)
        """
        random_index = tf.random.uniform((), 0, self._total_blocks, dtype=tf.int32)
        chosen_attributes = self.blocks[random_index]
        group_by_keys = tf.strings.reduce_join(
            tf.strings.as_string(tf.gather(stacked_raw_attributes, chosen_attributes, axis=1)),
            axis=1, separator=',')
        group_by_keys = tf.strings.to_hash_bucket_fast(group_by_keys, tf.int32.max)
        if tf.random.uniform(()) < 0.5:
            # with 50% probability we will also simulate OR-clause behaviour
            # by splitting big groups into smaller ones
            # and randomly colliding smaller groups
            return _collide_groups(_split_into_smaller_groups(group_by_keys))
        # otherwise just return obtained keys remapped into values between 0 and number of groups
        return tf.unique(group_by_keys)[1]
    

class GroupBy(tf.keras.layers.Layer):
    """
    Layer performing group-by like operation on embeddings.
    Taking group_by_key for each offer feature we consider vectors corresponding to a given group_by_key value
    and return mean and variance of those vectors.
    """
    def call(self, group_by_key, stacked_embeddings):
        """
        :param group_by_key: tensor containg key values to perform group by on
                             shape (batch size,)
        :param stacked_embeddings: embeddings of all offer features stacked together
                                   shape (batch size, number of features, embedding dimension)
        """
        unique_values, unique_idx = tf.unique(group_by_key)
        embeddings_grouped = tf.ragged.stack_dynamic_partitions(stacked_embeddings, unique_idx, tf.shape(unique_values)[0])
        mean, var = tf.nn.moments(embeddings_grouped, axes=1)
        return tf.gather(mean, unique_idx), tf.gather(var, unique_idx)


class UserFeaturesCompressor(tf.keras.layers.Layer):
    """
    Layer to combine some user features together and obtain new meta features
    """
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
        """
        :param number_of_meta_features: number of meta features to obtain in the end, so output shape will be
                                        (batch size, number_of_meta_features, embedding dimension)
        :param dropout_rate: droupout for intermediate layers
        """
        super().__init__(**kwargs)
        self.number_of_meta_features = number_of_meta_features
        self.dropout_rate = dropout_rate
        
    def build(self, stacked_embeddings_shape):
        embedding_dim = stacked_embeddings_shape[-1]
        self.compressor = tf.keras.Sequential([
            tf.keras.layers.experimental.EinsumDense('bur,ukrd->bkd',
                                                     output_shape=(self.number_of_meta_features * 2, embedding_dim),
                                                     activation='gelu', bias_axes='kd'),
            tf.keras.layers.Dropout(rate=self.dropout_rate),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.experimental.EinsumDense('bur,uk->bkr',
                                                     output_shape=(self.number_of_meta_features, embedding_dim),
                                                     activation='selu', bias_axes='k'),
            tf.keras.layers.BatchNormalization(),
        ])
        
    def call(self, stacked_embeddings):
        """
        :param stacked_embeddings: embeddings of all user features stacked together
                                   shape (batch size, number of features, embedding dimension)
        """
        return self.compressor(stacked_embeddings)

    
class kWTA(tf.keras.layers.Layer):
    """
    k-WTA activation layer following https://arxiv.org/pdf/1905.10510.pdf
    """
    def __init__(self, k: int, **kwargs):
        self.k = k
    
    def call(self, input_tensor):
        # shapes:
        # input_tensor (*, n)
        # k_winners_idx (*, k)
        k_winners_idx = tf.math.top_k(tf.math.abs(input_tensor), k=self.k).indices
        # one_hot_k_winners (*, k, n)
        # last_dim_multi_hot (*, n)
        one_hot_k_winners = tf.one_hot(k_winners_idx, tf.shape(input_tensor)[-1], dtype=input_tensor.dtype)
        last_dim_multi_hot = tf.reduce_max(one_hot_k_winners, axis=-2)
        return input_tensor * last_dim_multi_hot
    
    
class OfferFeaturesCompressor(tf.keras.layers.Layer):
    """
    Layer to combine some offer features together in a instance-wise manner and to obtain new meta features
    """
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
        """
        :param number_of_meta_features: number of meta features to obtain in the end, so output shape will be
                                        (batch size, number_of_meta_features, embedding dimension)
        :param dropout_rate: droupout for intermediate layers
        """
        super().__init__(**kwargs)
        self.number_of_meta_features = number_of_meta_features
        self.dropout_rate = dropout_rate
        
    def build(self, inputs_shape):
        embeddings_shape = inputs_shape[0]
        n_features = embeddings_shape[-2]
        self.compressor_kernel = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(n_features * self.number_of_meta_features, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            kWTA((n_features * self.number_of_meta_features) // 3),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(n_features * self.number_of_meta_features, activation='tanh'),
            tf.keras.layers.Reshape((n_features, self.number_of_meta_features))
        ])
    
    def call(self, inputs):
        """
        :param inputs: tuple with mean and variance of embeddings
                       both of shape (batch size, number of features, embedding dimension)
        """
        mean_embeddings, var_embeddings = inputs
        compressor_kernel = self.compressor_kernel(tf.concat([mean_embeddings, var_embeddings], axis=-1))
        return tf.einsum('...od,...om->...md', mean_embeddings, compressor_kernel)


class MaskNet(tf.keras.layers.Layer):
    """
    Layer to reduce relative importance of noisy coordinates in offer embeddings.
    More precisely we will learn instance-wise mask depending on mean and variance of embeddings
    """
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
        """
        :param number_of_meta_features: number of meta features from OfferFeaturesCompressor, so output shape will be
                                        (batch size, number_of_meta_features, embedding dimension)
        :param dropout_rate: droupout for intermediate layers
        """
        super().__init__(**kwargs)
        self.number_of_meta_features = number_of_meta_features
        self.dropout_rate = dropout_rate
        
    def build(self, inputs_shape):
        embeddings_shape = inputs_shape[0]
        embeddings_dim = embeddings_shape[-1]
        self.mask_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(embeddings_dim, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            kWTA(embeddings_dim // 3),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.number_of_meta_features * embeddings_dim, activation='tanh'),
            tf.keras.layers.Reshape((self.number_of_meta_features, embeddings_dim))])
    
    def call(self, inputs):
        """
        :param inputs: tuple with mean and variance of embeddings
                       both of shape (batch size, number of features, embedding dimension)
        """
        mean_embeddings, var_embeddings = inputs
        return self.mask_net(tf.concat([mean_embeddings, var_embeddings], axis=-1))
    

class BiLinearInteraction(tf.keras.layers.Layer):
    """
    Layer for the feature-wise interaction between user and offer (meta) features.
    Given user and offer inputs of shapes (bs, u, r),
                                          (bs, o, d)
    we will learn interaction kernels of dimension (r, d) for each pair of user/offer features (u, o)
    we output resulting interactions for each pair, resulting shape (bs, u * o)
    """
    def __init__(self, activation=None, initializer='uniform', regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, inputs_shape):
        user_shape, item_shape = inputs_shape
        
        kernel_shape = (user_shape[-2], item_shape[-2], user_shape[-1], item_shape[-1])
        bias_shape = (1, user_shape[-2] * item_shape[-2])
        self.out_shape = (-1, user_shape[-2] * item_shape[-2])
        
        self.interaction_kernel = self.add_weight(
            'interaction_kernel',
            shape=kernel_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            dtype=tf.float32,
            trainable=True)
        self.bias = self.add_weight(
            'interaction_bias',
            shape=bias_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            dtype=tf.float32,
            trainable=True)
        
        
    def call(self, inputs):
        user_embeddings, item_embeddings = inputs
        res = tf.einsum('...ud,...or,uodr->...uo', user_embeddings, item_embeddings, self.interaction_kernel)
        res = tf.reshape(res, self.out_shape)
        return self.activation(res + self.bias)
