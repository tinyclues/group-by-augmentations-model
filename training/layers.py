from typing import Optional

import tensorflow as tf


def get_input_layer(name, dtype=tf.int32, row_splits_dtype=tf.int32):
    """
    Returns 
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

    
class KeyGenerator(tf.keras.layers.Layer):
    def __init__(self, number_of_offer_attributes, average_number_of_attributes_in_key, **kwargs):
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
        random_index = tf.random.uniform((), 0, self._total_blocks, dtype=tf.int32)
        chosen_attributes = self.blocks[random_index]
        group_by_keys = tf.strings.reduce_join(
            tf.strings.as_string(tf.gather(stacked_raw_attributes, chosen_attributes, axis=1)),
            axis=1, separator=',')
        # TODO add OR collisions here
        return tf.strings.to_hash_bucket_fast(group_by_keys, tf.int32.max)
    

class GroupBy(tf.keras.layers.Layer):    
    def call(self, group_by_key, stacked_embeddings):
        unique_values, unique_idx = tf.unique(group_by_key)
        embeddings_grouped = tf.ragged.stack_dynamic_partitions(stacked_embeddings, unique_idx, tf.shape(unique_values)[0])
        mean, var = tf.nn.moments(embeddings_grouped, axes=1)
        return tf.gather(mean, unique_idx), tf.gather(var, unique_idx)


class UserFeaturesCompressor(tf.keras.layers.Layer):
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
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
        return self.compressor(stacked_embeddings)

    
class OfferFeaturesCompressor(tf.keras.layers.Layer):
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
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
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(n_features * self.number_of_meta_features, activation='tanh'),
            tf.keras.layers.Reshape((n_features, self.number_of_meta_features))
        ])
    
    def call(self, inputs):
        mean_embeddings, var_embeddings = inputs
        compressor_kernel = self.compressor_kernel(tf.concat([mean_embeddings, var_embeddings], axis=-1))
        return tf.einsum('...fd,...fm->...md', mean_embeddings, compressor_kernel)


class MaskNet(tf.keras.layers.Layer):
    def __init__(self, number_of_meta_features: int, dropout_rate: float, **kwargs):
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
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.number_of_meta_features * embeddings_dim, activation='tanh'),
            tf.keras.layers.Reshape((self.number_of_meta_features, embeddings_dim))])
    
    def call(self, inputs):
        mean_embeddings, var_embeddings = inputs
        return self.mask_net(tf.concat([mean_embeddings, var_embeddings], axis=-1))
    

class BiLinearInteraction(tf.keras.layers.Layer):
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
