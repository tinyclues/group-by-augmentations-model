import re

from layers import *
from utils import *

import tensorflow as tf
import tensorflow_addons as tfa


MOVIELENS_EPOCHS = 12
REES_EPOCHS = 8

NB_AUGMENTATIONS = 3
AVERAGE_NUMBER_OF_FEATURES_IN_AUGMENTATION = 2
USER_META_FEATURES = 5
OFFER_META_FEATURES = 3


def REGULARIZER(l1_coeff):
    return {'class_name': 'L1L2', 'config': {'l1': l1_coeff, 'l2': 0.}}


def fully_connected_layers(dim1, dim2, activation, l1_coeff, dropout, name, output=False):
    layers = [
        tf.keras.layers.Reshape((-1,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(dim1, kernel_regularizer=REGULARIZER(l1_coeff), bias_regularizer=REGULARIZER(l1_coeff)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Dense(dim2, kernel_regularizer=REGULARIZER(l1_coeff), bias_regularizer=REGULARIZER(l1_coeff)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Activation(activation),
    ]
    if output:
        layers.append(
            tf.keras.layers.Dense(1, kernel_regularizer=REGULARIZER(l1_coeff), bias_regularizer=REGULARIZER(l1_coeff))
        )
    return tf.keras.Sequential(layers, name=name)
    
    
def _get_vocabulary_sizes(inverse_lookups, user_features, offer_features):
    vocabulary_sizes = {}

    for feature in offer_features:
        vocabulary_sizes[feature] = inverse_lookups[feature].vocabulary_size()

    for feature in user_features:
        for key in inverse_lookups:
            pattern = re.compile(r"{}(\w+)_{}".format(AGG_PREFIX, key))
            if pattern.match(feature):
                vocabulary_sizes[feature] = vocabulary_sizes[key]
    
    return vocabulary_sizes
    

def _model(user_features, offer_features, inverse_lookups, params, name, group_by=True, mask_net=True, bi_linear_interaction=True):
    vocabulary_sizes = _get_vocabulary_sizes(inverse_lookups, user_features, offer_features)
    
    inputs = {}
    embedded_user_features, embedded_offer_features, variance_offer_features = {}, {}, {}
    for feature in user_features:
        inputs[feature] = get_input_layer(feature)
        emb_layer = WeightedEmbeddings(vocabulary_sizes[feature],
                                       params['embedding_dim'], name=f'{feature}_embedding',
                                       embeddings_regularizer=REGULARIZER(params['l1_coeff']))
        embedded_user_features[feature] = emb_layer(inputs[feature])
    for feature in offer_features:
        # for offer features we need weights:
        # with dummy weights during training, and the ones used for a feature's averaging at inference time
        inputs[f'{feature}_weight'] = get_input_layer(f'{feature}_weight', tf.float32)
        inputs[feature] = get_input_layer(feature)
        emb_layer = WeightedEmbeddings(vocabulary_sizes[feature],
                                       params['embedding_dim'], name=f'{feature}_embedding',
                                       embeddings_regularizer=REGULARIZER(params['l1_coeff']),
                                       calculate_variance=True)
        embedded_offer_features[feature], variance_offer_features[feature] =\
            emb_layer(inputs[feature], inputs[f'{feature}_weight'])
    
    user_stacked = tf.stack(list(embedded_user_features.values()), axis=1)
    offer_stacked = tf.stack(list(embedded_offer_features.values()), axis=1)
    offer_variance = tf.stack(list(variance_offer_features.values()), axis=1)
    stacked_raw_offer_attrs = tf.stack([tf.cast(inp.values, tf.int32) for feature, inp in inputs.items()
                                        if feature in offer_features], axis=1)
    
    if group_by:
        group_by_layer = GroupBy(name='group_by')
        key_generator = KeyGenerator(number_of_offer_attributes=len(offer_features),
                                     average_number_of_attributes_in_key=AVERAGE_NUMBER_OF_FEATURES_IN_AUGMENTATION,
                                     name='grp_key_generator')
        
        augmentations = []
        for i in range(NB_AUGMENTATIONS):
            group_by_key = key_generator(stacked_raw_offer_attrs)
            augmentations.append(group_by_layer(group_by_key, offer_stacked))
        
    else:
        augmentations = [(offer_stacked, None)]
        
    if mask_net:
        user_compressed = UserFeaturesCompressor(USER_META_FEATURES, params['dropout'],
                                                 name='user_compressor')(user_stacked)
        
        offer_features_compressor = OfferFeaturesCompressor(OFFER_META_FEATURES, params['dropout'],
                                                            name='offer_compressor')
        mask_net = MaskNet(OFFER_META_FEATURES, params['dropout'],
                           name='mask_generation')
        apply_mask = tf.keras.layers.Multiply(name='apply_mask')
        
        attention_augmentations = []
        for mean_offer_emb, variance_offer_emb in augmentations:
            compressed_offer_embeddings = offer_features_compressor([mean_offer_emb, variance_offer_emb])
            mask = mask_net([mean_offer_emb, variance_offer_emb])
            attention_augmentations.append(apply_mask([compressed_offer_embeddings, mask]))
        
        compressed_offer_embeddings = offer_features_compressor([offer_stacked, offer_variance])
        mask = mask_net([offer_stacked, offer_variance])
        eval_offer_embeddings = apply_mask([compressed_offer_embeddings, mask])
    else:
        user_compressed = user_stacked
        attention_augmentations = [mean_offer_emb for mean_offer_emb, _ in augmentations]
        eval_offer_embeddings = offer_stacked
        
    if bi_linear_interaction:
        if not mask_net:
            # we need to apply compression to keep model's footprint limited
            # and also to keep model robust with same hyperparams
            user_compressed = UserFeaturesCompressor(USER_META_FEATURES, params['dropout'],
                                                     name='user_compressor')(user_compressed)
            
        bi_linear_interaction_layer = BiLinearInteraction(number_of_negatives=params['number_of_negatives'],
                                                    dropout_rate=params['dropout'],
                                                    initializer='random_normal',
                                                    regularizer=REGULARIZER(params['l1_coeff']),
                                                    name='interaction')
        output_dnn = fully_connected_layers(*params['output_dnn_args'])
        
        augmentation_predictions = []
        for masked_offer_embeddings in attention_augmentations:
            augmentation_predictions.append(
                output_dnn(bi_linear_interaction_layer([user_compressed, masked_offer_embeddings],
                                                       generate_negatives=True))
            )
        output = tf.concat(augmentation_predictions, axis=1)
        
        eval_output = output_dnn(bi_linear_interaction_layer([user_compressed, eval_offer_embeddings],
                                                             generate_negatives=True))
    else:
        user_tower = fully_connected_layers(*params['user_tower_args'])(user_compressed)
        offer_tower_layer = fully_connected_layers(*params['offer_tower_args'])
        dot_interaction = DotWithNegatives(params['number_of_negatives'], name='prediction')
        
        augmentation_predictions = []
        for masked_offer_embeddings in attention_augmentations:
            offer_tower = offer_tower_layer(masked_offer_embeddings)
            augmentation_predictions.append(dot_interaction([user_tower, offer_tower], generate_negatives=True))
        output = tf.concat(augmentation_predictions, axis=1)
        
        eval_offer_embeddings = offer_tower_layer(eval_offer_embeddings)
        eval_output = dot_interaction([user_tower, eval_offer_embeddings], generate_negatives=True)
    
    LOSS = BroadcastLoss(tf.keras.losses.BinaryCrossentropy(from_logits=True), params['number_of_negatives'])
    AUC_METRIC = BroadcastMetric(tf.keras.metrics.AUC(from_logits=True), params['number_of_negatives'])
    
    model = tf.keras.Model(inputs, output, name=name)
    model.compile(optimizer=params['optimizer'], loss=LOSS, metrics=[AUC_METRIC])

    eval_model = tf.keras.Model(inputs, eval_output, name=f'{name}_eval')
    emb_model = tf.keras.Model(inputs, eval_offer_embeddings, name=f'{name}_emb')
    
    return model, eval_model, emb_model


def movielens_model(user_features, offer_features, inverse_lookups, number_of_negatives, name, group_by=True, mask_net=True, bi_linear_interaction=True):
    params = {
        'number_of_negatives': number_of_negatives,
        'embedding_dim': 100,
        'l1_coeff': 8.5e-7,
        'dropout': 0.17,
        'optimizer': tfa.optimizers.AdamW(weight_decay=8.5e-8, learning_rate=0.0008)
    }
    
    params['user_tower_args'] = [80, 40, 'tanh', params['l1_coeff'], params['dropout'], 'user_tower']
    params['offer_tower_args'] = [80, 40, 'tanh', params['l1_coeff'], params['dropout'], 'offer_tower']
    params['output_dnn_args'] = [80, 40, 'gelu', params['l1_coeff'], params['dropout'], 'output_dnn', True]
    
    return _model(user_features, offer_features, inverse_lookups, params, name, group_by, mask_net, bi_linear_interaction)


def rees_model(user_features, offer_features, inverse_lookups, number_of_negatives, name, group_by=True, mask_net=True, bi_linear_interaction=True):
    params = {
        'number_of_negatives': number_of_negatives,
        'embedding_dim': 100,
        'l1_coeff': 2e-7,
        'dropout': 0.1,
        'optimizer': tfa.optimizers.AdamW(weight_decay=4e-8, learning_rate=0.0008)
    }
    
    params['user_tower_args'] = [100, 50, 'tanh', params['l1_coeff'], params['dropout'], 'user_tower']
    params['offer_tower_args'] = [100, 50, 'tanh', params['l1_coeff'], params['dropout'], 'offer_tower']
    params['output_dnn_args'] = [100, 50, 'tanh', params['l1_coeff'], params['dropout'], 'output_dnn', True]
    
    return _model(user_features, offer_features, inverse_lookups, params, name, group_by, mask_net, bi_linear_interaction)
