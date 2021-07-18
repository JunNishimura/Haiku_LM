import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam

import itertools
from models import HaikuModel

def preprocess(haiku_list: list):
    '''
    preprocess for haiku list
    
    Parameters
    ----------
        haiku_list: list
            list of haiku
    
    Return
    ------
        vocab: set of vocabularies
        ids_from_chars: translate chars to ids
        chars_from_ids: translate ids to chars
    '''
    vocab = sorted(set(list(itertools.chain.from_iterable(haiku_list))+['\n']))

    ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab),
        mask_token=None
    )
    chars_from_ids = preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )

    return vocab, ids_from_chars, chars_from_ids

def text_from_ids(ids: list, chars_from_ids):
    '''
    translate ids to text

    Parameters
    ----------
        ids: list
            list of character id
        chars_from_ids:
            translate ids to chars
            
    Return
    ------
        text: text translated from ids
    '''
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def create_model(vocab_size: int, emb_dim: int, units: int, adam_lr: float):
    '''
    create a haiku model with Adam for optimizer and SparseCategoricalCrossEntropy for loss function.

    Parameters
    ----------
        vocab_size: int
            size of vocabulary
        emb_dim: int
            dimension of embedding
        units: int
            number of rnn units
        adam_lr: float
            learning rate for adam

    Return
    ------
        model: haiku model
    '''
    model = HaikuModel(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        rnn_units=units
    )

    adam = Adam(learning_rate=adam_lr)
    model.compile(
        optimizer=adam,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    return model
