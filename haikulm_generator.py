import tensorflow as tf

import pandas as pd
from utils import create_model, preprocess, text_from_ids
from models import OneStep
from params import *

if __name__=='__main__':
    df = pd.read_pickle(PKL_PATH)

    # 俳句リストの取得
    haiku_list = df.haiku.tolist()

    # vocab, ids_from_chars, chars_from_idsの取得
    vocab, ids_from_chars, chars_from_ids = preprocess(haiku_list)

    # モデルの作成
    vocab_size = len(ids_from_chars.get_vocabulary())
    model = create_model(
        vocab_size=vocab_size,
        emb_dim=EMBEDDING_DIM,
        units=RNN_UNITS,
        adam_lr=LEARNING_RATE
    )

    # チェックポイントから学習済みパラメータをモデルにロードする
    checkpoint_dir = './training_checkpoints'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    states = [None, None, None]
    next_char = tf.constant(['\n'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
