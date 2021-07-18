import tensorflow as tf

import pandas as pd
import numpy as np
import os
import time
from params import *
from utils import create_model, preprocess

def haiku2text(haiku_list: list):
    text = ''
    
    for haiku in haiku_list:
        text += haiku + '\n'
    return text

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

if __name__ == '__main__':
    # 俳句DataFrameの取得
    try:
        df = pd.read_pickle(PKL_PATH)
    except:
        import pickle
        with open(PKL_PATH, 'rb') as f:
            df = pickle.load(f)
    
    # 俳句リストの取得
    haiku_list = df.haiku.tolist()
    
    # vocab, ids_from_chars, chars_from_idsの取得
    vocab, ids_from_chars, chars_from_ids = preprocess(haiku_list)
    
    # 俳句リストを一つの文字列として変数に格納する
    text = haiku2text(haiku_list)

    # 文字列をID列に変換
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    # tf datasetに変換
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    # バッチ単位で分割したsequenceを生成
    sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

    # 入力列とターゲット列を持つdatasetの生成
    dataset = sequences.map(split_input_target)
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    vocab_size = len(ids_from_chars.get_vocabulary())
    model = create_model(
        vocab_size=vocab_size,
        emb_dim=EMBEDDING_DIM,
        units=RNN_UNITS,
        adam_lr=LEARNING_RATE
    )

    # チェックポイントの作成
    steps_per_epoch = len(dataset)
    save_freq = steps_per_epoch * 10 # 10epoch毎にセーブする
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        save_freq=save_freq
    )

    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        # チェックポイントが既にある場合（学習済み）
        print('load model parameters from latest checkpoint')
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
