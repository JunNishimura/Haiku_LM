import tensorflow as tf

import pandas as pd
from utils import create_model, preprocess, text_from_ids
from models import OneStep
from filters import HaikuFilter
from params import *

if __name__=='__main__':
    df = pd.read_pickle(HAIKU_PKL_PATH)

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
    checkpoint_dir = './training_checkpoints'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest) # load parameters

    # 事前準備
    haiku_filter = HaikuFilter()
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = [None, None, None]
    next_char = tf.constant(['\n'])
    generated_haiku_list = [] # 生成した俳句の一覧を格納するためのリスト

    # 指定回数まで俳句の生成を続ける
    while len(generated_haiku_list) < HAIKU_NUM:
        # 生成俳句を格納するためのリスト
        haiku = []

        # 改行が出るまで言語モデルより文字を生成する
        while True:
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            decoded_char = next_char[0].numpy().decode('utf-8')

            # 改行文字が出たら俳句完成と見なして、フィルターに掛ける
            if decoded_char == '\n':
                break
            haiku.append(next_char)

        # リストから文字列に変換して、utf-8へデコードする
        haiku = tf.strings.join(haiku)
        haiku = haiku[0].numpy().decode('utf-8')

        # # 17音であるかのチェック
        # if not word_count_check(haiku):
        #     continue
        
        # # 季語が含まれているかのチェック
        # if not kigo_check(haiku):
        #     continue

        # # 切れ字が含まれているかのチェック
        # if not kireji_check(haiku):
        #     continue

        # generated_haiku_listへ追加
        generated_haiku_list.append(haiku)

    # 生成俳句を出力
    for gh in generated_haiku_list:
        print(gh)