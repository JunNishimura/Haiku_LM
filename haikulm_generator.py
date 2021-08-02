import tensorflow as tf

import pandas as pd
import fasttext
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
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    model.load_weights(latest) # load parameters
    
    haiku_filter = HaikuFilter()
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = [None, None, None]
    next_char = tf.constant(['\n'])
    generated_haiku_list = [] # 生成した俳句の一覧を格納するためのリスト
    ft = fasttext.load_model(FASTTEXT_MODEL_PATH)
    associative_words = None # 連想文字の一覧

    # 指定回数まで俳句の生成を続ける
    while len(generated_haiku_list) < HAIKU_NUM:
        # 生成俳句を格納するためのリスト
        haiku = []

        # 改行が出るまで言語モデルより文字を生成する
        while True:
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            decoded_char = next_char[0].numpy().decode('utf-8')

            # 改行文字が出たら俳句完成と見なす（ただし空文字でない事）
            if decoded_char == '\n' and len(haiku):
                break
            haiku.append(next_char)

        # リストから文字列に変換して、utf-8へデコードする
        haiku = tf.strings.join(haiku)
        haiku = haiku[0].numpy().decode('utf-8')

        if IS_FILTER:
            
            # 季語フィルターを適用
            kigo, season = haiku_filter.check_kigo(haiku)
            if not kigo: # 季語がない場合はスキップ
                continue

            # # 文字数フィルター
            if not haiku_filter.check_wordcount(haiku, margin=2):
                continue

            # #-------------------- 発句 --------------------#
            # # 発句は切れ字を含んでいる必要がある
            # if len(generated_haiku_list) == 0:
            #     if not haiku_filter.check_kireji(haiku):
            #         continue

            # #-------------------- 脇句 --------------------#
            # # 脇句は発句と同じ季節の句である必要がある
            # if len(generated_haiku_list) == 1:
            #     prev_kigo, prev_season = haiku_filter.check_kigo(generated_haiku_list[0])
            #     if season != prev_season:
            #         continue

            # #-------------------- 二句目以降 --------------------#
            # # 二句目以降は前句と連続している必要がある
            # if len(generated_haiku_list) >= 1:
            #     if not haiku_filter.check_association(haiku, associative_words):
            #         continue

        # generated_haiku_listへ追加
        generated_haiku_list.append(haiku)

        # 季語から連想される単語の一覧を格納
        # 連想単語はfasttextを用いて、季語との近傍単語を50個選択
        # associative_words = ft.get_nearest_neighbors(kigo)
        # associative_words = [w[1] for w in associative_words]

    # 生成俳句を出力
    for gh in generated_haiku_list:
        print(gh)
