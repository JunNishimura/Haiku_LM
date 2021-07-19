import MeCab
import pickle

class HaikuFilter():
    def __init__(self):
        self.tagger = MeCab.Tagger('-Owakati')
        self.kigo_dict = None
        
    def check_wordcount(self, haiku) -> bool:
        '''
        17音であるかのチェック

        Parameters
        ----------
            haiku: str
                チェック対象となる俳句
        Return
        ------
            true if haiku is composed of 17 sounds
        '''
        yomi_haiku = ''
        n = self.tagger.parseToNode(haiku)
        while n:
            features = n.feature.split(',')
            if features[0] != u'BOS/EOS':
                yomi = n.feature.split(',')[6]
                yomi_haiku += yomi
            n = n.next
        
        return len(yomi_haiku) == 17

    def pos_extract(self, haiku, pos) -> list:
        '''
        指定された品詞の単語のみを取り出す

        Parameters
        ----------
            haiku: str
                俳句
            pos: str
                品詞

        Return
        ------
            list of words which passed the pos filter
        '''
        words = []
        n = self.tagger.parseToNode(haiku)
        while n:
            features = n.feature.split(',')
            if features[0] == pos:
                words.append(n.surface)
            n = n.next

        return words

    def check_kigo(self, haiku: str):
        '''
        季語が含まれているかをチェックし、含まれている場合は季語とその季節を返す

        Parameters
        ----------
            haiku: str
                チェック対象となる俳句
        Return
        ------
            kigo: str
                季語、ない場合はNone
            season: str
                季節、ない場合はNone
        '''
        # 季語辞書がまだない場合
        if not self.kigo_dict:
            with open('../pickles/kigo_dict.pkl', 'rb') as f:
                self.kigo_dict = pickle.load(f)
        
        # 俳句中に含まれる名詞を抽出
        nouns = self.pos_extract(haiku, '名詞')

        # 名詞が存在する場合
        if len(nouns):
            for noun in nouns:
                for k, v in self.kigo_dict.items():
                    # もし季語が見つかればTrueを返す
                    if noun in v:
                        return v, k

        return None, None

    def check_kireji(self, haiku: str) -> bool:
        '''
        切れ字が含まれているかのチェック

        Parameters
        ----------
            haiku: str
                チェック対象となる俳句
        Return
        ------
            true if haiku has kireji
        '''
        kireji_list = ['かな', '哉', 'もがな', 'し', 'じ', 'や', 'らん', 'か', 'けり', 'よ', 'ぞ', 'つ', 'せ', 'ず', 'れ', 'ぬ', 'へ', 'け', 'いかに']
        kireji_flag = False
        
        for w in self.tagger.parse(haiku).split():
            if w in kireji_list:
                kireji_flag = True
        
        # 切れ字がなければその時点でアウト
        if not kireji_flag:
            return False

        # 切れ字があ場合、適切な位置に切れ字があるかをチェック
        pos = 0
        n = self.tagger.parseToNode(haiku)
        while n:
            features = n.feature.split(',')
            if features[0] != u'BOS/EOS':
                yomi = features[6]
                pos += len(yomi)

                # 5-7-5の終わりに切れ字がある場合はok
                if pos == 5 or pos == 12 or pos == 17:
                    if n.surface in kireji_list:
                        return True
            n = n.next
        return False

    def check_association(self, haiku: str, associative_words: list) -> bool:
        '''
        句の連続性を確認

        Parameters
        ----------
            haiku: str
                俳句
            associative_words: list
                連想単語の一覧
        Return
        ------
            true if haiku has an associative word
        '''

        n = self.tagger.parseToNode(haiku)
        while n:
            features = n.feature.split(',')
            if features[0] != u'BOS/EOS':
                # もし連想単語リストに含まれている単語が俳句に含まれていたらTrueを返す
                if n.surface in associative_words:
                    return True
            n = n.next
        return False