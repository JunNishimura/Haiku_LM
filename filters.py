import MeCab
import pickle

class HaikuFilter():
    def __init__(self):
        self.tagger = MeCab.Tagger('-Owakati')
        self.kigo_dict = None
        
    def word_count(self, haiku) -> bool:
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

    def kigo(self, haiku: str) -> bool:
        '''
        季語が含まれているかのチェック

        Parameters
        ----------
            haiku: str
                チェック対象となる俳句
        Return
        ------
            true if haiku has kigo
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
                        return True

        return False

    def kireji(self, haiku: str) -> bool:
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
        kireji_list = ['や', 'かな', '哉', 'けり']
        n = self.tagger.parseToNode(haiku)
        while n:
            features = n.feature.split(',')
            if features[0] != u'BOS/EOS':
                yomi = n.feature.split(',')[6]
            
            n = n.next