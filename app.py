from flask import Flask, jsonify, request
import numpy as np
import sys
from gensim.models.word2vec import Word2Vec

# import MeCab
# tagger = MeCab.Tagger("-Owakati -u /usr/local/lib/mecab/dic/MANBYO_201907_Dic-utf8.dic")


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  

wv = Word2Vec.load("./w2v_model/wiki.vec.pt")

def cos_sim(word1, word2):
    v1 = wv[word1]
    v2 = wv[word2]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


@app.route('/')  # URLルーティング設定 '/' の場合 コンテキストルートになります。
def index():
    return "Index Page"


@app.route('/hello')
def hello_world():
    return "hello world"


@app.route('/word_similarity', methods=['POST'])
def get_horse_name():
    try:
        word1 = request.json['word1']   
        word2 = request.json['word2']   

        return jsonify({"status": "OK", "score": f"{abs(cos_sim(word1, word2)): .3f}"})
    except:
        raise(Exception)

@app.errorhandler(Exception)
def error_except(e):
    return jsonify({'status': 'error', 'message': "辞書にない単語です"}), 500


if __name__ == '__main__':
    app.run()  # Httpサーバー起動
