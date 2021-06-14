# transformersで日本語BERTを使ってみるサンプル
transformersやBERTの紹介用  
[Hugging Faceのtransformers](https://huggingface.co/transformers)を利用してBERTのモデルでいろいろな日本語のタスクを試すサンプル  
※サンプルコードではモデルは[東北大の日本語BERTモデル](https://github.com/cl-tohoku/bert-japanese)を使用しています

## 使い方
サンプルコードは三種のタスクを用意しています  
- fill-mask: [MASK]の内容を埋めるものです
- sentiment-analysis: テキストの感情を推測するものです(標準で使用している東北大のモデルは汎用モデルなので、このタスクの結果精度を求めるにはモデルの学習が必要です)
- word-similarity: 単語間の類似度を測定するものです(単語ベクトルのコサイン類似度を測っています)
  
```sh
$ docker-compose build

# fill-mask
$ docker-compose run --rm app python3.8 ./samples/fill-mask.py 出雲の[MASK]

# 以下のような結果が出力されます
# 候補: 国 score:0.023697344586253166
# 候補: 神 社 score:0.022133585065603256
# 候補: 道 score:0.017958277836441994
# 候補: 星 score:0.01459976751357317
# 候補: 神 話 score:0.014350272715091705

# sentiment-analysis
$ docker-compose run --rm app python3.8 ./samples/sentiment-analysis.py 好きになってよかった

# word-similarity
$ docker-compose run --rm app python3.8 ./samples/word-similarity.py 明日 未来
```
