import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertJapaneseTokenizer, TFBertModel, pipeline


def main():
    try:
        texts = [sys.argv[1], sys.argv[2]]
    except:
        print('単語を入力してください')
        return
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        do_subword_tokenize=False)
    model = TFBertModel.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    job = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    results = job(texts)
    X = np.array(results[0][1:-1])
    Y = np.array(results[1][1:-1])
    if len(X) > 1 or len(Y) > 1:
        print('入力は単語にしてください')
        return
    print(f'単語間のコサイン類似度: {cosine_similarity(X, Y)[0][0]}')


if __name__ == '__main__':
    main()
