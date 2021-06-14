import sys

from transformers import BertJapaneseTokenizer, TFBertForSequenceClassification, pipeline


def main():
    try:
        text = sys.argv[1]
    except:
        print('テキストを入力してください')
        return
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    model = TFBertForSequenceClassification.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    job = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    for result in job(text):
        print(f'label: {result["label"]} score:{result["score"]}')


if __name__ == '__main__':
    main()
