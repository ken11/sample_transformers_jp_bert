import sys

from transformers import BertJapaneseTokenizer, TFBertForMaskedLM, pipeline


def main():
    try:
        text = sys.argv[1]
    except:
        print('テキストを入力してください')
        return
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    model = TFBertForMaskedLM.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    job = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    for result in job(text):
        print(f'候補: {result["token_str"]} score:{result["score"]}')


if __name__ == '__main__':
    main()
