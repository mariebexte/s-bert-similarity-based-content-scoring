from somajo import SoMaJo

TOKENIZER = SoMaJo('en_PTB', split_sentences=False)


def tokenize(sentence):
    tokens = next(TOKENIZER.tokenize_text([sentence]))
    return [token.text for token in tokens]
