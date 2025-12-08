import re

class SimpleTokenizer:
    def __init__(self, vocab):
        # Ensure unknown token exists
        if "<|unk|>" not in vocab:
            vocab["<|unk|>"] = len(vocab)

        self.token_to_id = vocab
        self.id_to_token = {id: token for token, id in vocab.items()}

    def encode(self, text):
        # Tokenizing text based on punctuation and whitespace
        tokens = re.split(r'([\'\".,?_/()*!]|--| -|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        
        #Accounting for unknown tokens (words)
        # Replace Out-Of-Vocabulary (OOV) tokens with <|unk|>
        tokens = [
            token if token in self.token_to_id else "<|unk|>"
            for token in tokens
        ]

        # Always safe—<|unk|> now exists
        ids = [self.token_to_id[token] for token in tokens]
        return ids

    def decode(self, ids):
        # Convert IDs back to tokens
        tokens = [self.id_to_token[id] for id in ids]

        text = " ".join(tokens)

        # Fix spacing around punctuation
        text = re.sub(r'([\'\".,?_/()*!])', r' \1', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        return text
