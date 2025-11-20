import re

class SimpleTokenizer:
    def __init__(self, vocab):
        # Mapping from string to ID (str_to_int) and from ID to string (int_to_str)
        self.token_to_id = vocab
        self.id_to_token = {id: token for token, id in vocab.items()}
    
    def encode(self, text):
        # Tokenizing text based on punctuation and whitespace
        tokens = re.split(r'([\'\".,?_/()*!]|--| -|\s)', text)
        # Stripping whitespace and filtering out empty tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        # Converting tokens to IDs
        try:
            ids = [self.token_to_id[token] for token in tokens]
        except KeyError as e:
            raise ValueError(f"Token '{e.args[0]}' not found in vocabulary.") 
        return ids

    def decode(self, ids):
        # Decoding list of IDs to text
        try:
            text = " ".join([ self.id_to_token[id] for id in ids])
        except KeyError as e:
            raise ValueError(f"ID '{e.args[0]}' not found in vocabulary.")
        
        # Fix spacing around punctuation
        text = re.sub(r'([\'\".,?_/()*!])', r' \1', text)  # Add spaces around punctuation for readability
        text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
        return text
