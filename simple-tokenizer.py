import re
# Sample text
# text = "Text is a book or other written/printed 'work', --regarded in terms of its content rather than its \"physical\" form."

# Actual text data
tokens = re.split(r'([\'\".,?_/()*]|--| -|\s)', text)
tokens = [token.strip() for token in tokens if token.strip()]
print(tokens)
