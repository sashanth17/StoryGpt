from Tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer()
print(tokenizer.encode("Hello, World!"))
print(tokenizer.decode(tokenizer.encode("Hello, World!")))