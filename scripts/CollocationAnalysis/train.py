import json
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

from nltk.tokenize import PunktSentenceTokenizer
import nltk
nltk.download('punkt')

input_file = "finalsubset_commoncorpus.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [doc["text"] for doc in data if "text" in doc and doc["text"]]

#split sentences using NLTK Punkt tokenizer model
tokenizer = PunktSentenceTokenizer()

sentences = []
for text in texts:
    for sentence in tokenizer.tokenize(text):
        processed = simple_preprocess(sentence)
        if processed:
            sentences.append(processed)

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=3,
    workers=4,
    sg=1,
    epochs=10
)

model_path = "word2vec.model"
model.save(model_path)
print(f"Model saved to {model_path}")
