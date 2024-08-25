import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy.cli import download
from googletrans import Translator
import asyncio

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class TextProcessor:
    def __init__(self):
        self.nlp = nlp
        self.stop_words = set(stopwords.words('english'))
        self.translator = Translator()

    async def preprocess_query(self, query):
        detected_lang = await asyncio.to_thread(self.translator.detect, query)
        if detected_lang.lang != 'en':
            translated = await asyncio.to_thread(self.translator.translate, query, dest='en')
            query = translated.text

        # Tokenize and remove stop words, but preserve "and" and "or"
        tokens = word_tokenize(query.lower())
        filtered_tokens = [
            word for word in tokens if word not in self.stop_words or word in {"and", "or"}]

        processed_query = ' '.join(filtered_tokens)

        return processed_query

    async def parse_long_query(self, query):
        query = await self.preprocess_query(query)

        doc = self.nlp(query)
        query_structure = []
        current_group = []

        for sent in doc.sents:
            for token in sent:
                if token.text.lower() in ("and", "or"):
                    if current_group:
                        query_structure.append(current_group)
                        current_group = []
                    query_structure.append([token.text])
                elif token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
                    current_group.append(token.text)

        if current_group:
            query_structure.append(current_group)

        return query_structure

    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words]


