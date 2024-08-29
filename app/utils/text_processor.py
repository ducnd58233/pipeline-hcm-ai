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

        tokens = word_tokenize(query.lower())
        filtered_tokens = [
            word for word in tokens if word not in self.stop_words]

        processed_query = ' '.join(filtered_tokens)

        return processed_query

    async def parse_query(self, query):
        query = await self.preprocess_query(query)
        return query

    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words]
