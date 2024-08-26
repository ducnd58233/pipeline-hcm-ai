
from typing import Any, Dict
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy.cli import download
from googletrans import Translator
import asyncio
import logging

logger = logging.getLogger(__name__)

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
        self.stop_words = set(stopwords.words('english')) - {"and", "or"}
        self.translator = Translator()

    async def preprocess_query(self, query):
        query = query.strip()
        detected_lang = await asyncio.to_thread(self.translator.detect, query)
        logger.info(f"Detected language: {detected_lang.lang}")

        if detected_lang.lang != 'en':
            translated = await asyncio.to_thread(self.translator.translate, query, dest='en')
            query = translated.text

        return query.lower()

    async def parse_long_query(self, query):
        query = await self.preprocess_query(query)
        doc = self.nlp(query)

        query_structure = []
        current_chunk = []

        for token in doc:
            if token.text.lower() in ["and", "or"]:
                if current_chunk:
                    query_structure.append(current_chunk)
                    current_chunk = []
                query_structure.append(token.text.lower())
            else:
                current_chunk.append(token.text)

        if current_chunk:
            query_structure.append(current_chunk)

        query_structure.append(doc.text.split())

        return query_structure

    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words or w.isdigit()]


def parse_object_query(object_query: str) -> Dict[str, Any]:
    if not object_query:
        return {}

    query_parts = object_query.split(',')
    parsed_query = {}

    for part in query_parts:
        key_value = part.split(':')
        if len(key_value) == 2:
            key, value = key_value
            key = key.strip()
            value = value.strip()

            try:
                value = int(value)
            except ValueError:
                pass  # Keep it as a string if it's not an integer

            parsed_query[key] = value

    return parsed_query
