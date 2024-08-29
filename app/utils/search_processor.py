
from typing import Any, Dict
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy.cli import download
from googletrans import Translator
import asyncio
from app.log import logger
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
        logger.debug(f"Detected language: {detected_lang.lang}")

        if detected_lang.lang != 'en':
            translated = await asyncio.to_thread(self.translator.translate, query, dest='en')
            query = translated.text

        return query.lower()

    async def parse_query(self, query):
        query = await self.preprocess_query(query)
        doc = self.nlp(query)
        query_structure = []
        current_chunk = []

        for chunk in doc.noun_chunks:
            current_chunk.append(chunk.text)
            for token in doc:
                if token.text.lower() in ["and", "or"]:
                    if current_chunk:
                        query_structure.append(current_chunk)
                        current_chunk = []
                    query_structure.append(token.text.lower())
                elif token.pos in ["NOUN", "PROPN", "ADJ", "VERB", "NUM"]:
                    if not current_chunk or (current_chunk and token.dep != "conj"):
                        current_chunk.append(token.text)
                    else:
                        if current_chunk:
                            query_structure.append(current_chunk)
                        current_chunk = [token.text]

            if current_chunk:
                query_structure.append(current_chunk)

            return query_structure

    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words or w.isdigit()]
