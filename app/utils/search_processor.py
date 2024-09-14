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

logger = logger.getChild(__name__)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class TextProcessor:
    def __init__(self):
        self.nlp = nlp
        self.stop_words = set(stopwords.words('english')) - {'and', 'or'}
        self.translator = Translator()

    async def translate_to_english(self, query):
        try:
            detected_lang = await asyncio.to_thread(self.translator.detect, query)
            logger.debug(f"Detected language: {detected_lang.lang}")

            if detected_lang.lang != 'en':
                translated = await asyncio.to_thread(self.translator.translate, query, dest='en')
                return translated.text
            return query
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return query

    async def preprocess_query(self, query):
        query = query.strip()
        query = self.tokenize_and_remove_stopwords(query)
        return ' '.join(query)

    async def parse_query(self, query):
        preprocessed_query = await self.preprocess_query(query)
        entities = await asyncio.to_thread(self.extract_ner, preprocessed_query)
        return preprocessed_query, entities

    def tokenize_and_remove_stopwords(self, text):
        logger.debug(f'Text before tokenize: {text}')
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words or w.isdigit()]

    def extract_ner(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.debug(f"Extracted entities: {entities}")
        return entities
