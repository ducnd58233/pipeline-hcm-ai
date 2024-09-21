import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.cli import download
from googletrans import Translator
import asyncio
from typing import List
from app.log import logger

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

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
        self.lemmatizer = WordNetLemmatizer()

    async def translate_to_english(self, query):
        try:
            detected_lang = await asyncio.to_thread(self.translator.detect, query)
            logger.info(f"Detected language: {detected_lang.lang}")

            if detected_lang.lang != 'en':
                translated = await asyncio.to_thread(self.translator.translate, query, dest='en')
                return translated.text
            return query
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return query
        
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    async def extract_relevant_terms(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        relevant_terms = []
        for word, pos in pos_tags:
            if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) and word.lower() not in self.stop_words:
                wordnet_pos = self.get_wordnet_pos(pos)
                lemma = self.lemmatizer.lemmatize(word.lower(), wordnet_pos)
                relevant_terms.append(lemma)

        doc = self.nlp(text)
        noun_phrases = [
            ' '.join([self.lemmatizer.lemmatize(token.text.lower(), self.get_wordnet_pos(token.pos_))
                      for token in chunk
                      if token.text.lower() not in self.stop_words])
            for chunk in doc.noun_chunks
        ]

        all_terms = list(set(relevant_terms + noun_phrases))

        filtered_terms = [term for term in all_terms if len(term) > 1]

        logger.debug(f"Extracted terms: {filtered_terms}")
        return filtered_terms

    async def process_tag_query(self, query: str, additional_entities: List[str]) -> List[str]:
        query_terms = await self.extract_relevant_terms(query) if query else []
        all_terms = list(set(query_terms + additional_entities))
        logger.debug(f"Processed query: {query}")
        logger.debug(f"Extracted terms: {all_terms}")
        return all_terms

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

    def tokenize_and_remove_stopwords(self, text):
        logger.debug(f'Text before tokenize: {text}')
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w not in self.stop_words or w.isdigit()]

    def extract_ner(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.debug(f"Extracted entities: {entities}")
        return entities
