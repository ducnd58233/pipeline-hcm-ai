from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import asyncio

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

translator = Translator()
stop_words = set(stopwords.words('english'))


async def preprocess_query(query):
    # Translate if not in English
    detected_lang = await asyncio.to_thread(translator.detect, query)
    if detected_lang.lang != 'en':
        query = await asyncio.to_thread(translator.translate, query, dest='en')
        query = query.text

    # Tokenize and remove stop words
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens
    processed_query = ' '.join(filtered_tokens)

    return processed_query
