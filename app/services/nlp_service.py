from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from app.utils.clip_utils import get_clip_model, encode_text

nltk.download('punkt')
nltk.download('stopwords')

translator = Translator()
stop_words = set(stopwords.words('english'))
clip_model = get_clip_model()


def preprocess_query(query, expected_dim=768):
    # Translate if not in English
    detected_lang = translator.detect(query).lang
    if detected_lang != 'en':
        query = translator.translate(query, dest='en').text

    # Tokenize and remove stop words
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens
    processed_query = ' '.join(filtered_tokens)

    # Encode with CLIP
    query_vector = encode_text(clip_model, processed_query)

    # Ensure correct dimension
    if query_vector.shape[1] != expected_dim:
        # Pad or truncate to match expected dimension
        padded_vector = np.zeros((1, expected_dim), dtype=np.float32)
        padded_vector[:, :min(query_vector.shape[1], expected_dim)] = query_vector[:, :min(
            query_vector.shape[1], expected_dim)]
        query_vector = padded_vector

    return query_vector
