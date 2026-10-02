"""
nlp_engine.py - NLP Processing Module for CodeAlpha FAQ Chatbot
Uses TF-IDF vectorization and cosine similarity for intent matching.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import FAQS, GREETINGS, FAREWELLS, THANKS

import nltk

# Download required NLTK data silently
for resource, path in [
    ('punkt', 'tokenizers/punkt'),
    ('stopwords', 'corpora/stopwords'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords

# Build stopwords set (with fallback if NLTK fails)
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
        'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
        "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
        'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
        "won't", 'wouldn', "wouldn't"
    }


def preprocess(text: str) -> str:
    """
    Preprocess text for NLP matching.
    Steps: lowercase → remove punctuation → remove stopwords → join.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def classify_intent(user_input: str) -> str:
    """
    Classify the user's intent into one of:
    greeting, farewell, thanks, faq.
    """
    lower = user_input.lower().strip()

    for g in GREETINGS:
        if g in lower or lower.startswith(g):
            return 'greeting'

    for f in FAREWELLS:
        if f in lower:
            return 'farewell'

    for t in THANKS:
        if t in lower:
            return 'thanks'

    return 'faq'


class NLPEngine:
    """
    Core NLP engine using TF-IDF + cosine similarity for FAQ matching.
    """

    def __init__(self):
        self.questions = [faq['question'] for faq in FAQS]
        self.answers = [faq['answer'] for faq in FAQS]
        self.categories = [faq['category'] for faq in FAQS]

        # Build TF-IDF model on preprocessed questions
        processed_questions = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)

    def get_response(self, user_input: str) -> dict:
        """
        Return a response dict with keys:
        answer, confidence, category, intent, (optionally) matched_question.
        """
        intent = classify_intent(user_input)

        # --- Intent-based replies ---
        if intent == 'greeting':
            return {
                'answer': (
                    "Hello! 👋 I'm your AI assistant for Python, Machine Learning, "
                    "and Data Science questions. What would you like to know?"
                ),
                'confidence': 1.0,
                'category': 'Greeting',
                'intent': 'greeting',
            }

        if intent == 'farewell':
            return {
                'answer': "Goodbye! 👋 It was great chatting with you. Keep learning and coding! 🚀",
                'confidence': 1.0,
                'category': 'Farewell',
                'intent': 'farewell',
            }

        if intent == 'thanks':
            return {
                'answer': "You're welcome! 😊 Feel free to ask more questions anytime. Happy coding! 💻",
                'confidence': 1.0,
                'category': 'Thanks',
                'intent': 'thanks',
            }

        # --- FAQ matching ---
        processed_input = preprocess(user_input)

        if not processed_input.strip():
            return {
                'answer': "Could you please rephrase your question? I want to give you the best answer!",
                'confidence': 0.0,
                'category': 'Unknown',
                'intent': 'unknown',
            }

        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])

        if confidence > 0.1:
            return {
                'answer': self.answers[best_idx],
                'confidence': confidence,
                'category': self.categories[best_idx],
                'intent': 'faq',
                'matched_question': self.questions[best_idx],
            }

        # Fallback with suggestions
        suggestions = self._get_suggestions(similarities)
        suggestion_lines = "\n".join(f"  • {q}" for q in suggestions)
        return {
            'answer': (
                "I'm not sure about that. 🤔 Here are some topics I can help with:\n"
                + suggestion_lines
            ),
            'confidence': confidence,
            'category': 'Unknown',
            'intent': 'unknown',
        }

    def _get_suggestions(self, similarities: np.ndarray, top_k: int = 3) -> list:
        """Return the top-k most relevant questions as suggestions."""
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.questions[i] for i in top_indices]
