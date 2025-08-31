### Coding Sample Part 2 of 2
### Advanced NLP Analysis

## Installing libraries

import importlib
import os
import subprocess
import sys
from functools import lru_cache
import re
import ast
from datetime import datetime

from tqdm import tqdm
tqdm.pandas()


import os

# Run in offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# === Model paths (using absolute paths to avoid errors) ===

# Emotion model (GoEmotions)
EMOTION_MODEL_PATH = "XXXX" ## Replace with actual path

# Detoxify checkpoint
TOXIC_MODEL_PATH = "XXXX" ## Replace with actual path

# HateBERT model
HATE_MODEL_PATH = "XXXX" ## Replace with actual path

# Sentence embeddings model
EMBEDDING_MODEL_PATH = "XXXX" ## Replace with actual path


# Skip Colab-specific installation logic
try:
    import streamlit as st
except Exception:
    st = None

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from detoxify import Detoxify
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import yake
import torch

# GPU check
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    DEVICE = 0
else:
    print("Running on CPU")
    DEVICE = -1

# Confirm required NLTK resources are present (no downloading)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError as e:
    raise RuntimeError(f"Required NLTK resource not found: {e}")

# Safe warning system for Streamlit fallback
def safe_warning(msg):
    if st:
        try:
            st.warning(msg)
        except Exception:
            print(f"WARNING: {msg}")
    else:
        print(f"WARNING: {msg}")

# Safe spinner system
def safe_spinner(msg):
    if st and hasattr(st, 'spinner'):
        return st.spinner(msg)

    class Dummy:
        def __enter__(self): print(msg)
        def __exit__(self, exc_type, exc, tb): pass

    return Dummy()

# Streamlit resource caching fallback
if st and hasattr(st, 'cache_resource'):
    cache_resource = st.cache_resource
else:
    def cache_resource(func=None, **kwargs):
        if func is None:
            def wrapper(fn): return fn
            return wrapper
        return func

# Enforce offline behavior
def has_internet() -> bool:
    return False

# -----------------------------------------------
# Helper functions
# -----------------------------------------------
import re
from functools import lru_cache

@cache_resource(show_spinner=False)
def load_spacy_model():
    """Load spaCy model from local environment and prefer GPU."""
    try:
        spacy.prefer_gpu()
        return spacy.load("en_core_web_sm")
    except OSError:
        safe_warning('spaCy model "en_core_web_sm" not found locally.')
        return spacy.blank("en")


@cache_resource(show_spinner=False)
def load_vader():
    try:
        return SentimentIntensityAnalyzer()
    except Exception as e:
        safe_warning(f"VADER load error: {e}")
        return SentimentIntensityAnalyzer()


from functools import partial

@cache_resource(show_spinner=False)
def load_emotion_model():
    """Load GoEmotions model from local directory, offline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            return_all_scores=True,  # Return all emotion scores
        )
        # Partial wrapper ensures all input is truncated/padded
        return partial(pipe, truncation=True, padding=True)
    except Exception as e:
        safe_warning(f"Emotion model load error: {e}")
        return lambda x: []




@cache_resource(show_spinner=False)
def load_toxic_model():
    """Load Detoxify model using manually downloaded checkpoint."""
    try:
        return Detoxify(
            "unbiased",
            checkpoint=TOXIC_MODEL_PATH,
            device="cuda" if DEVICE == 0 else "cpu",
        )
    except Exception as e:
        safe_warning(f"Toxicity model load error: {e}")
        return lambda x: {"toxicity": 0}


from functools import partial

@cache_resource(show_spinner=False)
def load_hate_model():
    """Load hate-speech detection model from local files."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(HATE_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(HATE_MODEL_PATH)
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        return partial(pipe, truncation=True, padding=True)
    except Exception as e:
        safe_warning(f"Hate speech model load error: {e}")
        return lambda x: [{"label": "LABEL_0", "score": 0.0}]


@cache_resource(show_spinner=False)
def load_yake():
    try:
        return yake.KeywordExtractor(top=5, stopwords=None)
    except Exception as e:
        safe_warning(f"YAKE load error: {e}")
        return yake.KeywordExtractor(top=5, stopwords=None)


# Preprocessing
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    safe_warning("NLTK stopwords not found. Proceeding without stopword removal.")
    stop_words = set()


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


nlp = load_spacy_model()


def batch_preprocess(text_series: pd.Series) -> list:
    cleaned = text_series.apply(clean_text).tolist()
    docs = nlp.pipe(cleaned, batch_size=100)
    results = []
    for doc in docs:
        tokens = [t.lemma_.lower() for t in doc if t.is_alpha and t.text.lower() not in stop_words]
        results.append(tokens)
    return results


# -----------------------------------------------
# Feature Extraction Functions
# -----------------------------------------------

def vader_sentiment(text):
    analyzer = load_vader()
    return analyzer.polarity_scores(text)['compound']


def detect_emotions(text: str, pipe):
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"labels": [], "scores": []}
    try:
        out = pipe(text)  # list of {'label':..., 'score':...}
        # Keep labels above a threshold (tuneable)
        labels = [d["label"].lower() for d in out if d.get("score", 0) >= 0.30]
        return {"labels": labels, "scores": out}
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return {"labels": [], "scores": []}



def detect_toxicity(text):
    model = load_toxic_model()
    if hasattr(model, 'predict'):
        return model.predict(text)
    return {'toxicity': 0}


def detect_hate_speech(text):
    pipe = load_hate_model()
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"label": "N/A", "score": 0.0}
    try:
        outputs = pipe(text)
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            result = outputs[0]
            return {
                "label": result.get("label", "N/A"),
                "score": result.get("score", 0.0)
            }
        else:
            return {"label": "N/A", "score": 0.0}
    except Exception as e:
        print(f"Hate speech detection error: {e}")
        return {"label": "ERROR", "score": 0.0}


def misinformation_flag(row):
    text = row['text']
    if abs(row['sentiment_score']) > 0.8:
        return True
    # now check labels
    emo_labels = row.get('emotions', {}).get('labels', [])
    if any(em in emo_labels for em in ['anger', 'fear', 'disgust']):
        return True
    if re.search(r'([!?.])\1{2,}', text) or text.isupper():
        return True
    triggers = ['hoax', 'fake news', 'deep state', 'truth revealed']
    if any(t in text.lower() for t in triggers):
        return True
    return False


def extract_entities(text):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
    return list(set(ents))


@lru_cache(maxsize=2048)
def _cached_keywords(text: str):
    kw = load_yake()
    return tuple(k[0] for k in kw.extract_keywords(text))


def extract_keywords(text: str) -> list:
    return list(_cached_keywords(text))


def build_topics(texts):
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, local_files_only=True)
    topic_model = BERTopic(embedding_model=embed_model, seed=42)
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    topics, _ = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics


def compute_bot_scores(df):
    user_stats = []
    for user, group in df.groupby('user_id'):
        count = len(group)
        time_diffs = group['timestamp'].sort_values().diff().dt.total_seconds().dropna()
        avg_time = time_diffs.mean() if not time_diffs.empty else 0
        sent_stdev = group['sentiment_score'].std() if len(group) > 1 else 0
        dup_rate = (group['text'].duplicated().sum() / len(group)) if len(group) > 1 else 0

        score = min(1.0, (
            count / 50 +
            (1 if avg_time < 60 else 0) +
            dup_rate +
            (0 if np.isnan(sent_stdev) or sent_stdev == 0 else 1 / sent_stdev)
        ) / 4)

        user_stats.append({
            'user_id': user,
            'post_count': count,
            'avg_time_between_posts': avg_time,
            'sentiment_stdev': sent_stdev,
            'duplication_rate': dup_rate,
            'bot_score': score,
            'likely_bot': score > 0.8
        })

    return pd.DataFrame(user_stats)


# -----------------------------------------------
# Load Data
# -----------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns and 'full_text' in df.columns:
        df = df.rename(columns={'full_text': 'text'})
    if 'user_id' not in df.columns and 'author' in df.columns:
        df = df.rename(columns={'author': 'user_id'})
    if 'timestamp' not in df.columns and 'created_utc' in df.columns:
        df = df.rename(columns={'created_utc': 'timestamp'})
    
    cols = [c for c in ['id', 'text', 'user_id', 'timestamp'] if c in df.columns]
    df = df[cols]
    df.dropna(subset=['text', 'user_id'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    return df


# -----------------------------------------------
# Main preprocessing routine
# -----------------------------------------------

from tqdm import tqdm
tqdm.pandas()

def preprocess(input_path='reddit_sample_5000.csv',
               posts_out='reddit_processed.csv',
               bots_out='bot_stats.csv'):
    
    df = load_data(input_path)

    with safe_spinner('Processing posts...'):
        print("-> Tokenizing...")
        df['tokens'] = batch_preprocess(df['text'])

        print("-> Running VADER sentiment analysis...")
        df['sentiment_score'] = df['text'].progress_apply(vader_sentiment)

        print("-> Detecting emotions...")
        emotion_pipeline = load_emotion_model()
        df['emotions'] = df['text'].progress_apply(lambda x: detect_emotions(x, emotion_pipeline))

        print("-> Detecting toxicity...")
        df['toxicity_scores'] = df['text'].progress_apply(detect_toxicity)

        print("-> Detecting hate speech...")
        df['hate_speech'] = df['text'].progress_apply(detect_hate_speech)

        print("-> Flagging potential misinformation...")
        df['misinfo_flag'] = df.progress_apply(misinformation_flag, axis=1)

        print("-> Extracting named entities...")
        df['entities'] = df['text'].progress_apply(extract_entities)

        print("-> Extracting keywords...")
        df['keywords'] = df['text'].progress_apply(extract_keywords)

        print("-> Building topic model...")
        topic_model, topics = build_topics(df['text'].tolist())
        df['topic'] = topics

    print("-> Computing bot scores...")
    bot_stats = compute_bot_scores(df)

    print("-> Saving outputs...")
    df.to_csv(posts_out, index=False)
    bot_stats.to_csv(bots_out, index=False)
    print(f"[OK] Saved processed posts to: {posts_out}")
    print(f"[OK] Saved bot stats to: {bots_out}")


preprocess(
    input_path='reddit_sample_5000.csv',
    posts_out='reddit_processed.csv',
    bots_out='bot_stats.csv'
)