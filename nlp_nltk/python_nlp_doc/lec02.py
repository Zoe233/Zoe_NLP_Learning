# -*- coding:utf-8 -*-
import nltk
from nltk.corpus import gutenberg
from nltk.book import *

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

print(unusual_words(gutenberg.words('austen-sense.txt')))

nltk.corpus.brown.tagged_words()