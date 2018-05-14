# -*- coding:utf-8 -*-
import nltk
import numpy as np
import matplotlib.pyplot as plt
import tweepy  # twitter信息抓取包
import TwitterSearch  # twitter api
import unidecode  # 文字处理利器
import langdetect  # 语言检测工具
import langid  # 语言检测工具
import gensim
## 是一个python的自然语言处理库，
# 能够将文档根据TF-IDF,LDA,LSI等模式转化成向量模式，以便进行进一步的处理。
# 还实现了word2vec 功能，能够将单词转化为词向量。

# You also need to run nltk.download() in order to download NLTK before proceeding:
# nltk.download()

# virtualenv venv
# source venv/bin/activate

# pip install -r requirements.txt


# 1-2 Text Analysis Using nltk.text
from nltk.tokenize import word_tokenize
from nltk.text import Text

my_string = "Two plus two is four, minus one that's three -- quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."
tokens = word_tokenize(my_string)
tokens = [word.lower() for word in tokens]
print(tokens[:5])


