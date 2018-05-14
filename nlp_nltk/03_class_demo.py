# -*- coding:utf-8 -*-

import nltk
nltk.corpus.gutenberg.fileids()
# ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print(len(emma))
# 192427
print(type(emma))
# <class 'nltk.corpus.reader.util.StreamBackedCorpusView'>

t = nltk.Text(emma)
print(type(t))
# <class 'nltk.text.Text'>

t.concordance('surprize')
# displaying 25 of 37 matches:
# ...

from nltk.corpus import gutenberg
gutenberg.fileids()

emma = gutenberg.words('austen-emma.txt')
