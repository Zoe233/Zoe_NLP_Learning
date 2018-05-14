# tokenize and tag some text
# tokenize
import nltk
sentence = "At eight o'clock on Tuesday morning Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)
print('tokenize'.center('*',20),tokens)

# tag some text
tagged = nltk.pos_tag(tokens)
tagged[0:6]

# identify named entities
entities = nltk.chunk.ne_chunk(tagged)
entities

# display a parse tree
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()


