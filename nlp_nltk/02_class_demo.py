# -*- coding:utf-8 -*-
import nltk
nltk.download()

from nltk.book import *

# 搜索文本
# 搜索单词
text1.concordance('monstrous')
text2.concordance('affection')
text3.concordance('lived')
text5.concordance('lol')

# 搜索相似词
text1.similar('monstrous')
text2.similar('monstrous')

# 搜索共同上下文
text2.common_contexts(['monstrous','very'])

# 词汇分布图
text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America'])

# 自动生成文章
text1.generate()
# 这个功能已经在最新的包内已经被去掉了

# 计数词汇
len(text3)

sorted(set(text3))

len(set(text3))

# 重复词密度
len(text3) / len(set(text3))

# 关键词密度
text3.count('smote')

100* text4.count('a')/len(text4)

def lexical_diversity(text):
    '''
    词汇密度
    '''
    return len(text)/len(set(text))

def percentage(count, total):
    return 100 * count/total

percentage(4,5)

percentage(text4.count('a'),len(text4))



# 文本在Python中的存储方式
# 词链表
sent1 = ['Call','me','Ishmael','.']
sent1

len(sent1)

lexical_diversity(sent1)  # 1.0

print(sent2)
# ['The', 'family', 'of', 'Dashwood', 'had', 'long', 'been', 'settled', 'in', 'Sussex', '.']

print(sent3)
# ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']

# 连接
sent4+sent1

# 追加
sent1.append('some')
print(sent1)

# 索引
text4[173]
# awaken

text4.index('awaken')
# 173

# 切片
print(text5[16715:16735])
print(text6[1600:1625])

# 索引从0开始，要注意
sent = ['word1','word2','word3']
print(sent[0])

### 简单统计
# 频率分布
fdist1 = FreqDist(text1)
fdist1

vocabulary1 = fdist1.keys()
vocabulary1[:50]

fdist1['whale']

fdist1.plot(50,cumulative=True) # 绘制前50个词（高频词）的累积频数分布

fdist1.hapaxes() # 低频词，只出现一次。return ['word1','word2',...]

# 细粒度的选择词
V = set(text4)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

V = set(text5)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

# 词语搭配
from nltk.util import bigrams
a = bigrams(['more','is','said','than','done'])
# <generator object bigrams at 0x1209ace08>
list(a)
# [('more','is'),('is','said'),('said','than'),('than','done')]

text4.collocations()

text8.collocations()










