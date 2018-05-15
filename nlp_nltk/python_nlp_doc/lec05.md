第5章 分词和标注词汇
5.1 使用词性标注器
5.2 标注语料库
    表示已标注的标识符
    读取已标注的语料库
    简化的词性标记集
    名词
    动词
    形容词和副词
    未简化的标记
    探索已标注的语料库
5.3 使用Python字典映射词及其属性
5.4 自动标注
    默认标注器
    正则表达式标注器
    查询标注器
    评估
5.5 N-gram标注
    一元标注（Unigram Tagging)
    分离训练和测试数据
    一般的N-gram的标注
    组合标注器
    标注生词
    存储标注器
    性能限制
    跨句子边界标注
5.6 基于转换的标注
5.7 如何确定一个词的分类
    形态学线索
    句法线索
    语义线索
    新词
    词性标注集中的形态学
5.8 小结


    名词、动词、形容词和副词之间的差异。

    这些"词类"不是闲置的文法家的发明，而是对许多语言处理任务都有用的分类。

    这些分类源于对文本中词的分布的简单的分析。

    本章的目的是要回答下列问题：
        1. 什么是词汇分类，在自然语言处理中它们是如何使用？
        2. 一个好的存储词汇和它们的分类的Python 数据结构是什么？
        3. 我们如何自动标注文本中词汇的词类？

    我们将介绍NLP 的一些基本技术，包括
        序列标注、N-gram 模型、回退和评估。
    这些技术在许多方面都很有用，标注为我们提供了一个表示它们的简单的上下文。
    我们还将看到标注为何是典型的NLP 流水线中继分词之后的第二个步骤。

    将词汇按它们的词性（parts-of-speech，POS）分类以及相应的标注它们的过程被称为词性标注（part-of-speech tagging, POS tagging）或干脆简称标注。
    词性也称为词类或词汇范畴。
    用于特定任务的标记的集合被称为一个标记集。我们在本章的重点是利用标记和自动标注文本。

5.1 使用词性标注器
    一个词性标注器Part-of-speech tagger或POS tagger处理一个词序列，为每个词附加一个词性标记。

        >>> import nltk
        >>> text = nltk.word_tokenize('And now for someting completely different')
        >>> nltk.pos_tag(text)
        [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('someting', 'VBG'), ('completely', 'RB'), ('different', 'JJ')]

        在这里我们看到and 是CC，并列连词；
                    now 和completely 是RB，副词；
                    for 是IN，介词；
                    something 是NN，名词；
                    different 是JJ，形容词。
    NLTK中提供了每个标记的文档，可以使用标记来查询，如：nltk.help.upenn_tagset('RB')，或正则表达式，如：nltk.help.upenn_brown_tagset('NN.*').

    一个标注器也可以为我们对未知词的认识过程建模；
    例如：我们可以根据词根scrobble猜测scrobbling 可能是一个动词，并有可能发生在he was scrobbling 这样的上下文中。

5.2 标注语料库
5.2.1 表示已标注的标识符
    按照NLTK的约定，一个已标注的标识符使用一个由标识符和标记组成的元组来表示。
    我们可以使用函数str2tuple()从表示一个已标注的标识符的标准字符串创建一个这样的特殊元组:
        >>> tagged_token = nltk.tag.str2tuple('fly/NN')
        >>> tagged_token
        ('fly', 'NN')
        >>> tagged_token[0]
        'fly'
        >>> tagged_token[1]
        'NN'

5.2.2 读取已标注的语料库
    NLTK中包括的若干语料库 已标注 了词性。
    下面是一个你用文本编辑器打开一个布朗语料库的文件就能看到的例子：
        The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr ...

    其他语料库使用各种格式存储词性标记。
    NLTK中的语料库阅读器提供了一个统一的接口，使你不必理会这些不同的文件格式。

    与刚才提取并显示的上面的文件不同，布朗语料库的月料库阅读器按如下所示的方式表示数据。
    注意：部分词性标记已转换为大写的；自从布朗语料库发布以来，这已成为标准的做法。

        >>> nltk.corpus.brown.tagged_words()
        [('The', 'AT'), ('Fulton', 'NP-TL'), ...]

    只要语料库包含已标注的文本，NLTK的语料库接口都将有一个tagged_words()方法。

    并非所有的语料库都采用同一组标记。
    最初，我们想避免这些标记集的复杂化，所以我们使用一个内置的到一个简化的标记集的映射：
        >>> nltk.corpus.brown.tagged_words(tagset='universal')



5.2.3 简化的词性标记集
    已标注的语料库使用许多不同的标记集约定来标注词汇。

        >>> from nltk.corpus import brown
        >>> brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
        >>> tag_fdist = nltk.FreqDist(tag for (word,tag) in brown_news_tagged)
        >>> tag_fdist.keys()
        >>> tag_fdist.plot(cumulative=True)

    我们可以使用这些标记做强大的搜索，结合一个图形化的POS一致性工具nltk.app.concordance()。
    用它来寻找任一词和POS标记的组合，如：N N N N , hid/VD, hit/VN或the ADJ man.



5.2.4 名词
5.2.5 动词
5.2.6 形容词和副词
5.2.7 未简化的标记
5.2.8 探索已标注的语料库

5.3 使用Python字典映射词及其属性
    (word,tag)形式的一个已标注是 词和词性标记的关联。

    一旦我们开始做词性标注，我们将会创建分配一个标记给一个词的程序，标记是在给定上下文中最可能的标记。
    我们可以认为这个过程是从词到标记的映射。

    在Python中最自然的方式存储映射是使用所谓的字典数据类型（在其他编程语言中又称为 关联数组 或 哈希数组）。

5.4 自动标注
    以不同的方式来给文本自动添加词性标记。

    我们将看到一个词的标记依赖于这个词和它在句子中的上下文。
    出于这个原因，我们将处理（已标注）句子层次而不是词汇层次的数据。
    我们以加载将要使用的数据开始。
        >>> from nltk.corpus import brown
        >>> brown_tagged_sents = brown.tagged_sents(categories='news')
        >>> brown_sents = brown.sents(categories='news')

5.4.1 默认标注器
    最简单的标注器是为每个标识符分配同样的标记。

    为了得到最好的效果，我们用最可能的标记标注每个词。
    让我们找出哪个标记是最优可能的。

    默认的标注器给每一个单独的词分配标记，即使是之前从未遇到过的词。
    碰巧的是，一旦我们处理了几千词的英文文本后，大多数新词都是名词。

    so，默认标注器可以帮我们提高语言处理系统的稳定性。

        >>> tags =[tag for (word,tag) in brown.tagged_words(categories='news')]
        >>> nltk.FreqDist(tags).max()
        'NN'
        >>> raw = 'I do not like green eggs and larm, I do not like them Sam I am !'
        >>> tokens = nltk.word_tokenize(raw)
        >>> default_tagger = nltk.DefaultTagger('NN')
        >>> default_tagger.tag(tokens)
        [('I', 'NN'), ('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('green', 'NN'), ('eggs', 'NN'), ('and', 'NN'), ('larm', 'NN'), (',', 'NN'), ('I', 'NN'), ('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('them', 'NN'), ('Sam', 'NN'), ('I', 'NN'), ('am', 'NN'), ('!', 'NN')]
        >>> default_tagger.evaluate(brown_tagged_sents)
        0.13089484257215028



5.4.2 正则表达式标注器
    正则表达式标注器基于匹配模式分配标记给标识符。
    例如：我们可能会猜测任一以ed结尾的词都是动词过去分词，任一以's 结尾的词都是名词所有格。
    可以用一个正则表达式的列表表示这些：

        >>> patterns = [
            ... (r'.*ing$', 'VBG'), # gerunds
            ... (r'.*ed$', 'VBD'), # simple past
            ... (r'.*es$', 'VBZ'), # 3rd singular present
            ... (r'.*ould$', 'MD'), # modals
            ... (r'.*\'s$', 'NN$'), # possessive nouns
            ... (r'.*s$', 'NNS'), # plural nouns
            ... (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
            ... (r'.*', 'NN') # nouns (default)
            ... ]
    请注意，这些是顺序处理的，第一个匹配上的会被使用。

    现在我们可以建立一个标注器，并用它来标记一个句子。
        >>> regexp_tagger = nltk.RegexpTagger(patterns)
        >>> regexp_tagger.tag(brown_sents[3])
        [('``', 'NN'), ('Only', 'NN'), ('a', 'NN'), ('relative', 'NN'), ('handful', 'NN'),
        ('of', 'NN'), ('such', 'NN'), ('reports', 'NNS'), ('was', 'NNS'), ('received', 'VBD'),
        ("''", 'NN'), (',', 'NN'), ('the', 'NN'), ('jury', 'NN'), ('said', 'NN'), (',', 'NN'),
        ('``', 'NN'), ('considering', 'VBG'), ('the', 'NN'), ('widespread', 'NN'), ...]
        >>> regexp_tagger.evaluate(brown_tagged_sents)
        0.20326391789486245
    最终的正则表达式«.*»是一个全面捕捉的，标注所有词为名词。
    除了作为正则表达式标注器的一部分重新指定这个，这与默认标注器是等效的（只是效率低得多）。
    有没有办法结合这个标注器和默认标注器呢？我们将很快看到如何做到这一点。


5.4.3 查询标注器
    很多高频词没有NN 标记。让我们找出100 个最频繁的词，存储它们最有可能的标记。
    然后我们可以使用这个信息作为“查找标注器”（NLTK UnigramTagger）的模型：
        >>> fd = nltk.FreqDist(brown.words(categories='news'))
        >>> cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
        >>> most_freq_words = []
        >>> for e in fd.keys():
        ...     if len(most_freq_words)<=100:
        ...             most_freq_words.append(e)
        ...     else:
        ...             break
        ...
        >>> most_freq_words
        ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.', 'jury', 'further', 'in', 'term-end', 'presentments', 'the', 'City', 'Executive', 'Committee', ',', 'which', 'had', 'over-all', 'charge', 'deserves', 'praise', 'and', 'thanks', 'Atlanta', 'for', 'manner', 'was', 'conducted', 'September-October', 'term', 'been', 'charged', 'by', 'Superior', 'Court', 'Judge', 'Durwood', 'Pye', 'to', 'investigate', 'reports', 'possible', 'hard-fought', 'won', 'Mayor-nominate', 'Ivan', 'Allen', 'Jr.', 'Only', 'a', 'relative', 'handful', 'such', 'received', 'considering', 'widespread', 'interest', 'number', 'voters', 'size', 'this', 'city', 'it', 'did', 'find', 'many', "Georgia's", 'registration', 'laws', 'are', 'outmoded', 'or', 'inadequate', 'often', 'ambiguous', 'It', 'recommended', 'legislators', 'act', 'have', 'these']
        >>> likely_tags = dict((word,cfd[word].max()) for word in most_freq_words)
        >>> likely_tags
        {'The': 'AT', 'Fulton': 'NP-TL', 'County': 'NN-TL', 'Grand': 'JJ-TL', 'Jury': 'NN-TL', 'said': 'VBD', 'Friday': 'NR', 'an': 'AT', 'investigation': 'NN', 'of': 'IN', "Atlanta's": 'NP$', 'recent': 'JJ', 'primary': 'NN', 'election': 'NN', 'produced': 'VBD', '``': '``', 'no': 'AT', 'evidence': 'NN', "''": "''", 'that': 'CS', 'any': 'DTI', 'irregularities': 'NNS', 'took': 'VBD', 'place': 'NN', '.': '.', 'jury': 'NN', 'further': 'JJR', 'in': 'IN', 'term-end': 'NN', 'presentments': 'NNS', 'the': 'AT', 'City': 'NN-TL', 'Executive': 'NN-TL', 'Committee': 'NN-TL', ',': ',', 'which': 'WDT', 'had': 'HVD', 'over-all': 'JJ', 'charge': 'NN', 'deserves': 'VBZ', 'praise': 'NN', 'and': 'CC', 'thanks': 'NNS', 'Atlanta': 'NP', 'for': 'IN', 'manner': 'NN', 'was': 'BEDZ', 'conducted': 'VBN', 'September-October': 'NP', 'term': 'NN', 'been': 'BEN', 'charged': 'VBN', 'by': 'IN', 'Superior': 'JJ-TL', 'Court': 'NN-TL', 'Judge': 'NN-TL', 'Durwood': 'NP', 'Pye': 'NP', 'to': 'TO', 'investigate': 'VB', 'reports': 'NNS', 'possible': 'JJ', 'hard-fought': 'JJ', 'won': 'VBD', 'Mayor-nominate': 'NN-TL', 'Ivan': 'NP', 'Allen': 'NP', 'Jr.': 'NP', 'Only': 'RB', 'a': 'AT', 'relative': 'JJ', 'handful': 'NN', 'such': 'JJ', 'received': 'VBD', 'considering': 'IN', 'widespread': 'JJ', 'interest': 'NN', 'number': 'NN', 'voters': 'NNS', 'size': 'NN', 'this': 'DT', 'city': 'NN', 'it': 'PPS', 'did': 'DOD', 'find': 'VB', 'many': 'AP', "Georgia's": 'NP$', 'registration': 'NN', 'laws': 'NNS', 'are': 'BER', 'outmoded': 'JJ', 'or': 'CC', 'inadequate': 'JJ', 'often': 'RB', 'ambiguous': 'JJ', 'It': 'PPS', 'recommended': 'VBD', 'legislators': 'NNS', 'act': 'NN', 'have': 'HV', 'these': 'DTS'}
        >>> baseline_tagger = nltk.UnigramTagger(model=likely_tags)
        >>> baseline_tagger.evaluate(brown_tagged_sents)
        0.33352228653260935

    仅仅知道100个最频繁的词的标记就使我们能够正确标注很大一部分标识符。

    让我们来看看它在未标注的输入文本上做的如何：
        >>> sent = brown.sents(categories='news')[3]
        >>> sent
        ['``', 'Only', 'a', 'relative', 'handful', 'of', 'such', 'reports', 'was', 'received', "''", ',', 'the', 'jury', 'said', ',', '``', 'considering', 'the', 'widespread', 'interest', 'in', 'the', 'election', ',', 'the', 'number', 'of', 'voters', 'and', 'the', 'size', 'of', 'this', 'city', "''", '.']
        >>> baseline_tagger.tag(sent)
        [('``', '``'), ('Only', 'RB'), ('a', 'AT'), ('relative', 'JJ'), ('handful', 'NN'), ('of', 'IN'), ('such', 'JJ'), ('reports', 'NNS'), ('was', 'BEDZ'), ('received', 'VBD'), ("''", "''"), (',', ','), ('the', 'AT'), ('jury', 'NN'), ('said', 'VBD'), (',', ','), ('``', '``'), ('considering', 'IN'), ('the', 'AT'), ('widespread', 'JJ'), ('interest', 'NN'), ('in', 'IN'), ('the', 'AT'), ('election', 'NN'), (',', ','), ('the', 'AT'), ('number', 'NN'), ('of', 'IN'), ('voters', 'NNS'), ('and', 'CC'), ('the', 'AT'), ('size', 'NN'), ('of', 'IN'), ('this', 'DT'), ('city', 'NN'), ("''", "''"), ('.', '.')]

    许多词都被分配了一个None的标签，因为它们不在100个最频繁的词中。
    在这些情况下，我们想分配默认标记为NN。

    换句话说，我们要先使用查找表，如果它不能指定一个标记就使用默认标注器，这个过程叫做 回退。
    我们可以通过指定一个标注器作为另一个标注器的参数做到这个。

    现在查找标注器将只存储名词以外的词的词-标记对，只要它不能给一个词分配标记，它将会调用默认标注器。
        >>> baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))

    示例：
        查找标注器的性能，使用不同大小的模型。
            # -*- coding:utf-8 -*-
            import nltk
            from nltk.corpus import brown

            def performance(cfd, wordlist):
                lt = dict((word, cfd[word].max()) for word in wordlist)
                baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
                return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

            def display():
                import pylab
                words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
                cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
                sizes = 2** pylab.arange(5)
                perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
                pylab.plot(sizes,perfs,'-bo')
                pylab.title('Lookup Tagger Performance with Varying Model Size.')
                pylab.xlabel('Model Size')
                pylab.ylabel('Performance')
                pylab.show()

            display()

    观察图5-4，随着模型规模的增长，最初的性能增加迅速，最终达到一个稳定水平，这时模型的规模大量增加性能的提高很小。



5.4.4 评估
    到对准确性得分的强调。 -- NLP的一个中心主题。

    一个模块输出中的任何错误都在下游模块大大的放大。

    我们对比专家分配的标记来评估一个标注器的性能。
    由于我们通常很难获得专业和公正的人的判断，所以使用 黄金标准 测试数据来代替。

    这是一个已经手动标注并作为自动系统评估标准而被接受的语料库。
    当标注器对给定词猜测的标记与黄金标准标记相同，标注器被视为是正确的。

    当然，设计和实施原始的黄金标准标注的也是人，更深入的分析可能会显示黄金标准中的错误，或者可能最终会导致一个修正的标记集和更复杂的知道方针。
    然而，黄金标准就目前有关的自动标注器的评估而言被定义为"正确的"。




5.5 N-gram标注
5.5.1 一元标注（Unigram Tagging)
    一元标注器基于一个简单的统计算法：
        对每个标识符分配这个独特的标识符最有可能的标记。

    例如：
        它将分配标记JJ 给词frequent 的所有出现，
        因为frequent 用作一个形容词（例如：a frequent word）比用作一个动词（例如：I frequent this cafe）更常见。
    一个一元标注器的行为就像一个查找标注器，除了有一个更方便的建立它的技术，称为训练。

    例子：
        我们训练一个一元标注器，用它来标注一个句子，然后评估：
            >>> from nltk.corpus import brown
            >>> brown_tagged_sents = brown.tagged_sents(categories='news')
            >>> brown_sents = brown.sents(categories='news')
            >>> unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
            >>> unigram_tagger.tag(brown_sents[2007])
            [('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'QL'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
            >>> unigram_tagger.evaluate(brown_tagged_sents)
            0.9349006503968017

    我们训练一个UnigramTagger，通过在我们初始化标注器时指定已标注的句子数据作为参数。
    训练过程中涉及检查每个词的标记，将所有词的最可能的标记存储在一个字典里面，这个字典存储在标注器内部。



5.5.2 分离训练和测试数据
    现在，我们正在一些数据上训练一个标注器，必须小心不要在相同的数据上测试，如我们在前面的例子中的那样。
    一个只是记忆它的训练数据，而不试图建立一个一般的模型的标注器会得到一个完美的得分，但在标注新的文本时将是无用的。

        >>> size = int(len(brown_tagged_sents)*0.9)
        >>> size
        4160
        >>> train_sents = brown_tagged_sents[:size]
        >>> test_sents = brown_tagged_sents[size:]
        >>> unigram_tagger = nltk.UnigramTagger(train_sents)
        >>> unigram_tagger.evaluate(test_sents)
        0.8121200039868434


5.5.3 一般的N-gram的标注
    在基于unigrams处理一个语言处理任务时，我们使用上下文中的一个项目。

    标注的时候，我们只考虑当前的标识符，与更大的上下文隔离。

    给定一个模型，我们能做的最好的是为每个词标注其先验的最可能的标记。
    这意味着我们将使用相同的标记标注一个词，如wind，不论它出现的上下文是the wind还是to wind。

    一个n-gram标注器 是一个一元unigram标注器的一般化，它的上下文是当前词和它迁建n-1个标识符的词性标记。

    p193 图5-5 标注器上下文

    注：
        1-gram标注器是一元标注器unigram tagger另一个名称：即用于标注一个标识符的上下文的只是标识符本身。
        2-gram标注器也称为二元标注器bigram taggers；
        3-gram标注器也称为三元标注器trigram taggers.

    NgramTagger类使用一个已标注的训练语料库来确定对每个上下文哪个词性标记最有可能。


    在这里，我们看到一个n-gram标注器的特殊情况，即一个bigram标注器。
        >>> bigram_tagger = nltk.BigramTagger(train_sents)
        >>> bigram_tagger.tag(brown_sents[2007])
        [('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'CS'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
        >>> unseen_sent = brown_sents[4203]
        >>> bigram_tagger.tag(unseen_sent)
        [('The', 'AT'), ('population', 'NN'), ('of', 'IN'), ('the', 'AT'), ('Congo', 'NP'), ('is', 'BEZ'), ('13.5', None), ('million', None), (',', None), ('divided', None), ('into', None), ('at', None), ('least', None), ('seven', None), ('major', None), ('``', None), ('culture', None), ('clusters', None), ("''", None), ('and', None), ('innumerable', None), ('tribes', None), ('speaking', None), ('400', None), ('separate', None), ('dialects', None), ('.', None)]

    请注意，bigram标注器能够标注训练中它看到过的句子中的所有词，但对一个没见过的句子表现很差。
    只要遇到一个新词，就无法给它分配标记。
    它不能标注下面的词，如million，即使是在训练过程中看到过的，只是因为在训练过程中从来没有见过它前面有一个None标记的词。
    因此，标注器标注句子的其余部分也失败了。
    它的整体准确度得分非常低：
        >>> bigram_tagger.evaluate(test_sents)
        0.10206319146815508

    当n越大，上下文的特异性就会增加，我们要标注的数据中包含训练数据中不存在的上下文的几率也增大。
    这杯称为 数据稀疏问题，在NLP中是相当普遍的。

    因此，我们的研究结果的精度和覆盖范围之间需要有一个权衡。
    —— 信息检索中的 精度/召回 权衡。

    注:
        N-gram标注器不应考虑跨越句子边界的上下文。
        因此，NLTK的标注器被设计用于句子链表，一个句子是一个词链表。
        在一个句子的开始，t_{n-1}和前面的标记被设置为None.


5.5.4 组合标注器
    解决精度和覆盖范围之间的权衡的一个办法是尽可能的使用更精确的算法，但却在很多时候落后于具有更广覆盖范围的算法。

    例如：
        我们可以按如下方式组合bigram标注器，unigram标注器和一个默认标注器。

        1. 尝试使用bigram 标注器标注标识符。
        2. 如果bigram 标注器无法找到一个标记，尝试unigram 标注器。
        3. 如果unigram 标注器也无法找到一个标记，使用默认标注器。

    大多数NLTK标注器允许指定一个回退标注器。
    回退标注器自身可能也有一个回退标注器。
        >>> t0 = nltk.DefaultTagger('NN')
        >>> t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        >>> t2 = nltk.BigramTagger(train_sents, backoff=t1)
        >>> t2.evaluate(test_sents)
        0.8452108043456593

    请注意，我们在标注器初始化时指定回退标注器，从而使训练能利用回退标注器。

    如果在一个上下文中的bigram标注器将分配与它的unigram回退标注器一样的标记，那么bigram标注器丢弃训练的实例。
    这样保持尽可能小的bigram标注器模型。
    我们可以进一步指定一个标注器需要看到一个上下文的多个实例才能保留它。

    例如：nltk.BigramTagger(sents, cutoff=2, backoff=t1) 将会丢弃那些只看到一次或两次的上下文。



5.5.5 标注生词
    我们标注生词的方法仍然是回退到一个正则表达式标注器或一个默认标注器。
    这些都无法利用上下文。

    因此，如果我们的标注器遇到词blog，训练过程中没有看到过，它会分配相同的标记，不论这个词出现的上下文是the blog还是to blog。
    我们怎样才能更好地处理这些生词，或者词汇表以外的项目？

    一个有用的基于上下文标注生词的方法是限制一个标注器的词汇表为最频繁的n个词。
    使用Python字典映射词及其属性 --  (word,tag)形式的一个已标注是 词和词性标记的关联
    的方法替代每个其他的词为一个特殊的词UNK。

    训练时，一个unigram标注器可能会学到UNK通常是一个名词。
    然而，n-gram标注器会检测它的一些其他标记中的上下文。

    例如：如果前面的词是to(标注为TO),那么UNK可能对标注为一个动词。


5.5.6 存储标注器
    在大语料库上训练一个标注器可能需要大量的时间。
    没有必要在每次我们需要的时候训练一个标准器，很容易将一个训练好的标注器保存到一个文件以后重复使用。

    # 保存我们的标注器t2到文件t2.pkl:
    pickle load dump *.pkl文件

5.5.7 性能限制
    一个n-gram 标注器的性能上限是什么？
    方法1：
        考虑一个trigram 标注器的情况。它遇到多少词性歧义的情况？
        我们可以根据经验决定这个问题的答案：
        假设我们总是挑选在这种含糊不清的上下文最优可能的标记，可以得出trigram标注器性能的一个下界。

    方法2：
        研究它的错误。
        有些不标记可能会被别的更难分配，可能需要专门对这些数据进行预处理或后处理。
        一个方便的方式查看标注错误是 混淆矩阵。
        它用图表表示期望的标记（黄金标准）与实际由标注器产生的标记。

        >>> test_tags = [tag for sent in brown.sents(categories='editorial')
        ...     for (word,tag) in t2.tag(sent)]
        >>> gold_tags = [tag for (word,tag) in brown.tagged_words(categories='editorial')]

        >>> print(nltk.ConfusionMatrix(gold_tags,test_tags))

        基于这样的分析，我们可能会决定修改标记集。
        或许标记之间很难做出的区分可以被丢弃，因为它在一些较大的处理任务的上下文中并不重要。

    方法3：
        来自人类标注者之间并非100％的意见一致。

        一般情况下，标注过程会消除区别：
            例如：当所有的人称代词被标注为PRP 时，词的特性通常会失去。
            与此同时，标注过程引入了新的区别从而去除了含糊之处：
            例如：deal标注为VB 或NN。
            这种消除某些区别并引入新的区别的特点是标注的一个重要的特征，有利于分类和预测。
            当我们引入一个标记集的更细的划分时，在n-gram 标注器决定什么样的标记分配给一个特定的词时，可以获得关于左侧上下文的更详细的信息。
            然而，标注器同时也将需要做更多的工作来划分当前的标识符，只是因为有更多可供选择的标记。
            相反，使用较少的区别（如简化的标记集），标注器有关上下文的信息会减少，为当前标识符分类的选择范围也较小。

    训练数据中的歧义导致标注器性能的上限。
    有时更多的上下文能解决这些歧义。

    然而，在其他情况下，如(Abney, 1996)中指出的，只有参考语法或现实世界的知识，才能解决歧义。
    尽管有这些缺陷，词性标注在用统计方法进行自然语言处理的兴起过程中起到了核心作用。
    1990 年代初，统计标注器令人惊讶的精度是一个惊人的示范：可以不用更深的语言学知识解决一小部分语言理解问题，即词性消歧。

5.5.8 跨句子边界标注
    一个n-gram 标注器使用最近的标记作为为当前的词选择标记的指导。
    当标记一个句子的第一个词时，trigram 标注器将使用前面两个标识符的词性标记，这通常会是前面句子的最后一个词和句子结尾的标点符号。
    然而，在前一句结尾的词的类别与下一句的开头的通常没有关系。
    为了应对这种情况，我们可以使用已标注句子的链表来训练、运行和评估标注器。

        >>> brown_tagged_sents = brown.tagged_sents(categories='news')
        >>> brown_sents = brown.sents(categories='news')
        >>> size = int(len(brown_tagged_sents)*0.9)
        >>> train_sents = brown_tagged_sents[:size]
        >>> test_sents = brown_tagged_sents[size:]
        >>> t0 = nltk.DefaultTagger('NN')
        >>> t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        >>> t2 = nltk.BigramTagger(train_sents, backoff=t1)
        >>> t2.evaluate(test_sents)
        0.8452108043456593


5.6 基于转换的标注

    n-gram 标注器的一个潜在的问题：
    问题1： 它们的n-gram表的大小（或语言模型）。
        如果使用各种语言技术的标注器部署在移动计算设备上，在模型大小和标注器性能之间取得平衡是很重要的。

        使用回退标注器的n-gram标注器可能存储trigram和bigram表，这是很大的稀疏矩阵列。
    问题2： 上下文。
        n-gram标注器从前面的上下文中获得的唯一的信息是标记，虽然词本身可能是一个有用的信息源。
        n-gram模型使用上下文中的词的其他特征为条件是不切实际的。

    本节中，我们考擦Brill标注，一种归纳标注方法，它的性能很好，使用的模型只有n-gram标注器的很小一部分。
    Brill标注是一种基于转换你的学习，以它的发明者命名。

    一般的想法很简单：
        猜每个词的标记，然后返回和修复错误的。
        在这种方式中，Brill标注器陆续将一个不良标注的文本转换成一个更好的。
    与n-gram标注一样，这是有监督的学习方法，因为我们需要已标注的训练数据来评估标注器的猜测是否是一个错误。
    然而，不像n-gram标注，它不计数观察结果，只编制一个转换修正规则链表。

    Brill 标注的的过程通常是与绘画类比来解释的。
    假设我们要画一棵树，包括大树枝、树枝、小枝、叶子和一个统一的天蓝色背景的所有细节。
    不是先画树然后尝试在空白处画蓝色，而是简单的将整个画布画成蓝色，然后通过在蓝色背景上上色“修正”树的部分。
    以同样的方式，我们可能会画一个统一的褐色的树干再回过头来用更精细的刷子画进一步的细节。

    Brill 标注器的另一个有趣的特性：
        规则是语言学可解释的。
        与采用潜在的巨大的n-gram 表的n-gram 标注器相比，我们并不能从直接观察这样的一个表中学到多少东西，
        而Brill标注器学到的规则可以。


5.7 如何确定一个词的分类
    我们已经详细研究了词类，现在转向一个更基本的问题：如何决定一个词属于哪一类？
    首先应该考虑什么？
        在一般情况下，语言学家使用形态学、句法和语义线索确定一个词的类别。

5.7.1 形态学线索
    -ness
    -ment
    -ing
    -ed

5.7.2 句法线索
    上下文语境。
    the near window ==>near为形容词

5.7.3 语义线索
5.7.4 新词
    名词 是 开放类
    借此 是 封闭类，词类成员随着很长时间的推移才逐渐改变。
5.7.5 词性标注集中的形态学

5.8 小结
    1. 词可以组成类，如名词、动词、形容词以及副词。这些类被称为词汇范畴或者词性。
        词性被分配短标签或者标记，如NN 和VB。
    2. 给文本中的词自动分配词性的过程称为词性标注、POS 标注或只是标注。
    3. 自动标注是NLP 流程中重要的一步，在各种情况下都十分有用，包括预测先前未见过的词的行为、分析语料库中词的使用以及文本到语音转换系统。
    4. 一些语言学语料库，如布朗语料库，已经做了词性标注。
    5. 有多种标注方法，如默认标注器、正则表达式标注器、unigram 标注器、n-gram 标注器。这些都可以结合一种叫做回退的技术一起使用。
    6. 标注器可以使用已标注语料库进行训练和评估。
    7. 回退是一个组合模型的方法：当一个较专业的模型（如bigram 标注器）不能为给定内容分配标记时，我们回退到一个较一般的模型（如unigram 标注器）
    8. 词性标注是NLP 中一个重要的早期的序列分类任务：利用局部上下文语境中的词和标记对序列中任意一点的分类决策。
    9. 字典用来映射任意类型之间的信息，如字符串和数字：freq['cat']=12。
        我们使用大括号来创建字典：pos = {}，pos = {'furiously': 'adv', 'ideas': 'n', 'colorless':'adj'}。
    10. N-gram 标注器可以定义较大数值的n，但是当n 大于3 时，我们常常会面临数据稀疏问题；
        即使使用大量的训练数据，我们看到的也只是可能的上下文的一小部分。
    11. 基于转换的标注学习一系列“改变标记s 为标记t 在上下文c 中”形式的修复规则，每个规则会修复错误，也可能引入（较小的）错误。