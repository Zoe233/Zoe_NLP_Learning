第2章 获得文本语料和词汇资源
    2.1 获取文本语料库
        古藤保语料库
        网络和聊天语料库
        布朗语料库
        就职演说语料库
        标注文本语料库
        在其他语言的语料库
        文本语料库的结构
        载入你自己的语料库
    2.2 条件概率分布
        条件和事件
        按文体计数词汇
        绘制分布图和分布表
        使用双连词生成随机文本
    2.3 更多关于Python：代码重用
    2.4 词典资源
        词汇列表语料库
        发音的词典
        比较词表
    2.5 WordNet
        意义和同义词
        WordNet的层次结构
        更多的词汇关系
        语义相似度
    2.6 小结



在自然语言处理的实际项目中，通常要使用大量的语言数据或者语料库。
本章的目的是要回答下列问题：
1. 什么是有用的文本语料和词汇资源，我们如何使用Python 获取它们？
2. 哪些Python 结构最适合这项工作？
3. 编写Python 代码时我们如何避免重复的工作？


一个文本语料库是一大段文本。
许多语料库的设计都要考虑一个或多个文体间谨慎的平衡。

我们曾在第1 章研究过一些小的文本集合，例如美国总统就职演说。
这种特殊的语料库实际上包含了几十个单独的文本——每个人一个演讲 ——
但为了处理方便，我们把它们头尾连接起来当做一个文本对待。

>>> from nltl.book import *

2.1 获取文本语料库

    通过nltk.download()下载配套的数据，本文文本语料库包括以下：
        Gutenberg 古腾堡语料库
            NLTK 包含古腾堡项目（Project Gutenberg）电子文本档案的经过挑选的一小部分文本。
            该项目大约有25,000（现在是36,000 了）本免费电子图书。
            >>> import nltk
            >>> nltk.corpus.gutenberg.fileids()
            ['austen-emma.txt', 'austen-persuasion.txt',
            'austen-sense.txt', 'bible-kjv.txt',
            ...'shakespeare-macbeth.txt', 'whitman-leaves.txt']  # 列表中的内容都是 文件标识符

            >>> emma = nltk.corpus.gutenberg.words('austen-emma.txt')
            >>> len(emma)
            192427
            >>> type(emma)
            <class 'nltk.corpus.reader.util.StreamBackedCorpusView'>

            # 另一种导入方式
            >>> from nltk.corpus import gutenberg
            >>> gutenberg.fileids()
            ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
            >>> emma = gutenberg.words('austen-emma.txt')

            # 循环遍历前面列出的gutenberg文件标识符链表相对应的fileid，然后计算统计每个文本的标识符数量。
            >>> len(gutenberg.raw('austen-emma.txt'))
            887071
            >>> len(gutenberg.words('austen-emma.txt'))
            192427
            >>> len(gutenberg.sents('austen-emma.txt'))
            7752
            >>> len(set([w.lower() for w in gutenberg.words(fileid)]))
            7344



            >>> for fileid in gutenberg.fileids():
                    num_chars=len(gutenberg.raw(fileid))  ###统计字符数
                    num_words=len(gutenberg.words(fileid))  ##统计单词书
                    num_sent=len(gutenberg.sents(fileid))  ###统计句子数
                    num_vocab=len(set([w.lower() for w in gutenberg.words(fileid)]))  ###唯一化单词
                    print(int(num_chars/num_words),int(num_words/num_sent),int(num_words/num_vocab),fileid)

            这个结果显示了每个文本的3个统计量：平局词长，平均句子长度和文本中每个词出现的平均次数(词汇多样性得分）。

            平均词长似乎是英语的一个一般属性，因为它的值总是4。（事实上，平均词长是3 而不是4，因为num_chars 变量计数了空白字符。）
            相比之下，平均句子长度和词汇多样性看上去是作者个人的特点。

            raw()函数给我们没有进行过任何语言学处理的文件的内容。
            例如：len(gutenberg.raw('blake-poems.txt')告诉我们文本中出现的词汇个数，包括词之间的空格。
            sents()函数把文本划分成句子，其中每一个句子是一个词链表。
                >>> macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
                >>> macbeth_sentences
                [['[', 'The', 'Tragedie', 'of', 'Macbeth', 'by', 'William', 'Shakespeare', '1603', ']'], ['Actus', 'Primus', '.'], ...]
                >>> macbeth_sentences[1037]
                ['Good', 'night', ',', 'and', 'better', 'health', 'Attend', 'his', 'Maiesty']
                >>> longest_len = max([len(s) for s in macbeth_sentences])
                >>> longest_len
                158
                >>> [ s for s in macbeth_sentences if len(s) == longest_len]
                [['Doubtfull', 'it',....]]


            除了words(),raw()和sents()以外，大多数NLTK语料库阅读器还包括了多种访问方法。
            一些语料库提供更加丰富的语言学内容，例如：
                词性标注，对话标记，句法树等。



        网络和聊天文本
            NLTK 的网络文本小集合的内容包括Firefox 交流论坛，
            在纽约无意听到的对话，
            《加勒比海盗》的电影剧本，
            个人广告和葡萄酒的评论等。
            >>> from nltk.corpus import webtext
            >>> webtext.fileids()
            ['firefox.txt', 'grail.txt', 'overheard.txt', 'pirates.txt', 'singles.txt', 'wine.txt']

            >>> from nltk.corpus import nps_chat
            >>> chatroom = nps_chat.posts('10-19-20s_706posts.xml')
            >>> chatroom[123]
            ['i', 'do', "n't", 'want', 'hot', 'pics', 'of', 'a', 'female', ',', 'I', 'can', 'look', 'in', 'a', 'mirror', '.']

        布朗语料库
            布朗语料库是第一个百万词级的英语电子语料库的，由布朗大学于1961 年创建。
            这个语料库包含500 个不同来源的文本，按照文体分类，如：新闻、社论等。

            http://icame.uib.no/brown/bcm-los.html

            >>> from nltk.corpus import brown
            >>> brown.categories()
            ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']


        路透社语料库
            路透社语料库包含10,788 个新闻文档，共计130 万字。
            这些文档分成90 个主题，按照“训练”和“测试”分为两组。he
            这样分割是为了训练和测试算法的，这种算法自动检测文档的主题。
            from nltk.corpus import reuters
            reuters.fileids()
            reuters.categories()

        就职演说语料库
            就职演说语料库，实际上是55 个文本的集合，每个文本都是一个总统的演说。
            这个集合的一个有趣特性是它的时间维度。
            from nltk.corpus import inaugural
            inaugural.fileids()

        标注文本语料库
            包含语言学标注，
            有词性标注、命名实体、句法结构、语义角色等。
            NLTK 中提供了很方便的方式来访问这些语料库中的几个，还有一个包含语料库和语料样本的数据包。

            http://www.nltk.org/data

        其他语言的语料库
            NLTK包含多国语言语料库。
            某些情况下你在使用这些语料库之前需要学习如何在Python中处理字符编码。


2.1.3 文本语料库的结构
    语料库结构。
    最简单的一种没有任何结构，仅仅是一个文本集合。
    通常，文本会按照其可能对应的文体、来源、作者、语言等分类。
    有时，这些类别会重叠，尤其是在按主题分类的情况下，因为一个文本可能与多个主题相关。
    偶尔的，文本集有一个时间结构，新闻集合是最常见的例子。

    isolated, categorized, overlapping, temporal

    NLTK语料库阅读器支持高效的访问大量语料库，并且能用于处理新的语料库。

    表2-3 列出了语料库阅读器提供的函数。
    fileids()
    fileids([categories])
    categories()
    categories([fileids])
    raw()
    raw(fileids=[f1,f2,f3])
    raw(categories=[c1,c2])
    words()
    words(fileids=[f1,f2,f3])
    words(categories=[c1,c2])
    sents()
    sents(fileids=[f1,f2,f3])
    sents(categories=[c1,c2])
    abspath(fileid)   指定文件在磁盘上的位置
    encoding(fileid)   文件的编码
    open(fileid)    打开指定语料库文件的文件流
    root()  到本地安装的语料库根目录的路径



2.1.4 载入你自己的语料库
    如果你有自己搜集的文本文件，并且想使用前面讨论的方法访问它们，
    你可以很容易地在NLTK中的PlaintextCorpusReader帮助下载入它们。

    第一步：检查你的文件在文件系统中的位置；
    第二步：PlaintextCorpusReader初始化函数的第二个参数。

    from nltk.corpus import PlaintextCorpusReader
    corpus_root = '/users/zoe/BinFiles/txt'
    wordlists = PlaintextCorpusReader(corpus_root,'.txt') # 第二个参数支持正则表达式
    # 例如可以是['a.txt','b.txt']或者'[abc]/.*\.txt'
    wordlists.fileids()

    也可以导入本地的如 宾州树库的拷贝等，可以使用BracketParseCorpusReader访问这些语料。

    from nltk.corpus import BracketParseCorpusReader
    corpus_root='/user/zoe'
    file_pattern = r'.*/wsj_.*\.mrg'
    ptb = BracketParseCorpusReader(corpus_root,file_pattern)
    ptb.fileids()
    len(ptb.sents())




2.2 条件概率分布
2.2.1 条件和事件
    频率分布计算观察到的事件，如文本中出现的词汇。
    条件频率分布需要给每个时间关联一个条件，所以不是处理一个词序列，我们必须处理的是一个配对序列。

    每对的形式是：（条件，事件）。

2.2.2 按文体计数词汇
    FreqDist() 以一个简单的链表作为输入。
    ConditionalFreqDist() 以一个配对链表作为输入。

        >>> from nltk.corpus import brown
        >>> cfd = nltk.ConditionalFreqDist(
        ...     (genre, word)
        ...     for genre in brown.categories()
        ...     for word in brown.words(categories = genre))
        >>> cfd.conditions()
        ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
        >>> cfd['adventure']
        FreqDist({'.': 4057, ',': 3488, 'the': 3370, 'and': 1622, 'a': 1354, 'of': 1322, 'to': 1309, '``': 998, "''": 995, 'was': 914, ...})
        >>> cfd['adventure']['may']
        5

2.2.3 绘制分布图和分布表
    除了组合两个或两个以上的频率分布和更容易初始化之外，ConditionalFreqDist还为制表和绘图提供了一些有用的方法。

    plot()
    tabulate()
    两个方法中可以使用conditions=parameter来选择指定哪些条件显示。
    samples=parameter 来限制要显示的样本。


2.2.4 使用双连词生成随机文本
    我们可以使用条件频率分布创建一个双连词表（词对）
    bigrams()函数接受一个词汇链表，并建立一个连续的词对链表。

        >>> sent = ['In','the','begining','God','created','the','heaven','and','the','earth','.']
        >>> nltk.bigrams(sent)
        <generator object bigrams at 0x11f7df780>
        >>> list(nltk.bigrams(sent))
        [('In', 'the'), ('the', 'begining'), ('begining', 'God'), ('God', 'created'), ('created', 'the'), ('the', 'heaven'), ('heaven', 'and'), ('and', 'the'), ('the', 'earth'), ('earth', '.')]


    条件频率分布是一个对许多NLP任务都有用的数据结构。

    表2-4 NLTK中的条件频率分布：定义、访问和可视化一个计数的条件频率分布的常用方法和习惯用法
        cfdist = ConditionalFreqDist(pairs)
        cfdist.conditions()
        cfdist[conditionn]
        cfdist[condition][sample]
        cfdist.tabulate()
        cfdist.tabulate(samples,conditions)
        cfdist.plot()
        cfdist.plot(samples,conditions)
        cfdist1 < cfdist2

2.3 更多关于Python：代码重用
2.4 词典资源
    词典或者词典资源是一个词 和/或 短语 以及一些相关信息的集合。
    例如：词性和词意定义等相关信息。

    词典资源附属于文本，通常在文本的帮助下创建和丰富。

    例如：如果我们定义了一个文本my_text, 然后vocab = sorted(set(my_text))建立my_text的词汇表，
    同时word_freq = FreqDist(my_text) 计数文本中每个词的频率。

    vocab和word_freq都是简单的词汇资源。

    词汇索引为我们提供了有关你词语用法的信息，可能在编写词典时有用。
    词项 包括 词目（也叫 词条）以及其他附加信息，例如：词性和词意定义。

    两个不同的词拼写相同被称为同音异义词。

    两个拼写相同的词条（同音异义词）的词汇项，包括词性和注释信息。

    一种简单的词典资源是除了一个词汇列表外什么也没有。
    复杂的词典资源包括在词汇项内和跨词汇项的复杂的结构。

2.4.1 词汇列表语料库
    词库语料库： -- nltk.corpus.words.words()
        NLTK包括一些仅仅包含词汇列表的语料库。
        词汇语料库是Unix中的/usr/dict/words文件，被一些拼写检查程序使用。
        我们可以用它来寻找文本语料中不寻常的或拼写错误的词汇。

        nltk.corpus.words.words()
        # <class 'list'> 235786个element。
        ['A', 'a', 'aa', 'aal', 'aalii',...]

    停用词语料库：-- nltk.corpus.stopwords.words()
        高频词汇，如: the, to ,...
        我们有时在进一步的处理之前想要将它们从文档中过滤。
        停用词通常几乎没有什么词汇内容，而它们的出现会使区分文本变得困难。

        >>> from nltk.corpus import stopwords
        >>> stopwords.words('english')
        ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', ...]

2.4.2 发音的词典
    一个稍微丰富的词典资源是一个表格（或电子表格），在每一行中含有一个词加一些性质。
    NLTK中包括美国英语的CMU发音词典，它是为语音合成器使用而设计的。

        >>> entries = nltk.corpus.cmudict.entries()
        >>> len(entries)
        127012
        >>> for entry in entries[39943:39951]:
        ... print entry
        ...
        ('fir', ['F', 'ER1'])
        ('fire', ['F', 'AY1', 'ER0'])
        ('fire', ['F', 'AY1', 'R'])
        ('firearm', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M'])
        ('firearm', ['F', 'AY1', 'R', 'AA2', 'R', 'M'])
        ('firearms', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M', 'Z'])
        ('firearms', ['F', 'AY1', 'R', 'AA2', 'R', 'M', 'Z'])
        ('fireball', ['F', 'AY1', 'ER0', 'B', 'AO2', 'L'])

    我们可以用任何词典资源来处理文本，如：
        过滤掉具有某些词典属性的词（如名词），或者映射文本中每一个词。

2.4.3 比较词表
    NLTK中包含了所谓的斯瓦迪士核心词列表（Swadesh wordlists),几种语言中约200个常用词的列表。
    语言标识符使用ISO639双字母码。

        >>> from nltk.corpus import swadesh
        >>> swadesh.fileids()
        ['be', 'bg', 'bs', 'ca', 'cs', 'cu', 'de', 'en', 'es', 'fr', 'hr', 'it', 'la', 'mk',
        'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'uk']
        >>> swadesh.words('en')
        ['I', 'you (singular), thou', 'he', 'we', 'you (plural)', 'they', 'this', 'that',
        'here', 'there', 'who', 'what', 'where', 'when', 'how', 'not', 'all', 'many', 'some',...]

    我们可以通过entries()方法中指定一个语言链表来访问多语言中的同源词。
    更进一步，我们可以把它转换成一个简单的词典。
        >>> fr2en = swadesh.entries(['fr','en'])
        >>> fr2en
        [('je', 'I'), ('tu, vous', 'you (singular), thou'), ('il', 'he'), ('nous', 'we'), ('vous', 'you (plural)'), ('ils, elles', 'they'), ('ceci', 'this'), ('cela', 'that'), ('ici', 'here'), ('là', 'there'), ('qui', 'who'),...]

        >>> translate = dict(fr2en)
        >>> translate
        {'je': 'I', 'tu, vous': 'you (singular), thou', 'il': 'he', ...}

   通过添加其他源语言，我们可以让我们这个简单的翻译器更为有用。
   让我们使用dict()函数把 德语-英语 和 西班牙语-英语 对相互转换成一个字典，然后用这些添加的映射更新我们原来的翻译词典。
        >>> de2en = swadesh.entries(['de','en'])
        >>> es2en = swadesh.entries(['es','en'])
        >>> translate.update(dict(de2en))
        >>> translate.update(dict(es2en))
        >>> translate['Hund']
        'dog'
        >>> translate['perro']
        'dog'


2.4.4 词汇工具：Toolbox和Shoebox
    可能最流行的语言学家用来管理数据的工具是Toolbox（工具箱），以前叫做Shoebox（鞋柜），因为它用满满的档案卡片占据了语言学家的旧鞋盒。
    http://www.sil.org/computing/toolbox

    一个Toolbox文件由一个大量条目的集合组成，其中每个条目由一个或多个字段组成。
    大多数字段都是可选的或重复的，这意味着这个词汇资源不能作为一个表格或电子表格来处理。

        >>> from nltk.corpus import toolbox
        >>> toolbox.entries('rotokas.dic')
        [('kaa', [('ps', 'V'), ('pt', 'A'), ('ge', 'gag'), ('tkp', 'nek i pas'),
        ('dcsv', 'true'), ('vx', '1'), ('sc', '???'), ('dt', '29/Oct/2005'),...]

    条目包括一系列的属性-值对，如('ps','V')，表示词性是V（动词），('ge','gag')表示英文注释是'gag'。
    最后的3个配对包含一个罗托卡特语例句和它的巴布亚皮钦语及英语翻译。

    Toolbox文件松散的结构是我们在现阶段很难更好的利用它。
    XML提供了一种强有力的方式来处理这种语料库。



2.5 WordNet
    WordNet是面向语义的英语词典，类似于传统词典，但具有更丰富的结构。
    NLTK中包括英语WordNet，共有155287个词和117659个同义词集合。

2.5.1 意义和同义词
        >>> from nltk.corpus import wordnet as wn
        >>> wn.synsets('motorcar')
        [Synset('car.n.01')]  # motocar只有一个可能的含义，它被定义为car.n.01,car的第一个名词意义。

        # car.n.01 被称为synset或"同义词集"，意义相同的词（或"词条"）的结合：
        >>> list(wn.synset('car.n.01').lemma_names())
        ['car', 'auto', 'automobile', 'machine', 'motorcar']

        # 同义词集中每个词可以有多种含义。
        # 同义词集也有一些一般的定义和例句。
        >>> wn.synset('car.n.01').definition()
        'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
        >>> wn.synset('car.n.01').examples()
        ['he needs a car to get to work']

    虽然定义帮助人们了解一个同义词集的本意，同义词集中的词往往对我们的程序更有用。
    为了消除歧义，我们将这些词标注为car.n.01.automobile, car.n.01.motocar等。
    这种同义词集合词的配对叫做 词条。
    我们可以得到指定同义词集的所有词条，查找特定的词条，得到一个词条对应的同义词集，也可以得到一个词条的"名字"。

        >>> from nltk.corpus import wordnet as wn
        >>> wn.synsets('motorcar')
        [Synset('car.n.01')]
        >>> wn.synset('car.n.01').lemmas()
        [Lemma('car.n.01.car'), Lemma('car.n.01.auto'), Lemma('car.n.01.automobile'), Lemma('car.n.01.machine'), Lemma('car.n.01.motorcar')]
        >>> wn.lemma('car.n.01.automobile')
        Lemma('car.n.01.automobile')
        >>> wn.lemma('car.n.01.automobile').synset()
        Synset('car.n.01')
        >>> wn.lemma('car.n.01.automobile').name()
        'automobile'

    与词automobile和motocar这些意义明确的只有一个同义词集的词不同那个，词car是含糊的，有5个同义词集：
        >>> wn.synsets('car')
        [Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'), Synset('cable_car.n.01')]
        >>> for synset in wn.synsets('car'):
        ...     print(synset.lemma_names())
        ...
        ['car', 'auto', 'automobile', 'machine', 'motorcar']
        ['car', 'railcar', 'railway_car', 'railroad_car']
        ['car', 'gondola']
        ['car', 'elevator_car']
        ['cable_car', 'car']
    为了方便起见，我们可以用下面的方式访问所有包含词car的词条：
        >>> wn.lemmas('car')
        [Lemma('car.n.01.car'), Lemma('car.n.02.car'), Lemma('car.n.03.car'), Lemma('car.n.04.car'), Lemma('cable_car.n.01.car')]


2.5.2 WordNet的层次结构
    WordNet的同义词集对应于抽象的概念，它们并不总是有对应的英语词汇。
    这些概念在层次结构中相互联系在一起。

    一些概念也很一般，如实体、状态、事件； 这些被称为 独一无二的根同义词集。

    其他的，如：油老虎和有仓门式后背的汽车等就比较具体的多。

    P81 图2-8 WordNet概念层次片段：
        每个节点对应一个同义词集；
        边表示 上位词/下位词关系，即 上级概念和从属概念的关系。

    WordNet使在概念之间漫游变的容易。
        >>> motorcar = wn.synset('car.n.01')
        >>> types_of_motorcar = motorcar.hyponyms()
        >>> types_of_motorcar[26]
        Synset('stanley_steamer.n.01')
        >>> for synet in types_of_motorcar:
        ...     for lemma in synet.lemmas():
        ...             print(lemma.name())
        ambulance
        beach_wagon
        station_wagon
        wagon
        ...
        sedan
        saloon
        sport_utility
        sport_utility_vehicle
        S.U.V.
        SUV
        sports_car
        sport_car
        Stanley_Steamer
        stock_car
        subcompact
        subcompact_car
        touring_car
        phaeton
        tourer
        used-car
        secondhand_car

    我们也可以通过访问上位词来浏览层次结构。
    有些词有多条路径，因为它们可以归类在一个以上的分类中。

    car.n.01与entity.n.01之间有两条路径，因为wheeled_vehicle.n.01可以同时被归类为车辆和容器。
        >>> motorcar.hypernyms()
        [Synset('motor_vehicle.n.01')]
        >>> paths = motorcar.hypernym_paths()
        >>> len(paths)
        2
        >>> for sysnet in paths[0]:
        ...     print(sysnet.name())
        ...
        entity.n.01
        physical_entity.n.01
        object.n.01
        whole.n.02
        artifact.n.01
        instrumentality.n.03
        container.n.01
        wheeled_vehicle.n.01
        self-propelled_vehicle.n.01
        motor_vehicle.n.01
        car.n.01
        >>> motorcar.root_hypernyms()  # 可以用这个方式得到一个 最一般的上位（或根上位）同义词集。
        [Synset('entity.n.01')]

    nltk.app.wordnet()  # 尝试NLTK中便捷的图形化WordNet浏览器。沿着上位词与下位词之间的链接，探索WordNet的层次结构。

2.5.3 更多的词汇关系
    上位词和下位词被称为 词汇关系，因为它们是同义集之间的关系。
    这个关系定位上下为"是一个"层次。

    WordNet网络另一个重要的漫游方式是从物品到它们的部件（部分）或被它们被包含其中的东西（整体）。

    例如：
        一棵树的部分是它的树干，树冠等；
        这些都是part_meronyms()。
        一棵树的实质是包括 心材 和 边材 组成的，即substance_meroyms()。
        树木的集合形成了一个森林，即member_holonyms()。

            >>> from nltk.corpus import wordnet as wn
            >>> wn.synset('tree.n.01').part_meronyms()
            [Synset('burl.n.02'), Synset('crown.n.07'), Synset('limb.n.02'), Synset('stump.n.01'), Synset('trunk.n.01')]
            >>> wn.synset('tree.n.01').substance_meronyms()
            [Synset('heartwood.n.01'), Synset('sapwood.n.01')]
            >>> wn.synset('tree.n.01').member_holonyms()
            [Synset('forest.n.01')]

    动词之间也有关系。
    例如：
        走路的动作包括抬脚的动作，所以走路 蕴涵 着抬脚。
        一些动词有多个蕴涵。
            >>> wn.synset('walk.v.01').entailments()
            [Synset('step.v.01')]
            >>> wn.synset('eat.v.01').entailments()
            [Synset('chew.v.01'), Synset('swallow.v.01')]
            >>> wn.synset('tease.v.03').entailments()
            [Synset('arouse.v.07'), Synset('disappoint.v.01')]

    词条之间的一些词汇关系，如：反义词
            >>> wn.lemma('supply.n.02.supply').antonyms()
            [Lemma('demand.n.02.demand')]
            >>> wn.lemma('rush.v.01.rush').antonyms()
            [Lemma('linger.v.04.linger')]


    可以使用dir()查看词汇关系和同义词集上定义的其他方法。
            >>> dir(wn.synset('harmony.n.02'))
            ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__unicode__', '__weakref__', '_all_hypernyms', '_definition', '_examples', '_frame_ids', '_hypernyms', '_instance_hypernyms', '_iter_hypernym_lists', '_lemma_names', '_lemma_pointers', '_lemmas', '_lexname', '_max_depth', '_min_depth', '_name', '_needs_root', '_offset', '_pointers', '_pos', '_related', '_shortest_hypernym_paths', '_wordnet_corpus_reader',
            'also_sees', 'attributes', 'causes', 'closure', 'common_hypernyms', 'definition', 'entailments', 'examples', 'frame_ids', 'hypernym_distances', 'hypernym_paths', 'hypernyms', 'hyponyms', 'instance_hypernyms', 'instance_hyponyms', 'jcn_similarity', 'lch_similarity', 'lemma_names', 'lemmas', 'lexname', 'lin_similarity', 'lowest_common_hypernyms', 'max_depth', 'member_holonyms', 'member_meronyms', 'min_depth', 'name', 'offset', 'part_holonyms', 'part_meronyms', 'path_similarity', 'pos', 'region_domains', 'res_similarity', 'root_hypernyms', 'shortest_path_distance', 'similar_tos', 'substance_holonyms', 'substance_meronyms', 'topic_domains', 'tree', 'unicode_repr', 'usage_domains', 'verb_groups', 'wup_similarity']



2.5.4 语义相思度
    我们已经看到同义词集之间构成复杂的词汇关系网络。
    给定一个同义词集，我们可以遍历WordNet网络来查找相关含义的同义词集。
    知道哪些词是语义相关的，对索引文本集合非常有用，当搜索一个一般性的用于--例如：车辆--时就可以匹配包含具体用语--例如：豪华轿车--的文档。

    回想一下，每个同义词集都有一个或多个上位词路连接到一个根上位词，如entity.n.01。
    连接到同一个根的两个同义词集可能有一些共同的上位词。
    如果两个同义词集共用一个非常具体的上位词--在上位词层次结构中处于较低层的上位词--它们一定有密切的联系。

        >>> right = wn.synset('right_whale.n.01')
        >>> orca = wn.synset('orca.n.01')
        >>> minke = wn.synset('minke_whale.n.01')
        >>> tortoise = wn.synset('tortoise.n.01')
        >>> novel = wn.synset('novel.n.01')
        >>> right.lowest_common_hypernyms(minke)
        [Synset('baleen_whale.n.01')]
        >>> right.lowest_common_hypernyms(orca)
        [Synset('whale.n.02')]
        >>> right.lowest_common_hypernyms(tortoise)
        [Synset('vertebrate.n.01')]
        >>> right.lowest_common_hypernyms(novel)
        [Synset('entity.n.01')]

    当然，我们知道，鲸鱼是非常具体的（须鲸更是如此），脊椎动物是更一般的，而实体完全是抽象的一般的。
    我们可以通过查找每个同义词集深度量化这个一般性的概念：

        >>> wn.synset('baleen_whale.n.01').min_depth()
        14
        >>> wn.synset('whale.n.02').min_depth()
        13
        >>> wn.synset('vertebrate.n.01').min_depth()
        8
        >>> wn.synset('entity.n.01').min_depth()
        0

    WordNet同义词集的集合上定义了类似的函数能够深入的观察。
    例如：path_similarityassigns是基于上位词层次结构中相互连接的概念之间的最短路径在0-1范围的打分（两者之间没有路径就返回-1）。
    同义词集与自身比较将返回1.

    考虑以下的相似度：
        露脊鲸 与 小须鲸、逆戟鲸、乌龟以及小说。
        数字本身的意义并不大，当我们从海洋生物的语义空间转移到非生物时它是减少的。

        >>> from nltk.corpus import wordnet as wn
        >>> minke = wn.synset('minke_whale.n.01')
        >>> right = wn.synset('right_whale.n.01')
        >>> right.path_similarity(minke)
        0.25
        >>> orca = wn.synset('orca.n.01')
        >>> right.path_similarity(orca)
        0.16666666666666666
        >>> tortoise = wn.synset('tortoise.n.01')
        >>> right.path_similarity(tortoise)
        0.07692307692307693
        >>> novel = wn.synset('novel.n.01')
        >>> right.path_similarity(novel)
        0.043478260869565216

    还有一些其他相似性度量的方法。可以输入help(wn)获得更多信息。
    NLTK还包括VerbNet，一个连接到WordNet的动词的层次结构的词典。



2.6 小结
    文本语料库是一个大型结构化文本的集合。NLTK包含了许多语料库，如：布朗语料库nltk.corpus.brown
    有些文本语料库是分类的，例如通过文本或者主题；有时候语料库的分类会相互重叠。
    条件频率分布式一个频率分布的集合，每个分布都有一个不同的条件。它们可以用于通过给定内容或者文体对词的频率计数。
    WordNet是一个面向语义的英语词典，由同义词的集合--同义词集synsets--组成，并且组织成一个网络。
