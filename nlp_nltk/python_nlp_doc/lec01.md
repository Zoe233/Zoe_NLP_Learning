第一章 语言处理与Python
    1.1 语言计算：文本和单词
        Python入门
        NLTK入门
        搜索文本
        计数词汇
    1.2 近观Python：将文本当做词链表
        链表
        索引列表
        变量
        字符串
    1.3 计算语言：简单的统计
        频率分布
        细粒度的选择词
        词语搭配和双连词bigrams
        计数和其他东西
    1.4 回到Python：决策与控制
        条件
        对每个元素进行操作
        嵌套代码块
        条件循环
    1.5 自动理解自然语言
        词意消歧
        指代消解
        自动生成语言
        机器翻译
        人机对话系统
        文本的含义
        NLP的局限性


NLP包含所有用计算机对自然语言进行的操作，从最简单的通过计数词出现的频率来比较不同的写作风格，到最复杂的完全"理解"人所说的话，至少要能达到对人的话语作出有效反应的程度。

基于NLP技术的应用：
    手机和手持电脑支持输入法联想提示和手写识别；
    网络搜索引擎能搜到非结构化文本中的信息；
    机器翻译能把中文文本翻译成西班牙文；
    ...

你将学到什么？
    十分简单的程序如何就能帮你处理和分析语言数据，以及如何写这些程序。
    NLP与语言学的关键概念是如何用来描述和分析语言的。
    NLP中的数据结构和算法是怎么样的。
    语言数据时如何存储为标准格式，以及如何使用数据来评估NLP技术的性能。


表P-2. 语言处理任务与相应NLTK 模块以及功能描述
    语言处理任务       NLTK 模块                    功能描述
    获取和处理语料库    nltk.corpus                  语料库和词典的标准化接口
    字符串处理         nltk.tokenize, nltk.stem     分词，句子分解提取主干
    搭配发现           nltk.collocations            t-检验，卡方，点互信息PMI
    词性标识符         nltk.tag                     n-gram，backoff，Brill，HMM，TnT
    分类              nltk.classify, nltk.cluster  决策树，最大熵，贝叶斯，EM，k-means
    分块              nltk.chunk                   正则表达式，n-gram，命名实体
    解析              nltk.parse                   图表，基于特征，一致性，概率，依赖
    语义解释           nltk.sem, nltk.inference    λ演算，一阶逻辑，模型检验
    指标评测           nltk.metrics                  精度，召回率，协议系数
    概率与估计         nltk.probability               频率分布，平滑概率分布
    应用              nltk.app, nltk.chat           图形化的关键词排序，分析器，WordNet查看器，聊天机器人
    语言学领域的工作    nltk.toolbox                   处理SIL 工具箱格式的数据

简易性
    提供一个直观的框架和大量模块，使用户获取NLP 知识而不必陷入像标注语言数据那样繁琐的事务中。
一致性
    提供一个具有一致的接口和数据结构并且方法名称容易被猜到的统一的框架。
可扩展性
    提供一种结构，新的软件模块包括同一个任务中的不同的实现和相互冲突的方法都可以方便添加进来。
模块化
    提供可以独立使用而与工具包的其他部分无关的组件。


1.1 语言计算：文本和单词
1.1.1 Python入门
        >>> import nltk
        >>> nltk.download()
    下载NLTK图书集，使用nltk.download()浏览器可用的软件包。
    下载器上的Collections选项卡显示软件包如何被打包分组。

    选择book标记所在行，可以获取本书的例子和练习所需的全部数据。

    一旦数据被下载到你的机器，你可以使用Python解释器加载其中一些。
        >>> from nltk.book import *
        >>> text1
        <Text: Moby Dick by Herman Melville 1851>
        >>> text2
        <Text: Sense and Sensibility by Jane Austen 1811>

1.1.2 搜索文本
    除了阅读文本之外，还有很多方法可以用来研究文本内容。
    词语索引视图 显示一个指定单词的每一次出现，连同一些上下文一起显示。
        >>> text1.concordance('monstrous')
        Displaying 11 of 11 matches:
        ong the former , one was of a most monstrous size . ... This came towards us ,
        ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
        ll over with a heathenish array of monstrous clubs and spears . Some were thick
        d as you gazed , and wondered what monstrous cannibal and savage could ever hav
        that has survived the flood ; most monstrous and most mountainous ! That Himmal
        they might scout at Moby Dick as a monstrous fable , or still worse and more de
        th of Radney .'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l
        ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
        ere to enter upon those still more monstrous stories of them which are to be fo
        ght have been rummaged out of this monstrous cabinet there is no telling . But
        of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u

    # 注：
    text1是《白鲸记》
    text2是《理智与情感》
    text3是《创世纪》
    text4是《就职演说语料》
    text5是《NPS聊天语料库》 ，这个语料库未经审查

    词语索引使我们看到词的上下文。
    例如：我们看到monstrous出现的上下文，如the monstrous pictures和 the monstrous size。


    还有哪些词出现在相似的上下文中？
    我们可以通过text1.similar('monstrous')来查找到。

        >>> text1.similar('monstrous')
        true contemptible christian abundant few part mean careful puzzled
        mystifying passing curious loving wise doleful gamesome singular
        delightfully perilous fearless

        >>> text2.similar('monstrous')
        very so exceedingly heartily a as good great extremely remarkably
        sweet vast amazingly
    观察我们从不同的文本中得到的结果。
    text1，text2中monstrous的完全不同。

    函数common_contexts允许我们研究两个或两个以上的词共同的上下文，如monstrous和very.

        >>> text2.common_contexts(['monstrous','very'])
        a_pretty am_glad a_lucky is_pretty be_glad
        >>> text2.common_contexts(['monstrous','very','so'])
        am_glad is_pretty

    自动检测出现在文本中的特定的词，并显示同样上下文中出现的一些词，这只是一个方面。
    我们也可以判断词在文本中的位置：从文本开头算起它前面有多少词。
    这个位置信息可以用 离散图 表示。
    每一个竖线代表一个单词，每一行代表整个文本。
        >>> text4.dispersion_plot(['citizens','democracy','freedom','duties','America'])


1.1.3 计数词汇
    《创世纪》中使用的例子：
        >>> len(text3)
        44764
        >>> len(set(text3))
        2789

    《创世纪》有44764个词和标点符号，或者叫做"标识符"。
    一个标识符是表示一个我们想要放在一组对待的字符序列 --  如： hairy，his或者:) 的术语。
    《创世纪》中有多少不同的词？len(set(text3)).
    一个文本词汇表只是它用到的标识符的集合，因为在集合中的所有重复的元素都只算一个。

    小说中有44764个标识符，但是有值2789个不同的词汇或"词类型"。
    一个 词类型 是指一个词在一个文本中独一无二的出现形式或拼写。

    也就是说，这个词在词汇表中是唯一的。
    我们计数的2789个项目中包括标点符号，所以我们把这些叫做 唯一项目类型 而不是 词类型。

    # 对文本词汇丰富度进行测量。
        >>> len(text3)/len(set(text3))
        16.050197203298673

    # 专注于特定的词，计数一个词在文本中出现的次数，计算一个特定的词在文本中占据的百分比
        >>> text3.count('smote')
        5
        >>> 100*text4.count('a')/len(text4)
        1.4643016433938312

        def lexical_diverity(text):
            '''对文本词汇丰富度进行测量'''
            return len(text)/len(set(text))

        def percentage(count,total):
            '''计算词汇在文本中占比'''
            return 100*count/total

        >>> lexical_diversity(text3)
        16.050197203298673
        >>> lexical_diversity(text5)
        7.4200461589185629
        >>> percentage(4, 5)
        80.0
        >>> percentage(text4.count('a'), len(text4))
        1.4643016433938312


1.2 近观Python：将文本当做词链表
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


1.3 计算语言：简单的统计
    让我们重新开始探索用我们的计算资源处理大量文本的方法。
    搜索词 text.concordance('word1')
    搜索上下文 text.common_contexts(['a','very])
    汇编一个文本中的词汇  set(text)
    产生一种文体的随机文本 text.generate()

1.3.1 频率分布
    我们如何能自动识别文本中最能提现文本的主题和风格的词汇？
    找到书中使用最频繁的词-- 为每个词设置一个计数器。

    频率分布，它告诉我们在文本中的每一个词项的频率。
    一般情况下，它能计数任何观察得到的事件。

    这是一个"分布"，因为它告诉我们文本中词标识符的总数是如何分布在词项中。

        >>> fdist1 = FreqDist(text1)
        >>> fdist1
        FreqDist({',': 18713, 'the': 13721, '.': 6862, 'of': 6536, 'and': 6024, 'a': 4569, 'to': 4542, ';': 4072, 'in': 3916, 'that': 2982, ...})
        >>> type(fdist1)
        <class 'nltk.probability.FreqDist'>
        >>> vocabulary1 = fdist1.keys()
        >>> fdist1['whale']
        906

    这个例子中有什么词有助于我们把握这个文本的主题和风格呢？
    只有一个词-- whale,稍微有些信息量！
    它出现了超过900次。其余的词没有告诉我们关于文本的信息，它们只是"管道"英语。
    这些词在文本中占多少比例？
    我们可以产生一个这些词汇的累积频率图。
        >>> fdist1.plot(50,cumulative=True) # 生成前50个高频词的累积频数图

    这50个词占了书的将近一半！
    如果高频词对我们没有帮助，那些只出现了一次的词（所谓的hapaxes）又如何呢？
        >>> fdist1.hapaxes()

    返回只出现了一次的词组成的列表。
    低频次太多，没看到上下文我们很可能有一半的hapaxes猜不出它们的意义！

    既然高频词和低频词都没有帮助，我们需要尝试其他的方法。


1.3.2 细粒度的选择词
    文本中的长词，也许它们有更多的特征和信息量。
    为此我们采用集合论的一些符号。
    a. {w|w ∈ V & P(W)}  # 此集合中所有w都满足w是集合V（词汇表）的一个元素且w有性质P。
    b. [w for w in V if P(W)]  # 列表生成式

        >>> V = set(text1)
        >>> long_words = [w for w in V if len(w) > 15]
        >>> sorted(long_words)
        ['CIRCUMNAVIGATION', 'Physiognomically', 'apprehensiveness', 'cannibalistically', 'characteristically', 'circumnavigating', 'circumnavigation', 'circumnavigations', 'comprehensiveness', 'hermaphroditical', 'indiscriminately', 'indispensableness', 'irresistibleness', 'physiognomically', 'preternaturalness', 'responsibilities', 'simultaneousness', 'subterraneousness', 'supernaturalness', 'superstitiousness', 'uncomfortableness', 'uncompromisedness', 'undiscriminating', 'uninterpenetratingly']


    text4中长词反映国家主题--
        constituionally（按宪法规定的，本质的）
        transcontinental (横贯大陆的）

    text5中长词是非正规表达式
        boooooolyyyyy
        yuuuuuuummmmmmmmm

    我们是否已经成功的自动提取文本中的特征词汇呢？
    好的，这些很长的词通常是hapaxes(唯一的），也许找长词出现的频率会更好。

    这样看起来更有前途，因为这样忽略了短高频词（如the)和长低频词（如antiphiosophists）。

    # 以下是聊天语料库中所有长度超过7个字符，出现次数超过7词的词：
        >>> fdist5 = FreqDist(text5)
        >>> sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])
        ['#14-19teens', '#talkcity_adults', '((((((((((', '........', 'Question', 'actually', 'anything', 'computer', 'cute.-ass', 'everyone', 'football', 'innocent', 'listening', 'remember', 'seriously', 'something', 'together', 'tomorrow', 'watching']

    最后，我们已成功地自动识别出与文本内容相关的高频词。



1.3.3 词语搭配和双连词bigrams
    一个搭配 是异乎寻常的经常在一起出现的词序列。
    red wine是一个搭配而 the wine不是。

    一个搭配 的特点是其中的词不能被类似的词置换。
    例如： maroom wine（粟子酒）听起来就很奇怪。

    要获取搭配，我们先从提取文本词汇中的词对也就是双连词开始。
    使用bigrams().

        >>> e = bigrams(['more','is','said','than','done'])
        >>> list(e)
        [('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]

    在这里我们看到词对than-done是一个双连词。
    现在，除非我们更加注重包含不常见词的情况，搭配基本上就是频繁的双连词。
    特别的，我们希望找到比我们基于单个词的频率预期得到的更频繁出现的双连词。

    collocations()函数为我们做这些。
        >>> text4.collocations()
        United States; fellow citizens; four years; years ago; Federal
        Government; General Government; American people; Vice President; Old
        World; Almighty God; Fellow citizens; Chief Magistrate; Chief Justice;
        God bless; every citizen; Indian tribes; public debt; one another;
        foreign nations; political parties

        >>> text8.collocations()
        would like; medium build; social drinker; quiet nights; non smoker;
        long term; age open; Would like; easy going; financially secure; fun
        times; similar interests; Age open; weekends away; poss rship; well
        presented; never married; single mum; permanent relationship; slim
        build

    文本中出现的搭配很能提现文本的风格。
    为了找到red wine这个搭配，我们将需要处理更大的文本。



1.3.4计数和其他东西
    计数词汇是有用的，我们也可以计数其他东西。

    例如，我们可以查看文本中 词长的分布 ，通过创造一长串数字的链表的FreqDist，其中每个数字是文本中对应词的长度。
        >>> [len(w) for w in text1]
        [1,4,4,2,6,8,....]
        >>> fdist = FreqDist([len(w) for w in text1])
        >>> fdist
        FreqDist({3: 50223, 1: 47933, 4: 42345, 2: 38513, 5: 26597, 6: 17111, 7: 14399, 8: 9966, 9: 6428, 10: 3528, ...})
        >>> fdist.keys()
        dict_keys([1, 4, 2, 6, 8, 9, 11, 5, 7, 3, 10, 12, 13, 14, 16, 15, 17, 18, 20])

        >>> fdist.items()
        dict_items([(1, 47933), (4, 42345), (2, 38513), (6, 17111), (8, 9966), (9, 6428), (11, 1873), (5, 26597), (7, 14399), (3, 50223), (10, 3528), (12, 1053), (13, 567), (14, 177), (16, 22), (15, 70), (17, 12), (18, 1), (20, 1)])
        >>> fdist.max()
        3
        >>> fdist[3]
        50223
        >>> fdist.freq(3)
        0.19255882431878046
    由此我们看到，最频繁的词长度是3，长度为3 的词有50,000 多个（约占书中全部词汇的20％）。
    虽然我们不会在这里追究它，关于词长的进一步分析可能帮助我们了解作者、文体或语言之间的差异。


    表1-2. NLTK 频率分布类中定义的函数
        >>> fdist = FreqDist(text1)  # 创建包含给定样本的频率分布
        >>> fdist['the']    # 指定词汇的在样本中出现的频数
        13721
        >>> fdist.freq('the') # 计算给定词在样本中的频率
        0.052607363727335814
        >>> fdist.N()   # 计算给定样本的element总数
        260819
        >>> fdist.keys()   # 以频率递减顺序排序的<class 'dict_keys'>
        >>> for sample in fdist: print(sample)  # 以频率递减的顺序遍历样本
        >>> fdist.max()     # 数值最大的样本
        ','
        >>> fdist.tabulate()   # 绘制频率分布表
        >>> fdist.tabulate(10)
            ,   the     .    of   and     a    to     ;    in  that
        18713 13721  6862  6536  6024  4569  4542  4072  3916  2982

        >>> fdist.plot()   # 绘制频率分布图
        >>> fdist.plot(cumulative=True)  # 绘制累积频率分布图
        >>> fdist1 < fdist2    # 测试样本在fdist1中出现的频率是否小于fdist2, return True or False
        >>> fdist[',']<fdist['.']
        False
        >>> fdist[',']>fdist['.']
        True


1.4 回到Python：决策与控制
1.4.1 条件

    >>> from nltk.book import *
    >>> sent7  # 华尔街日报的第一句话
    ['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
    >>> [w for w in sent7 if len(w) < 4]
    [',', '61', 'old', ',', 'the', 'as', 'a', '29', '.']
    >>> [w for w in sent7 if len(w) <= 4]
    [',', '61', 'old', ',', 'will', 'join', 'the', 'as', 'a', 'Nov.', '29', '.']
    >>> [w for w in sent7 if len(w) == 4]
    ['will', 'join', 'Nov.']
    >>> [w for w in sent7 if len(w) != 4]
    ['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', '29', '.']
    >>> sorted([w for w in set(text1) if w.endswith('ableness')])
    ['comfortableness', 'honourableness', 'immutableness', 'indispensableness', 'indomitableness', 'intolerableness', 'palpableness', 'reasonableness', 'uncomfortableness']
    >>> sorted([term for term in set(text4) if 'gnt' in term])
    ['Sovereignty', 'sovereignties', 'sovereignty']
    >>> sorted([item for item in set(text6) if item.istitle()])
    ['A', 'Aaaaaaaaah', 'Aaaaaaaah', 'Aaaaaah', 'Aaaah', 'Aaaaugh', ...]
    >>> sorted([item for item in set(text7) if item.isdigit()])
    ['0', '1', '10', '100', '101', '102', '103', '105', '106', ...]

    >>> sorted([w for w in set(text7) if '-' in w and 'index' in w])
    ['Stock-index', 'index-arbitrage', 'index-fund', 'index-options', 'index-related', 'stock-index']
    >>> sorted([wd for wd in set(text3) if wd.istitle() and len(wd) > 10])
    ['Abelmizraim', 'Allonbachuth', 'Beerlahairoi', 'Canaanitish', 'Chedorlaomer', 'Girgashites', 'Hazarmaveth', 'Hazezontamar', 'Ishmeelites', 'Jegarsahadutha', 'Jehovahjireh', 'Kirjatharba', 'Melchizedek', 'Mesopotamia', 'Peradventure', 'Philistines', 'Zaphnathpaaneah']



1.4.2 对每个元素进行操作
1.4.3 嵌套代码块
1.4.4 条件循环

1.5 自动理解自然语言
1.5.1 词意消歧
    在词意消歧中，我们要算出特定上下文中的词被赋予的是哪个意思。

    换句话说，自动消除歧义需要使用上下文，利用相邻词汇有相近含义这样一个简单的事实。

    在另一个有关上下文影响的例子是词by，它有几种含义，例如：
        the book by Che sterton（施事——Chesterton 是书的作者）；
        the cup by the stove（位置格——炉子在杯子旁边）；
        submit by Friday（时间——星期五前提交）。
    观察(3)中斜体字的含义有助于我们解释by 的含义。
    (3) a. The lost children were found by the searchers (施事)
        b. The lost children were found by the mountain (位置格)
        c. The lost children were found by the afternoon (时间)



1.5.2 指代消解
    一种更深刻的语言理解是解决“谁对谁做了什么”，即检测主语和动词的宾语。

    在句子the thieves stole the paintings 中，很容易分辨出谁做了偷窃的行为。
    考虑(4)中句子的三种可能，尝试确定是什么被出售、被抓和被发现（其中一种情况是有歧义的）。
        (4) a. The thieves stole the paintings. They were subsequently sold.
            b. The thieves stole the paintings. They were subsequently caught.
            c. The thieves stole the paintings. They were subsequently found.
    要回答这个问题涉及到寻找代词they 的先行词thieves 或者paintings。
    处理这个问题的计算技术包括指代消解（anaphora resolution）——
        确定代词或名词短语指的是什么——
        和
        语义角色标注（semantic role labeling）——
        确定名词短语如何与动词相关联（如施事，受事，工具等）。


1.5.3 自动生成语言
    如果我们能够解决自动语言理解等问题，我们将能够继续那些包含自动生成语言的任务，如自动问答和机器翻译。

    在自动问答中，一台机器要能够回答用户关于特定文本集的问题：
        (5) a. Text: ... The thieves stole the paintings. They were subsequently sold. ...
            b. Human : Who or what was sold?
            c. Machine: The paintings.
    机器的回答表明，它已经正确的计算出they 是指paintings，而不是thieves。

    所有这些例子中，弄清楚词的含义、动作的主语以及代词的先行词是理解句子含义的步骤，也是我们希望语言理解系统能够做到的事情。


1.5.4 机器翻译
    长久以来，机器翻译（MT）都是语言理解的圣杯，人们希望能找到从根本上提供高品质的符合语言习惯的任意两种语言之间的翻译。
    其历史可以追溯到冷战初期，当时自动翻译的许诺带来大量的政府赞助，它也是NLP 本身的起源。


1.5.5 人机对话系统
    在人工智能的历史，主要的智能测试是一个语言学测试，叫做图灵测试：一个响应用户文本输入的对话系统能否表现的自然到我们无法区分它是人工生成的响应？
    相比之下，今天的商业对话系统能力是非常有限的，但在较小的给定领域仍然有些作用。

    对话系统给我们一个机会来说说一般认为的NLP 流程。
    图1-5 显示了一个简单的对话系统架构。沿图的顶部从左向右是一些语言理解组件的“管道”。
    这些组件从语音输入经过文法分析到某种意义的重现。
    图的中间，从右向左是这些组件的逆向流程，将概念转换为语音。
    这些组件构成了系统的动态方面。在图的底部是一些有代表性的静态信息：语言相关的数据仓库，这些用于处理的组件在其上运作。

    >>> import nltk
    >>> nltk.chat.chatbots()
    # 原始的对话系统的例子。

    图1-5：简单的语音对话系统的流程框架。
        分析语音输入 --> 识别单词 --> 文法分析 --> 在上下文中解释 --> 应用相关的具体操作 --> 响应规划 --> 实现文法结构 --> 适当的词形变化 --> 语音输出
        Phonology       Morphology  Syntax    Semantics        Reasoning            Semantics    Syntax         Morphology       Phonology



1.5.6 文本的含义
    近年来，一个叫做文本含义识别(Recognizing Textual Entailment 简称RTE)的公开的“共享任务”使语言理解所面临的挑战成为关注焦点.



1.5.7 NLP的局限性
























