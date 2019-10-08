# Experiment-1

# Inverted index and BooleanRetrieval Model
<br>

### 一、实验要求

#### 1.构建invertedindex:

构建倒排索引是基于tweets数据集的。

我们首先观察该数据集，如下：

![image1](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/1.PNG)

&emsp;我们可以看到，tweets数据集中包含了许多信息，包括username，clusterNo，text，timestr,tweetid，errorcode，textcleaned，relevance。其中，我们需要提取的信息是text部分。

<br>
&emsp; –1. 实现Boolean Retrieval Model，使用TREC 2014 test
topics进行测试；

&emsp; –2. Boolean Retrieval Model：
<br>

#### 2.输入输出要求：

• **Input**： a query (like Ron and Weasley)

• **Output**: print the qualified tweets.

• **Query**：支持and, or ,not；查询优化可以选做；

#### 3.注意：

•**预处理**：对于tweets与queries使用相同的预处理；

### 二、实验步骤

#### 1.文本预处理

首先，我们定义一个函数，对每一行文本进行处理：对该行文本进行处理。

1.首先，将文本的每一个字符转换为小写，然后使用index函数，查找字符串中“text”字符和“timestr”字符的位置，从而定位出text文本在字符串中的位置。保存到新的字符串中。

2.其次，我们使用文本处理工具TextBlob对文本进行处理：
&emsp;(1) 使用singularize函数对词汇进行单数化（复数转化为单数）

&emsp;(2) 遍历每一个词汇，将动词转化为动词原形。

&emsp;(3) 返回词汇列表。

#### 2.创建倒排索引记录表

&emsp;(1) 首先，逐行读取tweet文件，遍历每一行，并将每一行的每一个词汇存入一个集合中，即实现集合中词汇的去重；遍历集合中的每一个term，如果该term在全局字典中存在对应的keys，则对以该terms为keys的元组存在时，那么append该term存在的行号，(此处以行号作为每一个term的标识)，如果不存在，则创建对应的元组。

&emsp;(2)记录文本对应的行数。并输出提示信息。


**注意**：postings词典 是在全局声明的。

#### 3.创建基本的查询函数

&emsp;(1) 对两个词项的and操作：

&emsp;1)传入参数有两个，即要执行and操作的两个term。创建一个空的列表ans。

&emsp;2)如果两个term都不在字典中，那么直接放回ans。

&emsp;3)否则，获取以term1和term2为键值的列表，分别设定两个迭代器对列表进行遍历，如下：
```python

        len1=len(postings[term1])
        len2=len(postings[term2])
        p1=0
        p2=0
        while p1<len1 and p2<len2:
            if postings[term1][p1] == postings[term2][p2]:
                ans.append(postings[term1][p1])
                p1+= 1
                p2+= 1
            elif postings[term1][p1] < postings[term2][p2]:
                p1 += 1
            else:
                p2+= 1


```
最终返回记录得到的ans列表。

&emsp;(2) 对两个词项的or操作：
```python
 ans=[]
    if term1 not in postings and term2 not in postings:
        ans=[]#都不在为空
    elif term1 in postings and term2 not in postings:
        ans=postings[term1]
    elif term2 in postings and term1 not in postings:
        ans=postings[term2]
    else:
        ans=postings[term1]
        for item in postings[term2]:
            if item not in ans:
                ans.append(item)
```

&emsp;(3) 对两个词项的not操作：

&emsp;1)首先，创建一个全局列表，该列表是一个和文本数目相同的从1开始的列表。

&emsp;2)然后，设计单独的函数，实现对单个词项的not操作，返回一个列表。如下：

```python
    ans=[]
    if term not in postings:
        return ans
    # print("not")
    # print(all)
    # print("not")
    for item in all:
        if item not in postings[term]:
            ans.append(item)
    return ans
```

&emsp;3)在单个此项取not的基础上，设计对两个词项之间not操作的函数。因为我们的not事实上指的是 and not 的缩写，所以，此时，对应的处理和and操作类似，即相当于两个列表的合并。代码如下所示：

```python
    if term1 not in postings or term2 not in postings:
        return ans
    else:
        ans1=postings[term1]
        ans2=signal_not(term2)
        len1=len(ans1)
        len2=len(ans2)
        i=0
        j=0
        while i<len1 and j < len2:
            if ans1[i] == ans2[j]:
                ans.append(ans1[i])
                i=i+1
                j=j+1
            elif ans1[i]<ans2[j]:
                i=i+1
            else:
                j=j+1
    return ans
```

#### 4.输入文本处理及main函数书写

&emsp;1) 对于输入文本，对term的处理方式同对文本的处理方式。

&emsp;2) 对于词项的数目小于等于3时，做如下处理：

```python
if terms==[]:
        print("your input is empty")
    if len(terms)==1:

        print(postings[terms[0]])
    elif len(terms)==2:
        print("sorry ,your input format is wrong!")
    #简单查询
    elif len(terms)==3:
        search_three_tuple(terms)

#词项数目等于3
def search_three_tuple(terms):
    global postings

    if len(terms)==3:
        answer = []
        # A and B
        if terms[1] == "and":
            answer = merge_and2(terms[0], terms[2])
            print(answer)
        # A or B
        elif terms[1] == "or":
            answer = merge_or2(terms[0], terms[2])
            print(answer)
        # A and (not) B   为方便处理，此处省略  and
        elif terms[1] == "not":
            answer = twice_not(terms[0], terms[2])
            print(answer)
        # 输入的三个词格式不对
        else:
            print("sorry ,your input format is wrong!")

```

#### 5.改进

如果输入的文本包含了括号或者输入的查询大于3，那么仅仅通过以上函数就无法实现了，因此，我对输入的查询大于3的情况作了如下改进。

&emsp;1) 此时，对输入文本的处理略有不同，应该保留括号等元素(之前的处理直接将特殊符号去除了)此处使用nltk中的分词工具进行处理，其他地方基本一致。

&emsp;2) 如果查询中不包含括号，那么对应的操作就应该是基于两个term操作的基础之上的，对两个term进行and or not操作得到一个列表，对得到的列表和第三个至多个元素进行操作，即为列表的交并补集操作。如下：
```python
elif '(' not in split_list and ')' not in split_list:
        for i in range(len(split_list)):
            #print(ans)

            if i%2!=0:
                continue
            #print(i)
            if i == 0:
                if split_list[i + 1] == 'and':
                    ans = merge_and2(split_list[i], split_list[i + 2])
                elif split_list[i + 1] == "or":
                    ans = merge_or2(split_list[i], split_list[i + 2])
                elif split_list[i + 1] == "not":
                    ans = twice_not(split_list[i], split_list[i + 2])
                #i+=2
            elif i>=len(split_list)-1:
                break
            else:
                if split_list[i + 2] not in postings:
                    ans=ans
                else:
                    if split_list[i + 1] == 'and':
                        #and 对应列表的交集
                        #print(postings[split_list[i + 2]])

                        #print(set(postings[split_list[i + 2]])&set(ans))
                        ans = sorted(list(set(ans) & set(postings[split_list[i + 2]])))
                    elif split_list[i + 1] == "or":
                        #or对应列表相加
                        ans = ans + postings[split_list[i + 2]]
                        ans = sorted(list(set(ans)))
                    elif split_list[i + 1] == "not":
                        temp=[]
                        for i in all:
                            if i not in postings[split_list[i + 2]]:
                                temp.append(i)
                        ans = sorted(list(set(ans) & set(temp)))
                #i+=2

        return ans
```
&emsp;3) 如果输入查询中包含了括号，暂时不做处理。（还未实现）

### 三、实验结果

#### 1.单词项测试

![image2](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/test1.PNG)

#### 2.两个词项执行and操作

![image3](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/test2.PNG)

#### 3.两个词项执行or操作

![image4](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/test3.PNG)

#### 4.两个词项执行not操作(and not)

![image5](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/test4.PNG)

#### 5.多个词项执行and，or，not操作

![image6](https://github.com/Eternal-Sun625/IR_experiment/blob/master/experiment1/test5.PNG)

### 四、实验改进与不足

##### (1) 实验中没有将倒排索引记录表存储在文件中，每次动态建立倒排索引记录表；

##### (2) 实验中已经实现对于多个词项的布尔查询，但是还未实现对于含括号时的文本处理与检索分析；

##### (3)对于tweet数据集只保留了text信息，同时，对每一行文本使用集合set处理，丢失了词频的信息。