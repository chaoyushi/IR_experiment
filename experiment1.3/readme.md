# Experiment-1.3

# IR Evaluation
<br>

### 一、实验要求


#### 1.输入输出要求：

##### 在Homework1.2的基础上实现IR Evaluation

• **指标评价方法**：

&emsp; (1) Mean Average Precision (MAP)

&emsp; (2) Mean Reciprocal Rank (MRR)

&emsp; (3) Normalized Discounted Cumulative Gain (NDCG)

• **Input**： a query (like Ron Weasley birthday)

• **Output**: Return the top K (e.g.,K = 10) relevant tweets.

• **Query**：支持and, or ,not；查询优化可以选做；

#### 2.Use SMART notation: lnc.ltc

• **Document**: logarithmic tf (l as first character), no idf and cosine
normalization

• **Query**: logarithmic tf (l in leftmost column), idf (t in second column), no
normalization

#### 3.改进Inverted index

• 在Dictionary中存储每个term的DF

• 在posting list中存储term在每个doc中的TF with pairs (docID, tf)

#### 4.选做

• 支持所有的SMART Notations。

### 二、实验步骤

#### 1.创建新的倒排索引记录表(二元组)

&emsp;(1)首先，

&emsp;(2)计算df和idf，即文档频率以及逆文档频率。其中idf=log(N/df)

&emsp;(3)将原来的postings以及现在的postings_for_topk，以及document(存放逆文档频率的字典)保存成文件，使用numpy中的save函数保存为npy文件。

修正后的get_postings代码如下：
```python

```

**注意**：postings，postings_for_topk,document都是词典 是在全局声明的。

#### 2.Use SMART notation: lnc.ltc


**参考计算方式如下图：**

![image1](smartnotation.PNG)

![image2](lncltc.PNG)

&emsp;(1) 将query分成多个term，在postings_for_topk中可以返回一串二元组，每个二元组包含了docID以及tf，根据公式计算并累加求和得到针对相关句子的score。

&emsp;(2) 根据score对涉及到的句子分数从高到低进行排序，选择topk个相关的句子，如果涉及到的句子不足topk个，那么直接把所有的句子由相似度从高到低输出。

相关代码如下
```python

```
#### 3.已保存变量载入

载入保存的文件，而不是像第一个实验中动态建立，可以加快运行速度。

```python

```

#### 4.设计简单交互和运行提示


```python

```


### 三、实验结果

#### 1.界面与返回功能

![image3](run1.PNG)

#### 2.Ranked retrieval model

此处取topk=10

**Test1：**


![image4](run2.PNG)

**Test2**

![image5](run3.PNG)

### 四、实验改进与不足

##### (1) 实验中将倒排索引记录表存储在文件中，方便了每次直接从文件中读取变量。

##### (2)对于tweet数据集只保留了text信息，同时，对每一行文本使用集合set处理，同时保留了词频的信息。
