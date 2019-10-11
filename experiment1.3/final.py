import re
import nltk
import numpy as np
import string
import sys
import math
import json
from textblob import TextBlob
from textblob import Word
from collections import defaultdict

#index  最先出现的位置
#rindex  str.rindex(str, beg=0 end=len(string))返回子字符串 str
# 在字符串中最后出现的位置，如果没有匹配的字符串会报异常，
# 你可以指定可选参数[beg:end]设置查找的区间。


postings = defaultdict(dict)
postings_for_topk = defaultdict(dict)
all=[]
global_dict=[]
topK=10    #前K个
document={}
global tweet_id



def preprocess(document):
    global postings
    document=document.lower()
    start=document.index("text")+6
    end=document.index("timestr")-3
    document=document[start:end]
    terms = TextBlob(document).words.singularize()

    result = []
    for word in terms:
        expected_str = Word(word)
        expected_str = expected_str.lemmatize("v")
        # expected_str = expected_str.lemmatize("n")
        # expected_str = expected_str.lemmatize("a")
        result.append(expected_str)
    #print("预处理完成")
    return result

def get_postings():
    global postings
    global all
    global postings
    f = open("tweets1.txt", 'r')
    lines=f.readlines()
    i=1
    for line in lines:
        all.append(i)
        line=preprocess(line)
        #print(line)
        #计算TF
        unique_terms=set(line)
        t1=0
        for every_term in unique_terms:
            #统计TF
            term_frequency =1+math.log10(line.count(every_term))
            t1+=term_frequency*term_frequency
            #print(term_frequency)
            if every_term in postings.keys():
                postings[every_term].append(i)
            else:
                postings[every_term]=[i]

        ans1=math.sqrt(t1)
        for every_term in unique_terms:
            term_frequency = 1 + math.log10(line.count(every_term))
            #重新计算一次，保存到二元组中的数值为归一化之后的数值
            if every_term in postings_for_topk.keys():
                postings_for_topk[every_term].append((i, term_frequency/ans1))
            else:
                postings_for_topk[every_term] = [(i, term_frequency/ans1)]

        #计算文档频率
        for every_term in unique_terms:
            if every_term in document.keys():
                document[every_term]+=1
            else:
                document[every_term]=1
        i=i+1

    number=i-1
    #计算idf
    for every_key in document.keys():
        document[every_key]=math.log10(float(number)/document[every_key])

    np.save("idf.npy",document)
    np.save("postings.npy",postings)
    np.save("postings_for_topk.npy",postings_for_topk)
    # print(all)
    print("预处理完成")
    print("文本数量共计：",number,"项")
    print("倒排索引记录表建立完成")

def merge_and2(term1,term2):
    global postings
    ans=[]
    if term1 not in postings and term2 not in postings:
        return ans
    else:
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
        return ans

def merge_or2(term1,term2):
    global postings
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
    return ans


def signal_not(term):
    global postings
    #print("not")
    global all

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


def twice_not(term1,term2):
    global postings
    ans=[]
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
                j = j + 1
    return ans

def split_input1(input_string):
    global postings
    input_string= input_string.lower()
    terms = TextBlob(input_string).words.singularize()

    result = []
    for word in terms:
        expected_str = Word(word)
        expected_str = expected_str.lemmatize("v")
        result.append(expected_str)
    return result

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

def split_input2(input_string):
    global postings
    result=[]
    input_string=input_string.lower()
    input_string = nltk.word_tokenize(input_string)
    for item in  input_string:
        expected_str = Word(item)
        expected_str = expected_str.lemmatize("v")
        result.append(expected_str)
    #print(input_string)
    return result


def calculate(split_string):
    global postings
    ans = []
    split_list = split_input2(split_string)
    #print(split_list)
    if '(' in split_list and ')' in split_list:
        return ans
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
                        for ele in all:
                            if ele not in postings[split_list[i + 2]]:
                                temp.append(ele)
                        ans = sorted(list(set(ans) & set(temp)))
                #i+=2

        return ans
    else:
        print("input error ")
        return ans



def lncltn(terms,k):
    # 计算tf
    query_dict = {}
    unique_terms = set(terms)
    score = {}

    # 计算涉及到的doc的分数
    for every_term in unique_terms:
        query_dict[every_term] = 1 + math.log10(terms.count(every_term))
        for every_tuple in postings_for_topk[every_term]:
            # 对应的tf
            if every_tuple[0] in score.keys():
                score[every_tuple[0]]\
                    += query_dict[every_term] * document[every_term] * every_tuple[1]
            else:
                score.update({every_tuple[0]:
                                  query_dict[every_term] * document[every_term] * every_tuple[1]})
    # 根据分数对score字典进行排序
    ans = sorted(score.items(), key=lambda item: item[1], reverse=True)
    #print(ans)
    answer = []
    if len(ans) <= k:
        for every_doc in ans:
            answer.append(every_doc[0])
    else:
        temp = ans[:k]
        for every_doc in temp:
            answer.append(every_doc[0])
    return answer

def search1():
    global postings
    input_str=input("Please input your query:")
    if input_str=="Exit":
        exit(0)
    if input_str == "Back":
        return False
    terms=split_input1(input_str)
    #print(postings["surprisingly"])
    if terms==[]:
        print("your input is empty")
    if len(terms)==1:

        print(postings[terms[0]])
    elif len(terms)==2:
        print("sorry ,your input format is wrong!")
    #简单查询
    elif len(terms)==3:
        search_three_tuple(terms)
    elif len(terms)>3:
        print(calculate(input_str))
    return True

def search2():
    global postings
    global postings_for_topk
    global tweet_text
    input_str = input("Please input your query:")
    if input_str=="Exit":
        exit(0)
    if input_str == "Back":
        return False
    terms = split_input1(input_str)
    answer=lncltn(terms,topK)

    for docid in answer:
        print(tweet_text[docid-1])

    return True

def process_mb171(inputfile,outputfile):
    f1=open(inputfile,'r+')
    querynum=[]
    querytext=[]
    f2=open(outputfile,'w')
    for ele in f1.readlines():
        #print(ele[1])
        if ele!='\n':
            #print(ele[0:5])
            if ele[0:5]=='<num>':
                querynum.append(ele[16:19])
            elif ele[0:7]=='<query>':
                end=ele.find('</query>')
                #print(end)
                querytext.append(str(ele[8:end-1]))
    for i in range(55):
        f2.write(querynum[i]+' '+querytext[i]+'\n')
    #print(querynum[i],querytext[i])



def generate_query_result():
    input_query=open("MB-out.txt",'r')
    output_query=open("my_query_result.txt",'w')
    for line in input_query.readlines():
        query = line[4:]
        terms = split_input1(query)
        answer = lncltn(terms,30000)
        for docid in answer:
            output_query.write(line[:3]+' '+str(tweet_id[docid-1]))

def save_tweetid(inputfile="tweets1.txt"):
    tweetIDlist=[]
    with open(inputfile, 'r', errors='ignore') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            tweetid = json_obj['tweetId']
            tweetIDlist.append(tweetid)
    with open("tweetID.txt",'w') as out:
        for ele in tweetIDlist:
            out.write(ele+'\n')
    #np.save("tweetID.npy",tweetIDlist)

def choose_model():
    print("Tips:\n"
          "You can choose the following two models:\n"
          "Model 1(Boolean Retrieval Model):\n"
          "   (1)you can input only one term;\n"
          "   (2)your input can include 'and','or' and 'not',each operator is a binary operator;\n"
          "Model 2:(Ranked retrieval model)\n"
          "   (1)You can enter the query text freely just like web search\n"
          "You can choose the mode by entering Arabic numerals like '1' or '2'\n"
          "You can input 'Exit' to exit the program\n"
          "You can input 'Back' to go back to choose option\n")
    while True:
        opt=int(input("input your option here:"))
        if opt==1:
            while search1():
                pass
        elif opt==2:
            while search2():
                pass
        else:
            print("No such option")



def load_npy():
    print("文件载入中……")
    document=np.load("idf.npy").item()
    postings=np.load("postings.npy").item()
    postings_for_topk=np.load("postings_for_topk.npy").item()
    tweet_text = open("tweet_text.txt", 'r+').readlines()
    tweet_id =open('tweetID.txt','r+').readlines()
    print("文件载入完成")
    return document,postings,postings_for_topk,tweet_text,tweet_id

def main():
    global document,postings_for_topk,postings,tweet_text,tweet_id
    document,postings,postings_for_topk,tweet_text,tweet_id=load_npy()
    generate_query_result()
    #save_tweetid("tweets1.txt")
    #choose_model()


if __name__ == '__main__':
    main()

