import re
import nltk
import numpy
import string
import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict

#index  最先出现的位置
#rindex  str.rindex(str, beg=0 end=len(string))返回子字符串 str
# 在字符串中最后出现的位置，如果没有匹配的字符串会报异常，
# 你可以指定可选参数[beg:end]设置查找的区间。


postings = defaultdict(dict)
all=[]

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
    f = open("tweets1.txt", 'r')
    lines=f.readlines()
    #merger or surprisingly
    i=1
    for line in lines:
        all.append(i)
        line=preprocess(line)
        unique_terms=set(line)
        for every_term in unique_terms:
            if every_term in postings.keys():
                postings[every_term].append(i)
            else:
                postings[every_term]=[i]
        i=i+1

    number=i-1
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
                j=j+1
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

def search():
    global postings
    input_str=input("Please input your query:")
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

    #elif len(terms)==5:

def tips():
    print("Tips:\n"
          "Model 1: you can input only one term;\n"
          "Model 2: your input can include 'and','or' and 'not',each operator is a binary operator;\n"
          #"Model 3: your input can include '()'to represent the operator's priority.\n"
          )

def main():
    global postings
    get_postings()
    tips()
    while True:
        search()

if __name__ == '__main__':
    main()

