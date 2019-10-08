import re
import nltk
import numpy
import string
import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict
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
    print("not")
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

def split_input2(input_string):
    input_string = input_string.lower()
    input_string = nltk.word_tokenize(input_string)
    # print(input_string)
    return input_string


def calculate(split_string):
    ans = []
    split_list = split_input2(split_string)
    #print(split_list)
    if '(' in split_list and ')' in split_list:
        return ans
    elif '(' not in split_list and ')' not in split_list:
        for i in range(len(split_list)):
            print(i)
            if i == 0:
                if split_list[i + 1] == 'and':
                    ans = merge_and2(split_list[i], split_list[i + 2])
                elif split_list[i + 1] == "or":
                    ans = merge_or2(split_list[i], split_list[i + 2])
                elif split_list[i + 1] == "not":
                    ans = twice_not(split_list[i], split_list[i + 2])
            else:
                if split_list[i + 1] == 'and':
                    ans = merge_and2(ans, split_list[i + 2])
                elif split_list[i + 1] == "or":
                    ans = merge_or2(ans, split_list[i + 2])
                elif split_list[i + 1] == "not":
                    ans = twice_not(ans, split_list[i + 2])
            i = i + 2

        return ans
    else:
        print("input error ")
        return ans


calculate(input("input here"))