import os
import sys


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



generate_query_result()
