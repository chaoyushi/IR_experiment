
from textblob import TextBlob
from textblob import Word
from collections import defaultdict
f1=open("tweets1.txt",'r+')
f2=open("tweet_text.txt",'w')

lines=f1.readlines()
for line in lines:
    start=line.index("text")+6
    end=line.index("timeStr")-3
    document=line[start: end]
    document=document+'\n'
    f2.writelines(document)
