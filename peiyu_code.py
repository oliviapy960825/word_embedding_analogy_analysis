# -*- coding: utf-8 -*-
"""
Created on Sat May  2 08:21:32 2020

@author: Peiyu Wang
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
#from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import ast
from scipy.spatial import distance
import pickle

#get male words out in a list
fname = 'male_word_list.txt'
male_words=[]
with open(fname, "r", encoding='utf8') as f:
    for line in f:
        s = line.split()
        male_words.append(s[0])
len(male_words)

#get profession words out in a list
fname = 'Profession.txt'
with open(fname, 'r') as f:
    mylist = ast.literal_eval(f.read())
profession_list=[]       
for i in mylist:
    profession_list.append(i[0])
print(profession_list)
print(len(profession_list))

#get female words out in a list
fname = 'female_word_list.txt'
female_words=[]
with open(fname, "r", encoding='utf8') as f:
    for line in f:
        s = line.split()
        female_words.append(s[0])    
female_words

#get word indexes and embeddings from the original word embedding
fname = 'coha_embedding_1990.txt'
words_original=[]
vecs_original=[]
with open(fname, "r", encoding='utf8') as f:
    for line in f:
        s = line.split()
        v = np.array([float(x) for x in s[1:]])
        words_original.append(s[0])
        vecs_original.append(v)

#have a look at the original embedding shape
print(len(words_original))
print(len(vecs_original))
print(vecs_original[0].shape)



#find analogies of the profession in the original word embedding
dic={}
she_index=words_original.index('she')
she_embedding=vecs_original[she_index]
he_index=words_original.index('he')
he_embedding=vecs_original[he_index]
she_he=(she_embedding-he_embedding).reshape(-1,300)
for pro_1 in profession_list:
    if pro_1 in words_original:
        pro_1_index=words_original.index(pro_1)
        pro_1_embedding=vecs_original[pro_1_index]
        max_score=0
        max_score_reverse=0
        reserve_word=''
        reverse_word='
        for word_2 in words_original:
            if word_2!=pro_1:
                word_2_index=words_original.index(word_2)
                word_2_embedding=vecs_original[word_2_index]
                if distance.euclidean(pro_1_embedding,word_2_embedding)<=1:
                    score=cosine_similarity((pro_1_embedding-word_2_embedding).reshape(-1,300),she_he)
                    score_reverse=cosine_similarity((pro_1_embedding-word_2_embedding).reshape(-1,300),he_she)
                    if score>max_score:
                        max_score=score
                        reserve_word=word_2
                    if score_reverse>max_score_reverse:
                        max_score_reverse=score_reverse
                        reverse_word=word_2
                        
        if reserve_word!='':
            print(pro_1+ " : "+reserve_word)
            dic[pro_1]=reserve_word
        elif reverse_word!='':
            print(reverse_word+ " : "+pro_1)
            dic[reverse_word]=pro_1

new_dic={}
he_she=(he_embedding-she_embedding).reshape(-1,300)
for pro_1 in dic.keys():
    if dic[pro_1]=='':
        if pro_1 in words_original:
            pro_1_index=words_original.index(pro_1)
            pro_1_embedding=vecs_original[pro_1_index]
            max_score=0
            reserve_word=''
            for word_2 in words_original:
                if word_2!=pro_1:
                    word_2_index=words_original.index(word_2)
                    word_2_embedding=vecs_original[word_2_index]
                    if distance.euclidean(pro_1_embedding,word_2_embedding)<=1:
                        score=cosine_similarity((pro_1_embedding-word_2_embedding).reshape(-1,300),he_she)
                        if score>max_score:
                            max_score=score
                            reserve_word=word_2 
            if reserve_word!='':
                print(reserve_word+ " : "+pro_1)
                new_dic[reserve_word]=pro_1
    else:
         new_dic[pro_1]=dic[pro_1]
#print(dic)
print(new_dic)

file_name="original_profession"
outfile=open(file_name,'wb')
pickle.dump(new_dic,outfile)
outfile.close()

dic={}
for pro_1 in female_words[:50]:
    if pro_1 in words_original:
        pro_1_index=words_original.index(pro_1)
        pro_1_embedding=vecs_original[pro_1_index]
        max_score=0
        reserve_word=''
        for word_2 in words_original:
            if word_2!=pro_1:
                word_2_index=words_original.index(word_2)
                word_2_embedding=vecs_original[word_2_index]
                if distance.euclidean(pro_1_embedding,word_2_embedding)<=1:
                    score=cosine_similarity((pro_1_embedding-word_2_embedding).reshape(-1,300),she_he)
                    if score>max_score:
                        max_score=score
                        reserve_word=word_2
        if reserve_word!='':
            print(pro_1+ " : "+reserve_word)
            dic[pro_1]=reserve_word
            
print(dic)


file_name="original_female"
outfile=open(file_name,'wb')
pickle.dump(dic,outfile)
outfile.close()

