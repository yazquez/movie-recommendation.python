from collections import defaultdict

from pymongo import MongoClient
import nltk
import json
import itertools
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities, matutils
import numpy as np

import os
import uuid
import re

def load_projects(max_projects=90000):
    def get_repository():
        client = MongoClient('localhost', 27017)
        db = client.github
        return db.projects

    projects_repository = get_repository()

    return list(projects_repository.find({'done': True}).limit(max_projects))


projects = load_projects(100000)

def get_global_texts(projects):
    global_texts = []
    [global_texts.append(project['library']) for project in projects]

    return global_texts

global_texts = get_global_texts(projects)


# A modo de ejemplo, mostramos las 5 primeras entradas de la primera proyecto

# In[6]:

global_texts[0][:5]

dictionary = corpora.Dictionary(global_texts)
dictionary

# once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
# dictionary.filter_tokens(once_ids)
# dictionary.compactify()

corpus = [dictionary.doc2bow(text) for text in global_texts]
corpus[0][:5]


def create_tfidf(corpus):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


corpus_tfidf = create_tfidf(corpus)
corpus_tfidf[0]




def create_lsi_model(corpus_tfidf, dictionary):
    print("Creación del modelo LSA: Latent Semantic Analysis")
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)
    return lsi


lsi = create_lsi_model(corpus_tfidf, dictionary)

for i, topic in enumerate(lsi.print_topics(-1)):
    print('Topic {}:'.format(i))
    #print(topic.replace(' + ', '\n'))
    print(topic[1].replace(' + ', '\n'))
    print('')

