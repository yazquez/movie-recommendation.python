import os
from pymongo import MongoClient
import re
import sys
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





ROOT_PATH = "d:/tfm/tmp/{0}"

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.github
    return db.projects

def get_words(text):
    def add_word(word):
        word = word.lower()
        if word not in stop_words and not word.replace('.', '').isdigit():
            words.append(stemmer.stem(word))

    words = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(tokenizer.tokenize(text))):
        # nltk.word_tokenize    devuelve la lista de palabras que forman la frase (tokenización)
        # nltk.pos_tag          devuelve el part of speech (categoría) correspondiente a la palabra introducida
        # nltk.ne_chunk         devuelve la etiqueta correspondiente al part of speech (POC)
        try:
            if chunk.label() == 'PERSON':
                # PERSON es un POC asociado a los nombres propios, los cuales no vamos a añadir
                pass
            else:
                for c in chunk.leaves():
                    add_word(c[0])
        except AttributeError:
            add_word(chunk[0])

    return words


def process_readme_file(project):
    project['done'] = True
    project['readme_words'] = get_words(project['readme_txt'])
    projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

projects = get_repository()

for project in projects.find({'done':False}):
    print("Processing 'readme' of",project["name"])
    process_readme_file(project)