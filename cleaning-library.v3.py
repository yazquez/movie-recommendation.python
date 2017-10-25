

# coding: utf-8

# # Importación de librerías y módulos

# In[1]:


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





# ## Carga de los datos
# 
# Funcion para cargar las sinopsis de las proyectos y los metadados de las mismas (género, pais, etc). Nos quedaremos con las proyectos para las cuales tengamos tanto los metadatos como la sinpsis. La lista de proyectos será un objeto de tipo **list** y los datos de cada proyecto un diccionario.

# In[2]:
#
# library = ['calendar', 'select', 'struct', 'sys', '..termios', 'time', '..dbus', 'serial', 'gobject', 'dbus.mainloop.glib']
# x = [elem for elem in library if not elem.startswith('.')]
# print(x)
#


def load_projects(max_projects=90000):
    def get_repository():
        client = MongoClient('localhost', 27017)
        db = client.github
        return db.projects

    projects_repository = get_repository()

    return list(projects_repository.find({'done': True}).limit(max_projects))

projects = load_projects(2)



tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


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


def get_global_texts_original(projects):
    print("Extrayendo palabras de los textos...")
    global_texts = []
    [global_texts.append(get_words(project['readme_txt'])) for project in projects]

    return global_texts

global_texts = get_global_texts_original(projects)


for line in global_texts:
    for word in line:
        if word.isdigit():
            print(word)

#
#
# for project in projects:
#     print(project['readme_txt'])
