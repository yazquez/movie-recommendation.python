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

import nltk

def load_projects(max_projects=90000):
    def get_repository():
        client = MongoClient('localhost', 27017)
        db = client.github
        return db.projects

    projects_repository = get_repository()

    return list(projects_repository.find({'done': True}).limit(max_projects))


projects = load_projects(100)

def get_global_texts(projects):
    global_texts = []
    [global_texts.append(project['library']) for project in projects]
    return global_texts

global_texts = get_global_texts(projects)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer0 = CountVectorizer(min_df=1)

X_train0 = vectorizer0.fit_transform(global_texts)



print(vectorizer0.get_feature_names())

# Lo que obtenemos es una matriz dispersa con la representacion de una fila por cada documento y una columna por cada palabra
print(X_train0[0])

# Podemos ver las dimensioneas de esta matriz

X_train0.shape

# Tambien podemos mostrarla como una matriz normal
print(X_train0.toarray())


import scipy as sp



def mas_cercano(X_train, docs, nuevo_doc, vectorizer, dist):
    X_n = vectorizer.transform([nuevo_doc])[0]
    mejor_dist = float("inf")
    mejor_i = None
    print("Vector de la consulta: %s" % X_n.toarray())

    for i, (doc, doc_vec) in enumerate(zip(docs, X_train)):
        d = dist(doc_vec, X_n)
        print("=== Documento {0} con distancia={1:.2f}: {2}".format(i, d, doc))

        if d < mejor_dist:
            mejor_dist = d
            mejor_i = i

    print("El documento más cercano es {0} con distancia={1:.2f}".format(mejor_i, mejor_dist))


# In[14]:
# Distancia del Coseno
# La distancia del coseno no es propiamente una distancia sino una
# medida de similaridad entre dos vectores en un espacio que tiene defi-
# nido un producto interior. En el espacio euclídeo este producto interior
# es el producto escalar, ecuación 5. La similaridad coseno no debe ser
# considerada como una métrica debido a que no cumple la desigualdad
# triangular.
# ~X1 · ~X2 = ||X1|| ||X2|| cos(θ) (5)
# similaridad = cos(θ) =
# ~X1 · ~X2
# ||X1|| ||X2|| (6)
# Para que la medida de similaridad esté en el rango (0,1), se puede
# calcular a través de la fórmular 1 −
# arccos(similaridad)
# π
# .
# En minería de datos se suele emplear como un indicador de cohesión
# de clusteres de textos.
def dist_cos(v1, v2):
    return 1 - (v1 * v2.transpose()).sum() / (sp.linalg.norm(v1.toarray()) * sp.linalg.norm(v2.toarray()))


mas_cercano(X_train0, global_texts, "sys os json codecs shutil re", vectorizer0, dist=dist_cos)


