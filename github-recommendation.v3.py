# coding: utf-8

# # Importación de librerías y módulos

# In[1]:

from pymongo import MongoClient
import nltk
import json
import itertools
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities, matutils
import numpy as np


# ## Carga de los datos
# 
# Funcion para cargar las sinopsis de las proyectos y los metadados de las mismas (género, pais, etc). Nos quedaremos con las proyectos para las cuales tengamos tanto los metadatos como la sinpsis. La lista de proyectos será un objeto de tipo **list** y los datos de cada proyecto un diccionario.

# In[2]:

def load_projects3(max_projects=90000):
    def get_repository():
        client = MongoClient('localhost', 27017)
        db = client.github
        return db.projects

    projects = get_repository()
    projects = []

    for project in projects.find({'done': True}).limit(1000):
        project = dict()
        project['id'] = project['id']
        project['name'] = project['name']
        project['date'] = project['created_at']
        project['libraries'] = project['library']
        project['readme_txt'] = project['readme_txt']
        projects.append(project)

    return projects


def load_projects(max_projects=90000):
    def load_plot_summaries():
        plot_summaries_file = open("data/plot_summaries.fake.txt", "r", encoding="utf8")
        # plot_summaries_file = open("data/plot_summaries.txt", "r", encoding="utf8")
        plot_summaries = dict()
        for plot_readme_txt_line in plot_summaries_file:
            plot_readme_txt_data = plot_readme_txt_line.split('\t')
            # Summaries structure
            # [0] Wikipedia project ID
            # [1] Summary plot
            plot_readme_txt = dict()
            plot_readme_txt['id'] = plot_readme_txt_data[0]
            plot_readme_txt['readme_txt'] = plot_readme_txt_data[1]

            plot_summaries[plot_readme_txt['id']] = plot_readme_txt

        return plot_summaries

    print("Cargando datos de proyectos...")

    plot_summaries = load_plot_summaries()

    metadata_file = open("data/movie.metadata.fake.tsv", encoding="utf8")
    # metadata_file = open("data/project.metadata.tsv", "r", encoding="utf8")
    projects = []

    projects_count = 0

    for metadata_line in metadata_file:
        project_metadata = metadata_line.split('\t')

        # Metadata structure
        # [0] Wikipedia project ID
        # [1] Freebase project ID
        # [2] Project name
        # [3] Project release date
        # [4] Project box office revenue
        # [5] Project runtime
        # [6] Project languages (Freebase ID:name tuples)
        # [7] Project countries (Freebase ID:name tuples)
        # [8] Project libraries (Freebase ID:name tuples)

        id = project_metadata[0]

        # Añadimos la proyecto solo si tiene sinopsis, incluimos una lista con las claves de los generos
        if (id in plot_summaries) & (projects_count < max_projects):
            projects_count += 1
            project = dict()
            project['id'] = id
            project['name'] = project_metadata[2]
            project['date'] = project_metadata[3]
            project['libraries'] = list(json.loads(project_metadata[8].replace("\"\"", "\"").replace("\"{", "{").replace("}\"", "}")).values())
            project['readme_txt'] = plot_summaries[id].get('readme_txt')
            projects.append(project)

    print("Número de proyectos cargadas:", len(projects))

    return projects


# Usando la funcíon anterior cargamos los datos, dado que el proceso es bastante pesado solo cargaremos un subconjunto de las mismas

# In[3]:

projects = load_projects(1000)

# # Procesado de las sinópsis
# 
# A continuación procesamos las sinopsis, quedándonos con las palabras diferentes que encontramos en cada una de ellas, adicionalmente las vamos a transformar en sus raices (stemmer) y obviaremos las **stopwords** y los nomrbres propios, dado que no aportan significado.
# 
# Primero definimos una función para hacerlo

# In[4]:

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def get_words(text):
    def add_word(word):
        word = word.lower()
        if word not in stop_words:
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


def get_global_texts(projects):
    print("Extrayendo palabras de los textos...")
    global_texts = []

    [global_texts.append(get_words(project['readme_txt'])) for project in projects]

    return global_texts


# Y la ejecutamos, tendremos una lista de listas, en la que para cada proyecto tendremos las palabras que definen su sinopsis

# In[5]:


global_texts = get_global_texts(projects)


# A modo de ejemplo, mostramos las 5 primeras entradas de la primera proyecto

# In[6]:

global_texts[0][:5]

# # Creación del diccionaro
# 
# El diccionario está formado por la concatenación de todas las palabras que aparecen en alguna sinopsis (modo texto) de alguna de las proyectos. Básicamente esta función mapea cada palabra única con su identificador. Es decir, si tenemos N palabras, lo que conseguiremos al final es que cada proyecto sea representada mediante un vector en un  espacio de N dimensiones.
# 
# Para ello, partiendo de la lista creada en el paso anterior, usaremos la función **corpora** del paquete **gensim**.

# This module implements the concept of Dictionary -- a mapping between words and
# their integer ids.
#
# Dictionaries can be created from a corpus and can later be pruned according to
# document frequency (removing (un)common words via the :func:`Dictionary.filter_extremes` method),
# save/loaded from disk (via :func:`Dictionary.save` and :func:`Dictionary.load` methods), merged
# with other dictionary (:func:`Dictionary.merge_with`) etc.

# # In[7]:

dictionary = corpora.Dictionary(global_texts)
dictionary

# # Creación del Corpus
# 
# Crearemos un corpus con la colección de todos los resúmenes previamente pre-procesados y transformados usando el diccionario.

corpus = [dictionary.doc2bow(text) for text in global_texts]

# A modo de ejemplo, mostramos las 5 primeras entradas de la primera proyecto

# In[9]:

corpus[0][:5]


# # Creación del TFID
# 
# Un alto peso en tf-idf se alcanza por una alta frecuencia en un Documento y una baja frecuencia en toda la colección de documentos; los pesos tienden a filtrar términos comunes. Para la creación de este corpus, vamos a usar la función **TfidfModel** del objeto **models** (perteneciente a la librería *gemsim*).

# In[10]:

def create_tfidf(corpus):
    print("Creación del Modelo Espacio-Vector Tf-Idf")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


corpus_tfidf = create_tfidf(corpus)
corpus_tfidf[0]
# Out[3]: [(0, 0.5080429008916749), (1, 0.5080429008916749), (2, 0.695546419520037)]


# # Creación del Modelo LSI
# 
# Para ello vamos a definir una función auxiliar y posteriormente la invocaremos, además de crear el modelo LSI, vamos a usarlo para crear la matriz de similitudes. Antes de nada vamos a definir un par de valores clave para controlar el proceso.
# 
#  * **TOTAL_LSA_TOPICS**
# Limita el numero de terminos, por supuesto tiene que ver con el tamaño de la muestra, mientras más proyectos tengamos mas terminos tendremos y por tanto la reduccion seria mayor, estamos clusterizando las proyectos en TOTAL_TOPICOS_LSA clusters
# 
# * **SIMILARITY_THRESHOLD** Umbral de similitud que se debe superar para que dos proyectos se consideren similares
# 
# 
# Damos valor a estos parámetros, como el número de proyectos que vamos a usar en el ejemplo es pequeño, vamos a ajustar el umbral a solo 0.4.

# In[11]:

TOTAL_LSA_TOPICS = 5
SIMILARITY_THRESHOLD = 0.6
LIBRARY_COINCIDENCE_RATE = 0.01


# Em corpus tenemos, para cada project, una lista con sus palabras y el tfidf de cada una.

# corpus_tfidf[0]
# Out[13]: [(0, 0.5080429008916749), (1, 0.5080429008916749), (2, 0.695546419520037)]
# corpus_tfidf[1]
# Out[14]: [(2, 0.2691478902490874),(3, 0.39318346293947504),(4, 0.39318346293947504),(5, 0.39318346293947504),(6, 0.39318346293947504),(7, 0.39318346293947504),(8, 0.39318346293947504)]

# Si queremos saber que palabra es cada uno de estos termonos podemo consultar el diccionario

# dictionary[0]
# Out[11]: 'uno'
# dictionary[1]
# Out[12]: 'ghost'
# dictionary[2]
# Out[15]: 'murder'



def create_lsi_model(corpus_tfidf, dictionary, total_lsa_topics):
    print("Creación del modelo LSA: Latent Semantic Analysis")
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=total_lsa_topics)
    similarity_matrix = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    return lsi, similarity_matrix


(lsi, similarity_matrix) = create_lsi_model(corpus_tfidf, dictionary, TOTAL_LSA_TOPICS)

# Definimos la función auxiliar que dado una proyecto, nos determina la lista de proyectos que superan el umbral de similitud, como variante vamos a bonificar las proyectos que
# compartan librerías con el proyecto analizado, de este modo, además del indice de similitud calcularo anteriormente, vamos a sumar el valor de la constante LIBRARY_COINCIDENCE_RATE
# por cada librería coincidente. A
# Para cada proyecto que supere el umbral, almacenaremos el índice dentro de la matriz de proyectos, para localizarla posteriormente, y el grado de similitud
def get_similarities(doc, libraries):
    def library_score(libraries_to_compare):
        common_libraries = len(set(libraries).intersection(libraries_to_compare))
        return common_libraries * LIBRARY_COINCIDENCE_RATE

    project_similarities = []

    vec_bow = dictionary.doc2bow(get_words(doc))

    vec_lsi = lsi[vec_bow]  # convert the query to LSI space

    similarities = similarity_matrix[vec_lsi]
    similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

    for sim in similarities:
        similarity_project = projects[int(sim[0])]
        similarity_score = sim[1] + library_score(similarity_project["libraries"])
        if similarity_score > SIMILARITY_THRESHOLD:
            project_similarities.append((similarity_project, similarity_score))
    return(project_similarities)

project_similarities = get_similarities("A murder woman", ["Mystery", "Drama", "Biographical film"])

for similarity in project_similarities:
    print("Project: {0}: - Similarity: {1}".format(similarity[0]["name"], similarity[1]))



# num_docs = len(global_texts)
#
# num_terms = len(dictionary)
#
# numpy_matrix = matutils.corpus2dense(corpus, num_terms=num_terms, num_docs=num_docs)
# numpy_matrix
#
#
# s = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv=False)
#
# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4])
#
#
# plt.figure(figsize=(10,5))
# plt.hist(s[0], bins=100)
# plt.xlabel('Singular values', fontsize=12)
# plt.show()


# write out coordinates to file
fcoords = open("data/coords.csv", 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()


