import warnings
warnings.filterwarnings('ignore')


# In[14]:

import requests
import json
import os
import sys
from time import sleep
from pymongo import MongoClient
from datetime import date, timedelta


# A continuación vamos a definir una serie de constantes que serán usadas en el código, de esta manera podremos cambiar facilimente alguno de estos valores.
# El significado de cada una de ellas es el siguiente:
#  - MIN_STARTS: Número mínimo de estrellas que tiene que tener un proyecto para ser considerado
#  - START_DATE: Fecha de inicio del intervalo, haciendo referencia a la fecha de creación del proyecto en Github
#  - END_DATE: Fecha de fin del intervalo
#  - URL_PATTERN: Patrón de la URL para la consulta a la API de Github, podemos ver que filtramos por lenguaje (Python), por fecha de creación (la suministraremos durante la ejecución) y por numero de estrellas (referido a la constante comentada anteriormente).
#  - ROOT_PATH: Directorio raiz donde clonaremos los repositorios
#  - CLONE_COMMAND: Comando usado para clonar los repositorios

# In[3]:

MIN_STARTS = 10
START_DATE = date(2017, 8, 1)
END_DATE = date(2017, 8, 2)
URL_PATTERN = 'https://api.github.com/search/repositories?q=language:Python+created:{0}+stars:>={1}&type=Repositories'
CLONE_COMMAND = "git clone {0} {1}"
ROOT_PATH = "d:/tfm/tmp"



# Como se ha comentado anteriormente, vamos a usar MongoDB como tecnología de persistencia, vamos a definir una funcion que nos devuelva una referencia a la colección de proyectos con la que estamos trabajando.

# In[4]:

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.github10
    return db.projects


# La siguiente función es la que realmente hace las llamadas a la API, tendrá como entrada la fecha de creación de los proyectos, usará la URL que hemos definicdo como constante.

# In[5]:

def get_projects_by_date(date):
    print("Processing date", date)
    url_pattern = URL_PATTERN
    url = url_pattern.format(date, MIN_STARTS)
    response = requests.get(url)
    if (response.ok):
        response_data = json.loads(response.content.decode('utf-8'))['items']
        for project in response_data:
            insert_project(project)
    else:
        response.raise_for_status()


# A continuanción definimos otra función auxiliar, que dada la información de la respuesta de la API, seleccionamos los atributos que vamos a necesitar y los almacenacomo un documento Mongo. Adicionalmente crea las propiedades mencionadas previamente (readme_txt,library,etc.).

# In[6]:

def insert_project(github_project):
    if repository_projects.find_one({"id": github_project["id"]}):
        print("Project {0} is already included in the repository".format(github_project["name"]))
    else:
        project = {
            'id': github_project["id"],
            'name': github_project["name"],
            'full_name': github_project["full_name"],
            'created_at': github_project["created_at"],
            'git_url': github_project["git_url"],
            'description': github_project["description"],
            'language': github_project["language"],
            'stars': github_project["stargazers_count"],
            'done': False,
            'readme_txt': "",
            'library': [],
            'raw_data': github_project
        }
        repository_projects.insert(project)


# Finalmente ejecutamos el programa que, usando las funciones anteriormente definidas, descarga la información de los proyectos y la inserta en la BBDD, podemos observar que descompone las llamadas para traer en cada una de ellas solo los proyectos de un determinado dia y que *"gestiona"* las restricciones que tenemos en cuanto a llamada en la ventana de tiempo actual. En primer lugar recuperamos la colección en la que vamos a insertar los proyectos.

# In[7]:

repository_projects = get_repository()

# Y cargamos los proyectos en la colección, aplicamos las técnicas descritas para salvar las restricciones que nos impone la API de Github (número de items devuelto por cada consulta y número de llamadas por minuto).

# In[8]:

# for project_create_at in [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]:
#     try:
#         get_projects_by_date(project_create_at)
#     except:
#         print(">> Reached call limit, waiting 61 seconds...")
#         sleep(61)
#         get_projects_by_date(project_create_at)
#
#
#
# os.chdir(ROOT_PATH)
#
# for project in repository_projects.find({'done':False}):
#     print("Cloning project", project['name'], "...")
#     path = ROOT_PATH + "/" + str(project["id"])
#     if not os.path.isdir(path):
#         os.system(CLONE_COMMAND.format(project["git_url"], project["id"]))


# En este punto tendríamos clonados todos los repositorios con los que construiremos el sistema de recomendación.
# 
# **NOTA**: Son muchos, muchos gigabytes.

# ## 3. Procesamiento de los proyectos clonados

# Comenzamos esta etapa definiendo una función que dado un fichero python (extensión **.py**), lo recorre linea por linea y, usando expresiones regulares, buscamos todas las librerias que se estén usando.
# Almacenamos esta lista en la propiedad **library** del proyecto en cuestión.

# In[10]:

def process_python_file(project, file_path):
    def add_to_list(item):
        if not item in library:
            library.append(item)

    library = project['library']
    pattern = '(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)[ ]*$'
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                match = re.search(pattern, line)
                if match:
                    if match.group(1) != None:
                        add_to_list(match.group(1))
                    else:
                        add_to_list(match.group(2))
            except:
                print("Error procesing project {0} [{1}] - {2}".format(project['id'], project['name'], sys.exc_info()[0]))

    project['library'] = library
    repository_projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


# Por otra parte, definimos también una función que dado un fichero **README** lo lea y almacena en la propiedad **readme_txt**

# In[11]:

def process_readme_file(project, file_path):
    with open(file_path, 'r') as f:
        project['readme_txt'] = f.read()
        repository_projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


# Finalmente recorremos los repositorios no procesados, como comentabamos anteriormente este proceso podría ser necesario lanzarlo reiteradas veces, para cada repositorio analizamos su fichero **README** y cada uno de los ficheros Python para extraer las librerías.

# In[17]:

#for project in repository_projects.find({'done':False}):
for project in repository_projects.find():
    try:
        path = ROOT_PATH.format(project["id"])
        if os.path.isdir(path):
            print("Processing project ", project["name"])
            project['done'] = True
            repository_projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)
            for root, dirs, files in os.walk(path):
                for file in files:
                    try:
                        if file.endswith(".py"):
                            process_python_file(project, os.path.join(root, file))
                        else:
                            if file.lower().startswith("readme."):
                                process_readme_file(project, os.path.join(root, file))
                    except:
                        print("Error procesing project {0} [{1}] - {2}".format(project['id'], project['name'], sys.exc_info()[0]))
    except:
        print("Error procesing project {0} [{1}] - {2}".format(project['id'], project['name'], sys.exc_info()[0]))


# En este punto, tendríamos una colección MongoDB con cada uno de los proyectos de nuestro **pre-corpus** con la lista de librerías que cada proyecto usa así como su descripción **extendida** extraida de su fichero READMA. Estaríamos en disposición por tanto de comenzar la implementación de nuestro recomendador.

# # Implementación del recomendador

# ## Principios teóricos

# Para identificar la similitud entre los proyectos basándonos en su descripción, entendiéndose como tal su fichero README, utilizamos el "análisis semántico latente" (LSA, usando la abreviatura en inglés), que es una técnica ampliamente utilizada en el procesamiento del lenguaje natural. LSA transforma cada texto en un vector, en un espacio de características. En nuestro caso, las características son palabras que ocurren en las descripciones. A continuación, se crea una matriz que contiene todos los vectores: las columnas representan las descripciones de los proyectos y las filas representan palabras únicas. Por consiguiente, el número de filas puede ascender a decenas de miles de palabras. 
# 
# Con el fin de identificar las características relevantes de esta matriz, usaremos la "descomposición de valores singulares" (SVD, usando la abreviatura en inglés), que es una técnica de reducción de dimensión, se utiliza para reducir el número de líneas -palabras-, manteniendo y resaltando la similitud entre columnas-descripción -. La dimensión de esta matriz de aproximación se establece mediante un hiperparámetro que es el número de temas, comúnmente llamado como tópicos. En este marco, un tópico consiste en un conjunto de palabras con pesos asociados que definen la contribución de cada palabra a la dirección de este tópico. Basándose en esta matriz de aproximación de baja dimensión, la similitud entre dos columnas -descripciones- se calcula utilizando el coseno del ángulo entre estos dos vectores.
# 

#    ## Importación de librerías y módulos
#         

# Comenzamos importando las librerías que vamos a usar en esta fase del proyecto.

# In[ ]:

import os
import uuid
import re
import nltk
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities, matutils


# ### Carga de los datos
# 
# Definimos una funcion para cargar los datos de los proyectos, entre los datos a cargar está el fichero **readme** y las librerías usadas, datos que serán los que usemos para calcular las similitudes entre los proyectos.

# In[ ]:

def load_projects(max_projects=50000):
    projects_repository = get_repository()
    return list(projects_repository.find({'done': True}).limit(max_projects))


# Usando la funcíon anterior cargamos los datos, dado que el proceso es bastante pesado solo cargaremos un subconjunto de las mismas

# In[ ]:

projects = load_projects()


# A modo de ejemplo, mostramos las primeras 5 librerías de uno de los proyectos

# In[ ]:

len(projects)


# In[ ]:

projects[1]['library']


# Y el contenido del fichero README

# In[ ]:

projects[2]['readme_txt']


# ## Preprocesamiento de los datos

# ### Descripciones
#  
# En primer lugar procesamos las descripciones, quedándonos con las palabras diferentes que encontramos en cada una de ellas, en primer lugar, eliminamos puntuaciones y palabras irrelevantes como nombres personales y palabras comunes que no aportan significado (denominados "stop words"). Esto evitará que se formen tópicos en torno a ellos
# 
# Además, utilizamos el stemmer Snowball, también llamado Porter2 stemmer, para detectar palabras similares presentes en diferentes formatos (eliminar sufijo, prefijo, etc.). Snowball es un lenguaje desarrollado por M.F. Porter, para definir de forma eficiente stemmers. Este algoritmo de derivación es el más utilizado en el dominio del procesamiento del lenguaje natural.
# 
# Para hacer el procesamietno usaremos la librería nltk( Natural Language Toolkit), proporciona un gran número de métodos que cubren diferentes temas en el dominio de los datos del lenguaje humano, como la clasificación, derivación, etiquetado, análisis y razonamiento semántico.
# 
# 

# In[ ]:

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


# Creamos una función que dado un texto lo descompone en las palabras con significado que lo componen. Vamos a excluir los siguientes tipos de palabras:
# 
# - Nombres propios
# - Palabras consideradas como "Stop Words"
# - Números
# 
# Adicionalmente aplicamos el proceso de "steamming" que comentabamos anteriormente

# In[ ]:

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


# Creamos una función que aplica la función anterior a los resúmenes de todos los proyectos.

# In[ ]:

def get_texts_from_readme(projects):
    texts = []
    [texts.append(get_words(project['readme_txt'])) for project in projects]
    return texts


# Finalmente ejecutamos la función que acabamos de definir, tendremos una lista de listas, en la que para cada proyecto tendremos las palabras que lo definen.

# In[ ]:

texts_from_readme = get_texts_from_readme(projects)


# A modo de ejemplo, mostramos las 5 primeras entradas de uno de los proyectos.

# In[ ]:

texts_from_readme[2][:5]


# In[ ]:

len(projects)


# ### Librerías
#  
# Las librerías también las vamos a "limpiar", este proceso lo podríamos haber hecho cuando cargamos los datos desde los proyectos clonados, pero he preferido hacerlo en un paso adicional para dejar ese proces lo más simple posible. Por otro lado es posible que con más conocimiento del dominio, pudisen surgir otras posibles mejoras de cara a mejorar la calidad de los datos. 

# En principio solo vamos a excluir las librerías cuyo nombre empieza por ".", esto es un atajo que le dice al interprete de Python que busque en el paquete actual antes del resto de paths del PYTHONPATH. 

# In[ ]:

for project in projects:
    project['library'] = [library for library in project['library'] if not library.startswith('.')]


# ## Creación del diccionario
#  
# El diccionario está formado por la concatenación de todas las palabras que aparecen en algún resumen de alguno de los proyectos. Básicamente esta función mapea cada palabra única con su identificador. Es decir, si tenemos N palabras, lo que conseguiremos al final es que cada proyecto sea representada mediante un vector en un espacio de N dimensiones.
#  
# Para ello, partiendo de la lista creada en el paso anterior, usaremos la función **corpora** del paquete **gensim**.
# 
# El diccionario consiste en una concatenación de palabras únicas de todas las descripciones. Gensim es una biblioteca eficiente para analizar la similitud semántica latente entre documentos.
# Este módulo implementa el concepto de Diccionario - un mapeo entre palabras y
# sus entes ids.
# 
# Los diccionarios pueden ser creados a partir de un corpus y luego pueden ver las frecuencia del documento (eliminación de palabras comunes mediante el método func: `Dictionary.filter_extremes`), guardado / cargado desde el disco (vía: func: `Dictionary.save` y: func:` Dictionary.load`), fusionado con otro diccionario (: func: `Dictionary.merge_with`) etc.

# In[ ]:

dictionary = corpora.Dictionary(texts_from_readme)
dictionary


# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/lda_training_tips.ipynb
# 
# We remove rare words and common words based on their document frequency. Below we remove words that appear in less than 20 documents or in more than 50% of the documents. Consider trying to remove words only based on their frequency, or maybe combining that with this approach.

# In[ ]:

#POC
dictionary.filter_extremes(no_below=1, no_above=0.5)


# Podemos ver la longitud del diccionario creado

# In[ ]:

print(dictionary)


# La función **token2i** asigna palabras únicas con sus ids. En nuestro caso, la longitud del diccionario es igual a *N* palabras lo que significa que cada descripción del proyecto será representada a través de un espacio vectorial de *N* dimensiones
# 
# Mostramos las primeras 10 entradas

# In[ ]:

list(itertools.islice(dictionary.token2id.items(), 0, 10))


# ## Creación del Corpus
# 
# Crearemos un corpus con la colección de todos los resúmenes previamente pre-procesados y transformados usando el diccionario. Vamos a convertir los textos a un formato que gensim puede utilizar, esto es, una representación como bolsa de palabras (BoW). Gensim espera ser alimentado con una estructura de datos de corpus, básicamente una lista de "sparce vectors", estos constan de pares (id, score), donde el id es un ID numérico que se asigna al término a través de un diccionario. 

# Mostraos el numero de elementos únicos que existen en los documentos (los que componen el diccionario) y el numero de textos (correspondientes a las descripciones de los proyectos).

# In[ ]:

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(texts_from_readme))


# In[ ]:

def create_corpus(texts):
    return [dictionary.doc2bow(text) for text in texts]


# In[ ]:

corpus = create_corpus(texts_from_readme)


# A modo de ejemplo, mostramos las 5 primeras entradas del primer proyecto

# In[ ]:

corpus[0][:5]


# ## Creación del TFID
# 
# Un alto peso en tf-idf se alcanza por una alta frecuencia en un Documento y una baja frecuencia en toda la colección de documentos; los pesos tienden a filtrar términos comunes. Para la creación de este corpus, vamos a usar la función **TfidfModel** del objeto **models** (perteneciente a la librería *gemsim*).
# 

# In[ ]:

def create_tfidf(corpus):
    print("Creación del Modelo Espacio-Vector Tf-Idf")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


corpus_tfidf = create_tfidf(corpus)


# En corpus tenemos, para cada project, una lista con sus palabras y el tfidf de cada una. Mostramos el primer elemento

# In[ ]:

corpus_tfidf[0][:10]


# Si queremos saber que palabra es cada uno de estos términos podemos consultar el diccionario

# In[ ]:

print(dictionary[0],",",dictionary[1],",", dictionary[2])


# Si imprimimos el diccionario nos muestra la información resumida.

# In[ ]:

print(dictionary)


# ## Creación del Modelo LSI
# 
# Para ello vamos a definir una función auxiliar y posteriormente la invocaremos, además de crear el modelo LSI, vamos a usarlo para crear la matriz de similitudes. Antes de nada vamos a definir una serie de constantes para controlar el proceso.
#  
# * **TOTAL_LSA_TOPICS**
# Limita el numero de terminos, por supuesto tiene que ver con el tamaño de la muestra, mientras más proyectos tengamos mas terminos tendremos y por tanto la reduccion seria mayor, estamos clusterizando las proyectos en TOTAL_TOPICOS_LSA clusters
#  
# * **SIMILARITY_THRESHOLD** Umbral de similitud que se debe superar para que dos proyectos se consideren similares
# 
# * **LIBRARY_COINCIDENCE_RATE** porcentaje en el que incrementaremos la similitud de los proyectos por cada librería que dos proyectos tengan en común
# 
#  
# 

# ### Determinación del número de topicos

# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/lda_training_tips.ipynb
# 
# Training
# 
# We are ready to train the LDA model. We will first discuss how to set some of the training parameters.
# First of all, the elephant in the room: how many topics do I need? There is really no easy answer for this, it will depend on both your data and your application. I have used 10 topics here because I wanted to have a few topics that I could interpret and "label", and because that turned out to give me reasonably good results. You might not need to interpret all your topics, so you could use a large number of topics, for example 100.
# The chunksize controls how many documents are processed at a time in the training algorithm. Increasing chunksize will speed up training, at least as long as the chunk of documents easily fit into memory. I've set chunksize = 2000, which is more than the amount of documents, so I process all the data in one go. Chunksize can however influence the quality of the model, as discussed in Hoffman and co-authors [2], but the difference was not substantial in this case.
# passes controls how often we train the model on the entire corpus. Another word for passes might be "epochs". iterations is somewhat technical, but essentially it controls how often we repeat a particular loop over each document. It is important to set the number of "passes" and "iterations" high enough.
# I suggest the following way to choose iterations and passes. First, enable logging (as described in many Gensim tutorials), and set eval_every = 1 in LdaModel. When training the model look for a line in the log that looks something like this:
# 2016-06-21 15:40:06,753 - gensim.models.ldamodel - DEBUG - 68/1566 documents converged within 400 iterations
# 
# If you set passes = 20 you will see this line 20 times. Make sure that by the final passes, most of the documents have converged. So you want to choose both passes and iterations to be high enough for this to happen.
# We set alpha = 'auto' and eta = 'auto'. Again this is somewhat technical, but essentially we are automatically learning two parameters in the model that we usually would have to specify explicitly.

# *Como se mencionó anteriormente, LSA busca identificar un conjunto de topicos relacionados las descripciones de los proyectos. El número de estos temas N es igual a la dimensión de la matriz de aproximación resultante de la técnica de reducción de dimensión SVD. Este número es un hiperparámetro que se debe ajustar cuidadosamente, es el resultado de la selección de los N valores singulares más grandes de la matriz del corpus tf-idf. Estos valores singulares se pueden calcular de la siguiente manera:*

# In[ ]:

#POC

from collections import OrderedDict

trained_models = OrderedDict()
for num_topics in range(20, 41, 10):
    print("Training LDA(k=%d)" % num_topics)
    lda = models.LdaMulticore(
        corpus_tfidf, id2word=dictionary, num_topics=num_topics, workers=4,
        passes=10, iterations=100, random_state=42, 
        eval_every=None, # Don't evaluate model perplexity, takes too much time.
        alpha='asymmetric',  # shown to be better than symmetric in most cases
        decay=0.5, offset=64  # best params from Hoffman paper
    )
    trained_models[num_topics] = lda


# In[ ]:

trained_models


# In[ ]:

trained_models[20]


# In[ ]:

from gensim.models import CoherenceModel
goodcm = CoherenceModel(model=trained_models[20], texts=texts_from_readme, dictionary=dictionary, coherence='c_v')
badcm = CoherenceModel(model=trained_models[40], texts=texts_from_readme, dictionary=dictionary, coherence='c_v')


# In[ ]:

goodcm.get_coherence()


# In[ ]:

print(goodcm.get_coherence())
print(badcm.get_coherence())


# Finalmente creamos constantes para los valores seleccionados

# In[ ]:

TOTAL_LSA_TOPICS = 5
SIMILARITY_THRESHOLD = 0.6
LIBRARY_COINCIDENCE_RATE = 0.01


# ## Creación del modelo LSA: Latent Semantic Analysis

# In[ ]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:

model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=TOTAL_LSA_TOPICS)


# Podemos ver como influyen las palabras en la determinación de los diferentes tópicos, por ejemplo mostramos los 3 primeros tópicos

# In[ ]:

for i, topic in enumerate(model.print_topics(3)):
    print('Topic {}:'.format(i))
    print(topic[1].replace(' + ', '\n'))
    print('')


# Usando nuestro modelo LSI construimos la matriz de similitud

# In[ ]:

similarity_matrix = similarities.MatrixSimilarity(model[corpus_tfidf])


# Definimos la función auxiliar que dado un documento (correspondiente a la descripción de un proyecto), nos determina la lista de proyectos que superan el umbral de similitud. Para cada proyecto que supere el umbral, almacenaremos el índice dentro de la matriz de proyectos, para localizarla posteriormente, y el grado de similitud. La función nos devolverá la lista ordenada por similitud.

# In[ ]:

def get_similarities_by_description(doc, model):
    ''' Calcula las similitudes de un documento, expresado este como una lista de palabras'''
    project_similarities = []

    # Convertimos el documento al espacio LSI
    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = model[vec_bow]

    similarities = similarity_matrix[vec_lsi]
    similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

    for sim in similarities:
        similarity_project = int(sim[0])
        similarity_score = sim[1]
        if similarity_score > SIMILARITY_THRESHOLD:
            project_similarities.append((similarity_project, similarity_score))

    return (project_similarities)


# A modo de ejemplo, como prueba de concepto, vamos a determinar los proyectos similares a uno dado.

# In[ ]:

poc_readme_doc = get_words("works on Windows and write down some installation instructions")
poc_library = ['sys', 'os', 'json', 'codecs', 'shutil']
project_similarities_by_description = get_similarities_by_description(poc_readme_doc, model)


# A continuación, mostramos los 10 proyectos más similares

# In[ ]:

for similarity in project_similarities_by_description[:10]:
    print("Project: {0}: - Similarity: {1}".format(projects[similarity[0]]["name"], similarity[1]))


# El objetivo final no no es encontrar proyectos similares, sino encontrar las librerias que usan esos proyectos similares. Así pues, recorremos esos proyectos e identificamos esas librerías, descartando las que nuestro proyecto ya incluye.
# 
# Creamos un diccionario donde la clave es cada libreria y el valor el scoring de esa librería, obtendremos dicho scoring sumando el scoring de similitud de cada proyecto en el que encontremos la librería

# Recorremos los 10 proyectos más similares y calculamos el scoring de cada librería usada en esos proyectos

# In[ ]:

def get_library_similarities_by_description(project_similarities):
    library_similarities=defaultdict(float)
    for similarity in project_similarities[:10]:
        project_similarity = similarity[1]
        project_libraries = projects[similarity[0]]["library"]
        for library in project_libraries:
            if library not in poc_library:
                library_similarities[library] += project_similarity
    return library_similarities


# Mostramos las 10 librerías más usadas por los proyectos similares

# In[ ]:

library_similarities_by_description = get_library_similarities_by_description(project_similarities_by_description)


# In[ ]:

for library in sorted(library_similarities_by_description, key=library_similarities_by_description.get, reverse=True)[:10]:
    print("Library: {0}: - Score: {1}".format(library, library_similarities_by_description[library]))


# ## Usando las librerias 

# Otro acercamiento que podemos realizar es usar como nivel de similitud las propias librerías que un proyecto usa, es decir buscaremos proyectos que usen las mismas librerías que nosotros ya estamos usando y mostraremos otras librerias que esos mismos proyectos estén usando y nuestro proyecto no.

# In[ ]:

import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from collections import defaultdict


# El modelo que construiremos necesita como entrada textos, así que transformamos nuestras listas de palabras en textos, en los que las librerías estarán separados por espacios.

# In[ ]:

global_texts = [" ".join(project['library']) for project in projects]


# Usaremos un objeto de **CountVectorizer**, el cual convierte una colección de documentos de texto en una matriz de conteos de tokens. 
# 
# Useremos el parámetro **min_df** con el valor **2**,  denomina comunmente como **corte**, esto hará que al construir el vocabulario, ignorará los términos que tienen una frecuencia de documento más baja que el umbral dado

# In[ ]:

countVectorizer = CountVectorizer(min_df=1)


# Usando el objeto creado, realizamos la transformación de nuestros datos

# In[ ]:

X_train_data = countVectorizer.fit_transform(global_texts)


# A continuación definimos una función que usando los datos anteriores, el objeto countVectorizer y una lista de librerias, calcula la distancia coseno con cada una de las listas de librerias de los proyectos objeto de estudio. La **distancia coseno**  no es propiamente una distancia sino una medida de similaridad entre dos vectores en un espacio que tiene definido un producto interior. 

# In[ ]:

def get_similarities_by_library(X_train, libraries, countVectorizer):
    def cos_distance(v1, v2):
        return 1 - (v1 * v2.transpose()).sum() / (sp.linalg.norm(v1.toarray()) * sp.linalg.norm(v2.toarray()))
    
    new_doc = " ".join(libraries)
    new_doc_vectorized = countVectorizer.transform([new_doc])[0]
    library_similarities = defaultdict(float)
    for i, doc_vec in enumerate(X_train):
        library_similarities[i] = 1 - cos_distance(doc_vec, new_doc_vectorized)
    return library_similarities


# Vamos a realizar una prueba de concepto usando la lista de librerias definidas anterioremente
# 
# poc_library = ['sys', 'os', 'json', 'codecs', 'shutil']
# 

# In[ ]:

project_similarities_by_library = get_similarities_by_library(X_train_data, poc_library, countVectorizer)


# Como hicimos con el análisis por descripciones, podemos mostrar los proyectos ordenados por similitud

# In[ ]:

for project_id in sorted(project_similarities_by_library, key=project_similarities_by_library.get, reverse=True)[:10]:
     print("Project: {0}: - Similarity: {1}".format(projects[project_id]["name"], project_similarities[project_id]))


# Aunque como ya dijimos entonces, el objetivo es recomendar librerías, no proyectos. Para ello definimos una función que dada la lista anterior, nos muestra la lista de las librerías más usadas.

# In[ ]:

def get_library_similarities_by_libraries(project_similarities):
    library_similarities=defaultdict(float)

    for project_id in sorted(project_similarities, key=project_similarities.get, reverse=True)[:10]:
        project_similarity = project_similarities[project_id]
        project_libraries = projects[project_id]["library"]
        for library in project_libraries:
            if library not in poc_library:
                library_similarities[library] += project_similarity
    return library_similarities


# Y la invocamos con nuestra lista de proyectos

# In[ ]:

library_similarities_by_libraries = get_library_similarities_by_libraries(project_similarities_by_library)


# Finalmente mostramos el resultado

# In[ ]:

for library in sorted(library_similarities_by_libraries, key=library_similarities_by_libraries.get, reverse=True)[:10]:
    print("Library: {0}: - Score: {1}".format(library, library_similarities_by_libraries[library]))


# ## Aplicación a un proyecto de ejemplo
# 
# Para poner en práctica el sistema de recomendación vamos a usar un proyecto real almacenado en Github. En primer lugar definimos una función que dado un proyecto, lo descargue en local y aplique el mismo procesamiento que hicimos con los proyectos que componen nuestro corpus.

# In[ ]:

def proccess_url(git_url):
    def process_python_file(project, file_path):
        def add_to_list(item):
            if not item in library:
                library.append(item)

        library = project['library']
        pattern = '(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)[ ]*$'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    if match.group(1) is not None:
                        add_to_list(match.group(1))
                    else:
                        add_to_list(match.group(2))
        project['library'] = library

    def process_readme_file(project, file_path):
        with open(file_path, 'r') as f:
            project['readme_txt'] = f.read()

    project = dict()
    os.chdir(ROOT_PATH)
    dir_name = uuid.uuid4().hex
    path = ROOT_PATH + "/" + dir_name

    if not os.path.isdir(path):
        os.system(CLONE_COMMAND.format(git_url, dir_name))
        project['git_url'] = git_url
        project['library'] = []
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    if file.endswith(".py"):
                        process_python_file(project, os.path.join(root, file))
                    else:
                        if file.lower().startswith("readme."):
                            process_readme_file(project, os.path.join(root, file))
                except:
                    pass

    return project


# Y la invocamos con nuestro projecto de ejemplo

# In[ ]:

sample_project = proccess_url("https://github.com/yazquez/movie-recommendation.python.git")


# Podemos ver las librerías que está usando

# In[ ]:

sample_project['library']


# Así como su descripción (fichero **readme**)

# In[ ]:

sample_project['readme_txt']


# Si recapitulamos, tenemos las siguientes funciones definidas:
# 
# - **get_similarities_by_description**: Devuelve los proyectos similares en funcíon de sus descripciones
# - **get_library_similarities_by_description**: Devuelve las librerías comunes usando la lista anterior
# - **get_similarities_by_library**: Devuelve los proyectos similares en funcíon de las librerías que se usa
# - **get_library_similarities_by_libraries(project_similarities)**: Devuelve las librerías comunes usando la lista anterior
# 

# Las aplicamos a nuestro proyecto de ejemplo

# Empezamos obteniendo los proyectos más similares al del ejemplo usando las descripciones y a partir de estos proyectos obtenemos las librerías

# In[ ]:

readme_doc = get_words(sample_project['readme_txt'])
project_similarities_by_description = get_similarities_by_description(poc_readme_doc, model)
library_similarities_by_description = get_library_similarities_by_description(project_similarities_by_description)


# Mostramos los 10 proyectos más similares

# In[ ]:

for similarity in project_similarities_by_description[:10]:
    print("Project: {0}: - Similarity: {1}".format(projects[similarity[0]]["name"], similarity[1]))


# Hacemos lo mismo usando las librerías del proyecto como medida de similitud

# In[ ]:

project_similarities_by_library = get_similarities_by_library(X_train_data, poc_library, countVectorizer)
library_similarities_by_libraries = get_library_similarities_by_libraries(project_similarities_by_library)


# Mostramos los 10 proyectos más similares

# In[ ]:

for project_id in sorted(project_similarities, key=project_similarities.get, reverse=True)[:10]:
     print("Project: {0}: - Similarity: {1}".format(projects[project_id]["name"], project_similarities[project_id]))


# Finalmente mostramos el resultado, o sea las librerías recomendadas

# **Librerías recomendadas en función de las descripciones**

# In[ ]:

for library in sorted(library_similarities_by_description, key=library_similarities_by_description.get, reverse=True)[:10]:
    print("Library: {0}: - Score: {1}".format(library, library_similarities_by_description[library]))


# **Librerías recomendadas en función de las librerías comunes**

# In[ ]:

for library in sorted(library_similarities_by_libraries, key=library_similarities_by_libraries.get, reverse=True)[:10]:
    print("Library: {0}: - Score: {1}".format(library, library_similarities_by_libraries[library]))


# In[ ]:



