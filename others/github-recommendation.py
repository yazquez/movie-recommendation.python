
# coding: utf-8

# # Importación de librerías y módulos

# In[1]:
import datetime

from pymongo import MongoClient
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities


def get_time_elapsed(t0):
    diff = datetime.datetime.now() - t0
    return((diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)) / 1000


# ## Carga de los datos
# 
# Funcion para cargar las sinopsis de las películas y los metadados de las mismas (género, pais, etc). Nos quedaremos con las películas para las cuales tengamos tanto los metadatos como la sinpsis. La lista de películas será un objeto de tipo **list** y los datos de cada película un diccionario.

# In[2]:

def load_movies3(max_movies=90000):
    def get_repository():
        client = MongoClient('localhost', 27017)
        db = client.github
        return db.projects

    projects = get_repository()
    movies = []

    for project in projects.find({'done': True}).limit(1000):
        movie = dict()
        movie['id'] = project['id']
        movie['name'] = project['name']
        movie['date'] = project['created_at']
        movie['genres'] = project['library']
        movie['summary'] = project['readme_txt']
        movies.append(movie)

    return movies



def load_movies(max_movies=90000):
    def load_plot_summaries():
        plot_summaries_file = open("data/plot_summaries.fake.txt", "r", encoding="utf8")
        # plot_summaries_file = open("data/plot_summaries.txt", "r", encoding="utf8")
        plot_summaries = dict()
        for plot_summary_line in plot_summaries_file:
            plot_summary_data = plot_summary_line.split('\t')
            # Summaries structure
            # [0] Wikipedia movie ID
            # [1] Summary plot
            plot_summary = dict()
            plot_summary['id'] = plot_summary_data[0]
            plot_summary['summary'] = plot_summary_data[1]

            plot_summaries[plot_summary['id']] = plot_summary

        return plot_summaries

    print("Cargando datos de peliculas...")

    plot_summaries = load_plot_summaries()

    metadata_file = open("data/movie.metadata.fake.tsv", encoding="utf8")
    # metadata_file = open("data/movie.metadata.tsv", "r", encoding="utf8")
    movies = []

    movies_count = 0

    for metadata_line in metadata_file:
        movie_metadata = metadata_line.split('\t')

        # Metadata structure
        # [0] Wikipedia movie ID
        # [1] Freebase movie ID
        # [2] Movie name
        # [3] Movie release date
        # [4] Movie box office revenue
        # [5] Movie runtime
        # [6] Movie languages (Freebase ID:name tuples)
        # [7] Movie countries (Freebase ID:name tuples)
        # [8] Movie genres (Freebase ID:name tuples)

        id = movie_metadata[0]

        # Añadimos la pelicula solo si tiene sinopsis, incluimos una lista con las claves de los generos
        if (id in plot_summaries) & (movies_count < max_movies):
            movies_count += 1
            movie = dict()
            movie['id'] = id
            movie['name'] = movie_metadata[2]
            movie['date'] = movie_metadata[3]
            movie['genres'] = list(json.loads(movie_metadata[8].replace("\"\"", "\"").replace("\"{", "{").replace("}\"", "}")).values())
            movie['summary'] = plot_summaries[id].get('summary')
            movies.append(movie)

    print("Número de películas cargadas:", len(movies))
    
    return movies


# Usando la funcíon anterior cargamos los datos, dado que el proceso es bastante pesado solo cargaremos un subconjunto de las mismas

# In[3]:

t0 = datetime.datetime.now()
movies = load_movies(1000)
print("Time load_movies {0}".format(get_time_elapsed(t0)))


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

def get_global_texts(movies):
    print("Extrayendo palabras de los textos...")
    global_texts = []

    [global_texts.append(get_words(movie['summary'])) for movie in movies]

    return global_texts


# Y la ejecutamos, tendremos una lista de listas, en la que para cada película tendremos las palabras que definen su sinopsis

# In[5]:

t0 = datetime.datetime.now()
global_texts = get_global_texts(movies)
print("Time get_global_texts {0}".format(get_time_elapsed(t0)))


# A modo de ejemplo, mostramos las 5 primeras entradas de la primera pelicula

# In[6]:

global_texts[0][:5]


# # Creación del diccionaro
# 
# El diccionario está formado por la concatenación de todas las palabras que aparecen en alguna sinopsis (modo texto) de alguna de las peliculas. Básicamente esta función mapea cada palabra única con su identificador. Es decir, si tenemos N palabras, lo que conseguiremos al final es que cada película sea representada mediante un vector en un  espacio de N dimensiones.
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

t0 = datetime.datetime.now()
dictionary = corpora.Dictionary(global_texts)
dictionary
print("Time corpora.Dictionary {0}".format(get_time_elapsed(t0)))


t0 = datetime.datetime.now()

# # Creación del Corpus
# 
# Crearemos un corpus con la colección de todos los resúmenes previamente pre-procesados y transformados usando el diccionario.

# In[8]:

new_movie = get_words("A woman is a murder")
new_movie_corpus = dictionary.doc2bow(new_movie)

# out[3]: [(2, 1), (15, 1)]
corpus = [dictionary.doc2bow(text) for text in global_texts]


# A modo de ejemplo, mostramos las 5 primeras entradas de la primera pelicula

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
# Limita el numero de terminos, por supuesto tiene que ver con el tamaño de la muestra, mientras más peliculas tengamos mas terminos tendremos y por tanto la reduccion seria mayor, estamos clusterizando las peliculas en TOTAL_TOPICOS_LSA clusters
# 
# * **SIMILARITY_THRESHOLD** Umbral de similitud que se debe superar para que dos películas se consideren similares
# 
# 
# Damos valor a estos parámetros, como el número de películas que vamos a usar en el ejemplo es pequeño, vamos a ajustar el umbral a solo 0.4.

# In[11]:

TOTAL_LSA_TOPICS = 5
SIMILARITY_THRESHOLD = 0.6
GENRE_COINCIDENCE_RATE = 0.01


# Em corpus tenemos, para cada movie, una lista con sus palabras y el tfidf de cada una.

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


def show_similarities(doc, genres):
    def genre_score(genres_to_compare):
        common_genres = len(set(genres).intersection(genres_to_compare))
        return common_genres * GENRE_COINCIDENCE_RATE

    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    # print(vec_lsi)
    similarities = similarity_matrix[vec_lsi]
    # print(list(enumerate(sims)))
    similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
    for sim in similarities:
        movie = movies[int(sim[0])]
        similarity_score =  sim[1] + genre_score(movie["genres"])
        if similarity_score>SIMILARITY_THRESHOLD:
            print("Movie: {0}: - Similarity: {1}".format(movie["name"],similarity_score))


# show_similarities("A murder woman", ["xx"])
show_similarities("A murder woman", ["Mystery","Drama", "Biographical film"])


exit(0)



# # Creación del modelo de similitud
# 
# Una vez generados todos los artefactos (diccionario, corpus, modelo, etc), pasamos a crear la lista de peliculas similares,

# La similitud la vamos a calcular recorriendo la lista de peliculas, en realidad su corpus TfIdf, y usando la matriz similitudes llamaremos a una función auxiliar que determina cuales son las películas que superan el umbral de similitud, esto lo haremos en una función auxiliar, una vez calculada esta lista, la insertaremos en la información de la película que está siendo analizada.

# In[13]:

def create_similitary_model(movies, corpus_tfidf, lsi, similarity_matrix, similarity_threshold):
    print("Creando enlaces de similitud entre películas")
    for i, doc in enumerate(corpus_tfidf):
        vec_lsi = lsi[doc]
        similarity_index = similarity_matrix[vec_lsi]
        movies[i]['similars'] = search_similitary_movies(movies[i], movies, similarity_index, similarity_threshold)


# Definimos la función auxiliar que dada una película, nos determina la lista de peliculas que superan el umbral de similitud, como variante vamos a bonificar las películas que sean del mismo género que la analizada, de este modo, además del indice de similitud calcularo anteriormente, vamos a sumar 0.1 por cada genero coincidente. Así para dos peliculas con una similutud de digamos 0.34 y un genero en común, obtendriamos una similitud de 0.44. Para cada pelicula que supere el umbral, almacenaremos el índice dentro de la matriz de peliculas (para localizarla posteriormente), el grado de similitud y los generos en los que coinciden. 

# In[14]:

def search_similitary_movies(movie, movies, similarity_index, similarity_threshold):
    def compare_genres(movie1, movie2):
        return len(set(movie1['genres']).intersection(movie2['genres']))

    similar_movies = []
    for j, elemento in enumerate(movies):
        if (movies[j]['id'] != movie['id']):  # Para que no se incluya a si misma
            common_genres = compare_genres(movie, movies[j])
            similarity = similarity_index[j] + common_genres / 1000
            if (similarity > similarity_threshold):
                similar_movies.append((j, similarity, common_genres ))

    return sorted(similar_movies, key=lambda item: -item[1])


# A continuación ejecutamos la función.

# In[15]:

create_similitary_model(movies, corpus_tfidf, lsi, similarity_matrix, SIMILARITY_THRESHOLD)


# # Prueba del algoritmo
# 
# Finalmenete mostramos la lista de películas con sus similitudes. Primero definimos una función auxiliar para sacar por pantalla los datos de una película dado su indice en la lista de películas

# In[16]:

def show_coincidences(movie_index):
    print("\n====================================================================================")
    print("Pelicula Id = ", movies[movie_index]['name'], "  Index=", movie_index, movies[movie_index]['name'])
    for j, similar in enumerate(movies[movie_index]['similars']):
        similar_movie_index, similarity, common_genres = similar
        print("Similitud:", similarity, ", Generos comunes (id): ", common_genres, ", Titulo:", movies[similar_movie_index]['name'])


# Y la ejecutamos para cada una de las películas de nuestra lista

# In[17]:

# show_coincidences(3)




print("Time rest {0}".format(get_time_elapsed(t0)))



show_similarities("A murder woman", ["Mystery","Drama", "Biographical film"])