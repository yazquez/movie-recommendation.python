import nltk
import gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities


def load_movies(max_movies=90000):
    def load_plot_summaries():
        plot_summaries_file = open("data/plot_summaries.fake.txt", "r")
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

    plot_summaries = load_plot_summaries()

    metadata_file = open("data/movie.metadata.fake.tsv", "r")
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

        # Añadimos la pelicula solo si tiene sinopsis
        if (id in plot_summaries) & (movies_count < max_movies):
            movies_count += 1
            movie = dict()
            movie['id'] = id
            movie['name'] = movie_metadata[2]
            movie['date'] = movie_metadata[3]
            movie['summary'] = plot_summaries[id].get('summary')

            movies.append(movie)

    return movies


def get_global_texts(movies):
    def get_words(text):
        def add_word(word):
            word = word.lower()
            if word not in stop_words:
                words.append(stemmer.stem(word))

        words = []
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer("english")

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

    global_texts = []
    [global_texts.append(get_words(movie['summary'])) for movie in movies]
    return global_texts


def create_dictionary(global_texts):
    print("Creación del diccionario global")
    return corpora.Dictionary(global_texts)


def create_corpus(dictionary):
    print("Creación del corpus global con los resúmenes de todas las películas")
    return [dictionary.doc2bow(text) for text in global_texts]


def create_tfidf(corpus):
    print("Creación del Modelo Espacio-Vector Tf-Idf")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


def create_lsi_model(corpus_tfidf, dictionary, total_lsa_topics):
    print("Creación del modelo LSA: Latent Semantic Analysis")
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=total_lsa_topics)
    similarity_matrix = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    return lsi, similarity_matrix


def crearModeloSimilitud(movies, corpus_tfidf, lsi, similarity_matrix, similarity_threshold):
    print("Creando enlaces de similitud entre películas")
    for i, doc in enumerate(corpus_tfidf):
        print("====================================================================================")
        print("Pelicula I = ", i, "  ", movies[i]['id'], "  ", movies[i]['summary'])

        vec_lsi = lsi[doc]
        similarity_index = similarity_matrix[vec_lsi]

        # print("doc>>              ", doc)
        # print("vec_lsi>>          ", vec_lsi)
        # print("indice_similitud>> ", indice_similitud)

        similar_movies = create_similitary_model(movies[i], movies, similarity_index, similarity_threshold)

        [print(similarity, movie['summary']) for (movie, similarity) in similar_movies]


def create_similitary_model(movie, movies, similarity_index, similarity_threshold):
    similar_movies = []

    for j, elemento in enumerate(movies):
        similarity = similarity_index[j]
        if (similarity > similarity_threshold) & (movies[j]['id'] != movie['id']):  # Para que e incluya a si misma
            similar_movies.append((movies[j], similarity))

    return sorted(similar_movies, key=lambda item: -item[1])


### Valores clave para controlar el proceso
TOTAL_LSA_TOPICS = 50
# Limita el numero de terminos, por supuesto tiene que ver con el tamaño de la mustra, mientras más peliculas tengamos mas terminos tendremos y por tanto la reduccion seria mayor
# estamos clusterizando las peliculas en TOTAL_TOPICOS_LSA cluster
SIMILARITY_THRESHOLD = 0.2

movies = load_movies(20)

# Lista con la lista de palabras "diferentes" para cada sinopsis
global_texts = get_global_texts(movies)

# El diccionario está formado por la concatenación de todas las palabras que aparecen en alguna sinopsis (modo texto) de alguna de las peliculas
# Básicamente esta función mapea cada palabra única con su identificador. Es decir, si tenemos N palabras, lo que conseguiremos al final es que cada película sea
# representada mediante un vector en un  espacio de N dimensiones
dictionary = create_dictionary(global_texts)

# Crearemos un corpus con la colección de todos los resúmenes previamente pre-procesados y transformados usando el diccionario
corpus = create_corpus(dictionary)

corpus_tfidf = create_tfidf(corpus)

(lsi, similarity_matrix) = create_lsi_model(corpus_tfidf, dictionary, TOTAL_LSA_TOPICS)

crearModeloSimilitud(movies, corpus_tfidf, lsi, similarity_matrix, SIMILARITY_THRESHOLD)

exit()

print(dictionary)

print(global_texts)

[print(m['name']) for m in movies]
