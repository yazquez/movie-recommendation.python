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
    movies = dict()

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

            movies[id] = movie

    return movies


def process_movies(movies):
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
    for movie in movies.values():
        words = get_words(movie['summary'])
        movie['text'] = " ".join(words)
        global_texts.append(words)

    return (movies, global_texts)


def create_dictionary(text_list):
    print("Creación del diccionario global")
    return corpora.Dictionary(text_list)


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
    #numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=50000)
    #svd = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv=False)

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=total_lsa_topics)

    index = similarities.MatrixSimilarity(lsi[corpus_tfidf])

    return lsi, index



def crearCodigosPeliculas(peliculas):
    codigosPeliculas = []
    for i, elemento in enumerate(peliculas):
        pelicula = peliculas[elemento]
        codigosPeliculas.append(pelicula['id'])
    return codigosPeliculas


def crearModeloSimilitud(movies, corpus_tfidf, lsi, index, similar_threshold):
    codigosPeliculas = crearCodigosPeliculas(movies)
    print("Creando enlaces de similitud entre películas")
    for i, doc in enumerate(corpus_tfidf):
        print("============================")
        peliculaI = movies[codigosPeliculas[i]]
        print("Pelicula I = ", i, "  ", peliculaI['id'], "  ", peliculaI['summary'])

        vec_lsi = lsi[doc]
        # print(vec_lsi)
        indice_similitud = index[vec_lsi]
        similares = []
        for j, elemento in enumerate(movies):
            s = indice_similitud[j]
            if (s > similar_threshold) & (i != j):  # i!=j para que no se incluya a si misma
                peliculaJ = movies[codigosPeliculas[j]]
                similares.append((codigosPeliculas[j], s))

                print("   Similitud: ", s, "   ==> Pelicula J = ", j, "  ", peliculaJ['id'], "  ", peliculaJ['summary'])

            similares = sorted(similares, key=lambda item: -item[1])

            peliculaI['similares'] = similares
            peliculaI['totalSimilares'] = len(similares)



### Valores clave para controlar el proceso
TOTAL_LSA_TOPICS = 50
# Limita el numero de terminos, por supuesto tiene que ver con el tamaño de la mustra, mientras más peliculas tengamos mas terminos tendremos y por tanto la reduccion seria mayor
# estamos clusterizando las peliculas en TOTAL_TOPICOS_LSA cluster
SIMILAR_THRESHOLD = 0.2

movies = load_movies(20)
movies, global_texts = process_movies(movies)
dictionary = create_dictionary(global_texts)
corpus = create_corpus(dictionary)
corpus_tfidf = create_tfidf(corpus)
(lsi, index) = create_lsi_model(corpus_tfidf, dictionary, TOTAL_LSA_TOPICS)

crearModeloSimilitud(movies, corpus_tfidf, lsi, index, SIMILAR_THRESHOLD)

exit()

print(dictionary)

print(global_texts)

for m in movies.values():
    print(m['name'])
