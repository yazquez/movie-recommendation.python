# import os
# from pymongo import MongoClient
# import re
# import sys
# from collections import defaultdict

from pymongo import MongoClient
# import nltk
# import json
# import itertools
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from gensim import corpora, models, similarities, matutils
# import numpy as np
#
# import os
# import uuid
# import re


ROOT_PATH = "d:/tfm/tmp/{0}"

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.tfm
    return db.projects

# def process_library(project):
#     libraries = project['library']
#     for library in libraries:
#         if "," in library:
#             print('\t',library)
#     # project['library'] = get_words(project['readme_txt'])
#     # projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


# client = MongoClient('localhost', 27017)
# db = client.github

repository_projects = get_repository()


# def process_library(libraries):
#     for library in libraries:
#         if "_" in library:
#             print(library)
#
# for project in repository_projects.find():
#     #print("Processing 'readme' of", project["name"])
#     try:
#         project['library'] = process_library(project['library'])
#     except:
#         pass


# print("project:",repository_projects.find_one({'id': 101110284}))
# print(repository_projects.count())
# x = repository_projects.delete_many({'id':100655179})
# print(x.deleted_count)
# print(repository_projects.count())

i = 0

for project in repository_projects.find():
    if len(project['library']) == 0 or len(project['readme_words']) == 0:
        print(project["id"])
        #repository_projects.delete_one({'id': project["id"]})
        i+=1

print('total', i)


#repository_projects.delete_many({'id': project["id"]})
# 98326571
# 98531792
# 98595746
# 99404030
# 99857669
# 99971231
# 100016532
# 100655179
# 101033179
# 101110284

def process_library(libraries):
    def add_library(library):
        if library not in libraries_processed and not library.startswith('.'):
            libraries_processed.append(library)
    libraries_processed = []
    for library in libraries:
        if "," in library:
            [add_library(l) for l in library.split(',')]
        else:
            add_library(library)

    return libraries_processed

for project in repository_projects.find({'data_processed': False}):
    print("Processing 'readme' of", project["name"])
    try:
        project['data_processed'] = True
        project['readme_words'] = get_words(project['readme_txt'])
        project['library'] = process_library(process_library)

        repository_projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)
    except:
        pass