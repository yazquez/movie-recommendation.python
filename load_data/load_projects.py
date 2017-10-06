# Usando la API Rest de Github descarga la infomación de todos los proyectos que cumplen una serie de criterios, estos son los siguientes:
#   - Identificados como proyectos Python
#   - Creado entre el 1 de Enero de 2012 y hasta la fecha actual
#   - Con un mínimo de 10 estrellas
#
# Toda la información es almacenada en MongoDB concretamante en una colección llamada prohects de na BD llamada github, la información almacenada es la siguiente:
#             'id': github_project["id"],
#             'name': github_project["name"],
#             'full_name': github_project["full_name"],
#             'created_at': github_project["id"],
#             'git_url': github_project["git_url"],
#             'description': github_project["description"],
#             'language': github_project["language"],
#             'stars': github_project["stargazers_count"],
#             'done': Se usa a nivel interno, indica si un proyecto está procesado, esto es identificados las librerias que usa y el fichero README cargado, en los puntos siguientes se detallan estos dos datos
#             'readme_txt': Almacenará el texto de los ficheros README de los proyectos, será la información base para el sistema de recomendación
#             'library': Lista con las librerias que el proyecto usa, en principio estará vacía y se alimentará cuando se procese el proyecto
#             'raw_data': toda la información del proyecto sin procesar por si más adelante es necesario algún dato que en estos momentos no parece relevante
# Cuando se usa la API de Github en su modo básico, esto es



import requests
import json
from time import sleep
from pymongo import MongoClient
from datetime import date, timedelta

MIN_STARTS = 10
START_DATE = date(2012, 1, 1)
END_DATE = date(2017, 8, 28)
URL_PATTERN = 'https://api.github.com/search/repositories?q=language:Python+created:{0}+stars:>={1}&type=Repositories'

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.github
    return db.projects


def insert_project(github_project):
    if projects.find_one({"id": github_project["id"]}):
        print("Project {0} is already included in the repository".format(github_project["name"]))
    else:
        project = {
            'id': github_project["id"],
            'name': github_project["name"],
            'full_name': github_project["full_name"],
            'created_at': github_project["id"],
            'git_url': github_project["git_url"],
            'description': github_project["description"],
            'language': github_project["language"],
            'stars': github_project["stargazers_count"],
            'done': False,
            'readme_txt': "",
            'library': [],
            'raw_data': github_project
        }
        projects.insert(project)


def get_projects_by_date(date):
    print("Processing date ", date)
    url_pattern = URL_PATTERN
    url = url_pattern.format(date, MIN_STARTS)
    response = requests.get(url)
    if (response.ok):
        response_data = json.loads(response.content.decode('utf-8'))['items']
        for project in response_data:
            insert_project(project)
    else:
        response.raise_for_status()


projects = get_repository()

for project_create_at in [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]:
    try:
        get_projects_by_date(project_create_at)
    except:
        print(">> Reached call limit, waiting 61 seconds...")
        sleep(61)
        get_projects_by_date(project_create_at)
