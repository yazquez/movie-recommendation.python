# Este script recorre la colección de proyectos y los clona en un directorio local, verifica que el proyecto aún no haya sido procesado y que el directorio de descarga no exista, este último control se hace por si es neceario reiniciar la descarga en algún momento.

import os
from pymongo import MongoClient

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.tfm
    return db.projects

ROOT_PATH = "d:/tfm/tfm_cloned"
CLONE_COMMAND = "git clone {0} {1}"

os.chdir(ROOT_PATH)

projects = get_repository()

for i in range(0,1000):
    try:
        for project in projects.find({'project_processed':False}):
            path = ROOT_PATH + "/" + str(project["id"])
            if not os.path.isdir(path):
                print('cloning project')
                os.system(CLONE_COMMAND.format(project["git_url"], project["id"]))
    except:
        pass


