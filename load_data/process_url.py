# Este script recorre la colección de proyectos y los clona en un directorio local, verifica que el proyecto aún no haya sido procesado y que el directorio de descarga no exista, este último control se hace por si es neceario reiniciar la descarga en algún momento.

import os
import uuid
import re

ROOT_PATH = "d:/tfm/tmp"
CLONE_COMMAND = "git clone {0} {1}"

def proccess_url(git_url):
    def process_python_file(project, file_path):
        def add_to_list(item):
            if not item in library:
                library.append(item)

        print("Processing file", file_path)
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

#project = proccess_url("https://github.com/yazquez/movie-recommendation.python.git")
project = proccess_url("https://github.com/yazquez/programmable-agents_tensorflow.git")
print(project['library'])







# https://github.com/yazquez/spell-checker.scala.git