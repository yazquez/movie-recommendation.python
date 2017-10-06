import os
from pymongo import MongoClient
import re
import sys

ROOT_PATH = "d:/tmp/tfm_to_process/{0}"

def get_repository():
    client = MongoClient('localhost', 27017)
    db = client.github
    return db.projects

def process_python_file2(project, file_path):
    library = project['library']
    with open(file_path, 'r') as f:
        for line in f:
            if line.find("import") != -1:
                library.append(line.strip())
    project['library'] = library
    project['done'] = True
    projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


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
                if match.group(1) != None:
                    add_to_list(match.group(1))
                else:
                    add_to_list(match.group(2))
    project['library'] = library
    projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)


def process_readme_file(project, file_path):
    with open(file_path, 'r') as f:
        project['readme_txt'] = f.read()
        projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)

projects = get_repository()

for project in projects.find({'done':False}):
    try:
        print(".",end="")
        path = ROOT_PATH.format(project["id"])
        if os.path.isdir(path):
            print("Processing project: ", project["id"])
            project['done'] = True
            projects.update({'_id': project['_id']}, {"$set": project}, upsert=False)
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
    except:
        print("Error procesing project {0} [{1}] - {2}".format(project['id'], project['name'], sys.exc_info()[0]))