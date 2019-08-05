import os
import requests
import zipfile

model_path = os.path.join(os.path.dirname(__file__), 'model')
data_path = os.path.join(os.path.dirname(__file__), 'data')
db_path = os.path.join(os.path.dirname(__file__), 'db')

urls =  {model_path: '', data_path: '', db_path: ''}

for path in [model_path, data_path, db_path]:
    if not os.path.exists(path):
        url = requests.get(urls[path])
        r = requests.get(url)
        with open('tmp.zip', 'wb') as file:
            file.write(r.content)
        os.makedirs(path)
        zipfile = zipfile.ZipFile('tmp.zip')
        for names in zip_file.namelist():
            zip_file.extract(names, path=path)
        zip_file.close()
