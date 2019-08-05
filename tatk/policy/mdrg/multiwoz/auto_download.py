import os
import requests
import zipfile

model_path = os.path.join(os.path.dirname(__file__), 'model')
data_path = os.path.join(os.path.dirname(__file__), 'data')
db_path = os.path.join(os.path.dirname(__file__), 'db')

urls =  {model_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_model.zip', data_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_data.zip', db_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_db.zip'}

for path in [model_path, data_path, db_path]:
    if not os.path.exists(path):
        file_url = urls[path]
        print("Downloading from %d", file_url)
        r = requests.get(file_url)
        with open('tmp.zip', 'wb') as file:
            file.write(r.content)
        zip_file = zipfile.ZipFile('tmp.zip')
        for names in zip_file.namelist():
            zip_file.extract(names)
        zip_file.close()
