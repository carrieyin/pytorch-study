"""
预处理NMT 法-英 数据集数据

"""
import os

import requests


def extract_content(file_path):
    pass


def download_extract():
    file_path = r"../../../dataset/nmt/frg-eng.txt"
    if os.path.exists(file_path):
        return extract_content(file_path)

    data_url = "http://www.manythings.org/anki/fra-eng.zip"
    zip_file = requests.get(data_url)
    with open(file_path, 'a') as f:
        f.write(zip_file.content)
        f.close()
def preprocess():
    pass