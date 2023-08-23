"""
预处理NMT 法-英 数据集数据

"""
import os
import zipfile

from pytorch.src.seq_2_seq.config import raw_file_path, raw_zip_path


def extract_content():
    # folder_name = os.path.dirname(file_path)
    # file_name = os.path.basename(file_path)

    if not os.path.exists(raw_file_path):
        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file_path))

    with open(raw_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        return content


def download_extract():
    # file_path = r"../../../dataset/nmt/frg-eng.txt"
    # if os.path.exists(file_path):
    #     return extract_content(file_path)
    #
    # data_url = "http://www.manythings.org/anki/fra-eng.zip"
    # zip_path = r"../../../dataset/nmt/frg-eng.zip"
    # zip_file = requests.get(data_url, stream=True, verify=True)
    # with open(zip_path, 'wb') as f:
    #     f.write(zip_file.content)

    url = 'http://www.manythings.org/anki/fra-eng.zip'
    save_path = r"..\..\..\dataset\nmt"

    # 发起GET请求并下载文件
    # response = requests.get(url)

    # 保存文件
    # response = requests.get(url, stream=True)
    # with open(save_path, 'wb') as file:
    #     file.write(response.content)
    #
    # print("文件已下载并保存到", save_path)

def preprocess_nmt(text):
    """预处理“英语－法语”数据集
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        print(parts)
        if len(parts) == 3:
            target.append(parts[0].split(' '))
            source.append(parts[1].split(' '))
    return source, target


if __name__ == '__main__':
    content = extract_content()
    data = preprocess_nmt(content)
    #print(data)
    token_data = tokenize_nmt(data, 600)
    print(token_data)

