import os

import torch

raw_file_path = r"..\..\..\dataset\nmt\fra.txt"
raw_zip_path = r"..\..\..\dataset\nmt\fra-eng.zip"
URL = "http://www.manythings.org/anki/fra-eng.zip"
num_layers, hidden_size, batch_size, num_steps = 2, 300, 64, 10
embed_size = 100
num_examples = 800

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")