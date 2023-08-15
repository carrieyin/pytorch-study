import os

import torch
from torchtext import vocab

vocabulary = vocab.Vocab(counter={}, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")