from data_preprocess import *
import torch.utils.data as Data

def show_loader_data(loader):
    for index, (token, target) in enumerate(loader):
        print(token)
        print(target)


if __name__ == '__main__':
    # 读取数据
    train_data = [['"dick tracy" is one of our"', 1],
    ['arguably this is a  the )', 1],
    ["i don't  just to warn anyone ", 0]]
    #train_data = read_imdb('train')

    # 预处理数据
    processed_data = preprocess(train_data, 10)
    print(processed_data)
    #print(processed_data[0].shape)

    # 构建dateset和dataloader
    bathsize = 2
    train_dataset = Data.TensorDataset(*processed_data)
    train_dataloader = Data.DataLoader(train_dataset, bathsize)
    print('load sucess')
    show_loader_data(train_dataloader)