import torch
import torch.utils.data as Data
from data_preprocess import preprocess, read_imdb


class ImdbLoader(object):
    def __init__(self, set_name='train', batch_size='64'):
        super(ImdbLoader, self).__init__()
        self.data_set = set_name
        self.batch_size = batch_size

    def get_data_loader(self):
        # train_data = [['"dick tracy" is one of our"', 1],
        #               ['arguably this is a  the )', 1],
        #               ["i don't  just to warn anyone ", 0]]
        train_data = read_imdb(self.data_set)
        data = preprocess(train_data)
        #print(data)
        data_set = Data.TensorDataset(*data)
        data_loader = Data.DataLoader(data_set, self.batch_size)
        return data_loader


if __name__ == '__main__':
    imdb_load = ImdbLoader('train', batch_size=2)
    data_iter = data_load = imdb_load.get_data_loader()
    for x, y in data_iter:
        print('x.shape', x.shape, 'y.shape', y.shape)