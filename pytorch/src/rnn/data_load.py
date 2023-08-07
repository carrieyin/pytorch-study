import torch.utils.data as Data
from data_preprocess import preprocess, read_imdb


class ImdbLoader(object):
    def __init__(self, set_name='train', batch_size='64'):
        super(ImdbLoader, self).__init__()
        self.data_set = set_name
        self.batch_size = batch_size

    def get_data_loader(self):
        train_data = read_imdb(self.data_set)
        data = preprocess(train_data)
        data_set = Data.TensorDataset(data)
        data_loader = Data.dataloader(data_set, self.bathsize)
        return data_loader


if __name__ == '__main__':
    imdb_load = ImdbLoader('train', 2)
    data_load = imdb_load.get_data_loader()