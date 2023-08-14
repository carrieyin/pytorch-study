import torch
from torch import nn

from pytorch.src.rnn.config import device
from pytorch.src.rnn.data_load import ImdbLoader


def test(imdb_model, test_batch_size):
    imdb_model.eval()
    loader = ImdbLoader('test', test_batch_size)
    data_loader = loader.get_data_loader()
    with torch.no_grad():
        for idx, (inputs, target) in enumerate(data_loader):
            target = target.to(device)
            inputs = inputs.to(device)
            #print(inputs)
            output = imdb_model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

            predict = torch.max(output, dim=-1, keepdim=False)[-1]
            correct = predict.eq(target.data).sum()
            acc = 100. * predict.eq(target.data).cpu().numpy().mean()
            print('idx: {} loss : {}, accurate: {}/{} {:.2f}'.format(idx,  loss, correct, target.size(0), acc))


if __name__ == '__main__':
    pass