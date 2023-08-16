import pickle

import torch
from torch import nn

from pytorch.src.rnn.birnn_model import BiRNN
from pytorch.src.rnn.config import device
from pytorch.src.rnn.data_load import ImdbLoader
from pytorch.src.rnn.pretrain_glove import getGlove, load_pretrained_embedding


def test(imdb_model, test_batch_size):
    imdb_model.eval()
    imdb_model = imdb_model.to(device)
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
    # test_data = read_imdb('train')
    # tokenized_data = get_tokenized(test_data)
    # # print(tokenized_data)
    #
    # # 2. 获取分词词汇表(vocab类)
    # vo = get_vocab(tokenized_data)
    vo = pickle.load(open("../../resources/model_save/vocabulary.pkl", 'rb'))

    embed_size, hidden_size, num_layers = 100, 100, 2
    net = BiRNN(vo, embed_size, hidden_size, num_layers)
    glove_vab = getGlove()
    net.embedding.weight.data.copy_(load_pretrained_embedding(vo.get_itos(), glove_vab))
    net.embedding.weight.requires_grad = False
    net.load_state_dict(torch.load("../../resources/model_save/imdb_net.pkl"))
    test(net, 64)