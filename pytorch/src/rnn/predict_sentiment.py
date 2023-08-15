import torch

from pytorch.src.rnn import config
from pytorch.src.rnn.birnn_model import BiRNN
from pytorch.src.rnn.data_preprocess import read_imdb, get_tokenized, get_vocab
from pytorch.src.rnn.pretrain_glove import getGlove, load_pretrained_embedding


def predict_sentiment(net, vocabulary, sentence):
    sentence_index_list = torch.tensor([vocabulary[word] for word in sentence], device=config.device)
    label = torch.argmax(net(sentence_index_list.view(1, -1)), dim=1)
    return 'positive' if label.item() == 1 else 'negative'


if __name__ == '__main__':
    test_data = read_imdb('train')
    tokenized_data = get_tokenized(test_data)
    # print(tokenized_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)

    embed_size, hidden_size, num_layers = 100, 100, 2
    net = BiRNN(vo, embed_size, hidden_size, num_layers)
    glove_vab = getGlove()
    net.embedding.weight.data.copy_(load_pretrained_embedding(vo.get_itos(), glove_vab))
    net.embedding.weight.requires_grad = False
    net.load_state_dict(torch.load("../../resources/model_save/imdb_net.pkl"))

    label = predict_sentiment(net, vo, ['this', 'movie', 'is', 'so', 'good'])
    print(label)
    label = predict_sentiment(net, vo, ['this', 'movie', 'is', 'so', 'bad'])
    print(label)
