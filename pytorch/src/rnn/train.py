import torch.optim
from torch import nn
from torchtext.vocab import vocab

from pytorch.src.rnn.birnn_model import BiRNN
from pytorch.src.rnn.data_load import ImdbLoader
from pytorch.src.rnn.data_preprocess import get_tokenized, get_vocab
from pytorch.src.rnn.pretrain_glove import load_pretrained_embedding, getGlove

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch, imdb_model, lr, train_batch_size):
    imdb_model_device = imdb_model.to(device)
    # 过滤掉不需要计算梯度的embedding的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.require_grad, imdb_model_device.parameters()), lr=lr)
    loader = ImdbLoader('train', train_batch_size)
    data_loader = loader.get_data_loader()

    for idx, (target, inputs, input_len) in enumerate(data_loader):
        target = target.to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        output = imdb_model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            predict = torch.max(output, dim=-1, keepdim=False)[-1]
            acc = predict.eq(target.data).cpu().numpy().mean() * 100
            print('train Epoch:{} processed:[{} / {} ({.0f}%) Loss: {.6f}, ACC: {.6f}]'.format(epoch, idx*len(input),
                                                                      len(data_loader.dataset),
                                                                      100. * idx / len(data_loader),
                                                                      loss.item(),
                                                                      acc))
            torch.save(imdb_model.state_dict(), '../../resources/model_save/imdb_net.pkl')
            torch.save(optimizer.state_dict(), '../../resources/model_save/imdb_optimizer.pkl')


def test(imdb_model, test_batch_size):
    imdb_model.eval()
    loader = ImdbLoader('test', test_batch_size)
    data_loader = loader.get_data_loader()
    with torch.no_grad():
        for idx, (target, inputs, input_len) in enumerate(loader):
            target = target.to(device)
            inputs = inputs.to(device)
            output = imdb_model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

            predict = torch.max(output, dim=-1, keepdim=False)[-1]
            correct = predict.eq(target.data).sum()
            acc = 100. * predict.eq(target.data).cpu().numpy().mean()
            print('idx: {} loss : {}, accurate: {}/{} {.2f}'.format(idx,  loss, correct, target.size(0), acc))


if __name__ == '__main__':
    train_data = [['"dick tracy" is one of our"', 1],
                  ['arguably this is a  the )', 1],
                  ["i don't  just to warn anyone ", 0]]
    # 1.获取分词数据形式[[评论1分词1,评论1分词n], [评论i分词1， 评论i分词m].....]
    tokenized_data = get_tokenized(train_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)

    # 3. 构建模型
    embed_size, hidden_size, num_layers = 100, 100, 2
    net = BiRNN(vo, embed_size, hidden_size, num_layers)
    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, getGlove))
    net.embedding.weight.requires_grad = False

    # 4. 训练
    learning_rate = 0.01
    train_batch = 64
    test_batch = 5000

    for i in range(10):
        train(i, net, learning_rate, train_batch)

    # 5. 评估
    test(net, test_batch)
