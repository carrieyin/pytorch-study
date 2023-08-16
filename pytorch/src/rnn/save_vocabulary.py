import pickle

from pytorch.src.rnn.data_preprocess import read_imdb, get_tokenized, get_vocab


def save_vocabu():
    train_data = read_imdb('train')
    # 1.获取分词数据形式[[评论1分词1,评论1分词n], [评论i分词1， 评论i分词m].....]
    tokenized_data = get_tokenized(train_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)
    size_vo = len(vo.get_itos())
    print('vo len is:', size_vo)
    pickle.dump(vo, open("../../resources/model_save/vocabulary.pkl", 'wb'))
    pick_vo = pickle.load(open("../../resources/model_save/vocabulary.pkl", 'rb'))
    print("load len is:", len(pick_vo.get_itos()))


if __name__ == '__main__':
    save_vocabu()