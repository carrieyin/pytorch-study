from collections import Counter

from torchtext import vocab

# 从list构建词汇表
def to_cab(mylist):  # 修改函数参数的名称
    v = vocab.vocab(Counter(mylist))
    return v


def show_word_in_cab_index(va, mylist):
    for item in mylist:
        print(va.lookup_indices([item]))
        #print(va[item])


if __name__ == '__main__':
    cablist = ['dick', 'tracy', 'is', 'one', 'of', 'our', 'dick', 'tracy', 'is']
    va = to_cab(cablist)
    print(va)

    # print(va['one'])
    # print(len(va))

    show_word_in_cab_index(va, cablist)
