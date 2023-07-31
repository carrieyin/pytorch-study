
def to_up(my_list):
    uppercase_list = [[x.upper() for x in nested_list] for nested_list in my_list]
    return uppercase_list


if __name__ == '__main__':
    mylist =[['"dick', 'tracy"', 'is', 'one', 'of', 'our"'],
             ['arguably', 'this', 'is', 'a', '', 'the', ')'],
    ['i', "don't", '', 'just', 'to', 'warn', 'anyone', '']]
    print((to_up(mylist)))

