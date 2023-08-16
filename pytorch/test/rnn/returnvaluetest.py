def get_values():
    return 1, 2


if __name__ == '__main__':
    value = get_values()
    print(value)

    value1 = get_values()
    print(*value1)

