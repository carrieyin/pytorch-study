test_data = [['apple banana'], ['banana'], ['orange grape']]
# max_len = 500
def etlArrayWithLength(arr, maxLen):
    if len(arr) >= maxLen:
        return arr[:maxLen]
    return arr + [0] * (maxLen - len(arr))


#print(etlArrayWithLength(["abd"], 3))
etl_data = [etlArrayWithLength(words, 2) for words in test_data]
# for words in test_data:
#     etl_data.append(etlArrayWithLength(words, 3))

#print(etl_data)
#torch.tensor(etl_data)
