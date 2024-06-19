
def split_word(word):
    # return the three continuouse letter string for the word string
    dict_re={}
    for char in word:
        if len(word)>=3:
            for i in range(len(word)-1):
              dict_re.append(word[i:i+3])
        else:
            dict_re.append(word)
    return dict_re
# from the dictionary to get the word lis
def get_word_list(dict):
    word_list=[]
    for key in dict:
        word_list.append(split_word(key))
        return word_list


word_list=get_word_list(dict)
all_t