import os
curPath = os.getcwd()+os.sep
from collections import Counter
from operator import itemgetter
import pickle
import re
import numpy as np
import math

buckets = [(5, 15), (10, 20), (15, 25), (20, 30)]
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'

def loadData(directory):
    #统计每个桶的句子总数，并生成总字典
    assert os.path.isdir(directory)
    words_dict = {}
    word_count = Counter()
    for cur_dir,sub_dirs,files in os.walk(directory):
        for fileName in files:
            filePath = cur_dir + os.sep + fileName
            words_dict[fileName] = 0
            file = open(filePath,"r",encoding="utf-8")
            while True:
                line = file.readline()
                if line:
                    ask,answer = line.split()
                    word_count.update(ask)
                    word_count.update(answer)
                    words_dict[fileName] += 1
                else:
                    break
    word_count = sorted(word_count.items(),key=itemgetter(1,0),reverse=True)
    return word_count,words_dict

def dumpFile(word_count,words_dict):
    words_dict_file = open("./words_dict","wb")
    pickle.dump(words_dict,words_dict_file)
    word_count_file = open("./word_count","wb")
    words = [ word for word,number in word_count ]
    pickle.dump(words,word_count_file)

def load_dict():
    word_count_file = open("./word_count","rb")
    dictionary = pickle.load(word_count_file)
    # sentences_file = open("./words_dict","rb")
    # sentences = pickle.load(sentences_file)

    dictionary = [EOS,UNK,PAD,GO] + dictionary
    word_id,id_word = {},{}
    for idx,word in enumerate(dictionary):
        word_id[word] = idx
        id_word[idx] = word
    dim = len(dictionary)
    return dim,word_id,id_word

dim,word_id,id_word = load_dict()

def get_id_by_dict(sentence):
    _, word_id, _ = load_dict()
    words_id = []
    for word in sentence:
        id = word_id[word] if word in word_id.keys() else word_id['<unk>']
        words_id.append(id)
    return words_id

def get_buckets_by_id(directory,bucket_id):
    bucket_db = {}
    files = os.listdir(directory)
    for encoder_size,decoder_size in buckets:
        for file in files:
            if re.findall("{}_{}".format(encoder_size,decoder_size),file):
                bucket_db.update({file:directory+os.sep+file})
    if bucket_id < 0 or bucket_id > len(bucket_db):
        raise ValueError("the value of bucket_id is invalid :{}".format(bucket_id))
    return list(bucket_db.keys())[bucket_id]

def get_batch_data(batch_size,bucket_id,directory = curPath + "bucket_datas"):
    #随机从对应bucket文件中取一batch_size的问答对
    fileName = get_buckets_by_id(directory,bucket_id)
    for cur_dir, sub_dirs, files in os.walk(directory):
        for file in files:
            if re.findall(fileName, file):
                targetFile = directory+os.sep+file
                break
    assert os.path.isfile(targetFile)
    with open(targetFile,"r+",encoding="utf-8") as target:
        contents = target.readlines()
    data = []
    data_in = []
    for i in range(batch_size):
        random_number = int(np.floor(np.random.random() * len(contents)))
        content = contents[random_number].strip()
        ask = content.split()[0]
        answer = content.split()[1]
        data.append((ask,answer))
        data_in.append((answer,ask))
    return data,data_in

directory = curPath + "bucket_datas"
word_count, words_dict = loadData(directory)

def time(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret

def sentence_decode(sentence):
    if len(sentence) <= 0:
        return ""
    res = []
    for id in sentence:
        word = id_word[id]
        if word == EOS:
            break
        elif word not in [GO,PAD,UNK]:
            res.append(word)
    return "".join(res)

def main():
    #存储字典
    directory = curPath + "bucket_datas"
    word_count, words_dict = loadData(directory)
    dumpFile(word_count, words_dict)
    # 加载字典
    # load_dict()
    # res = get_buckets_by_id(directory,0)
    # print(res)
    # print(get_batch_data(10,0))

if __name__=="__main__":
    main()
