import os
import re
from operator import itemgetter
import jieba
import math
import random

def load_sentences(fileName, lower=False, zero=False):
    "返回格式：[[['偏', 'B-SGN'], ['斜', 'I-SGN']],[['偏', 'B-SGN'], ['斜', 'I-SGN']]]"
    #即每句话多个字，每个字和其标签一个列表，共有多句话
    assert os.path.isfile(fileName)
    sentences = []
    sentence = []
    with open(fileName,"r",encoding="utf-8") as file:
        pre_sentences = [ each.rstrip().tolower() if lower else each.rstrip() for each in file.readlines() ]
        pre_sentences = [ re.sub("\d", "0", each) if zero else each for each in pre_sentences ]

        for each in pre_sentences:
            word = []
            if each:
                word = each.split()
                assert len(word) == 2
                sentence.append(word)
            else:
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []

        if len(sentence) > 0: #最后一句话的内容
            sentences.append(sentence)
    return sentences #[[['偏', 'B-SGN'], ['斜', 'I-SGN']],[['偏', 'B-SGN'], ['斜', 'I-SGN']]]

def check_BIO(tags):
    for index,tag in enumerate(tags):
        #1、若是O，则跳过
        #2、若是B，也跳过
        #3、若是I，那么如果前一个元素存在并且是B或者I，则跳过。否则，都改成B
        #4、其它的，报错，没法修
        if tag == "O":
            continue
        tag = tag.split("-")[0]
        if tag == "B":
            continue
        elif tag == "I":
            if index != 0 and tags[index-1].split("-")[0] in ["B","I"]:
                continue
            else:
                tags[index] = "B"
        else:
            return False
    return True

def BIOES_format(tags):
    #1、若是O，则跳过
    #2、若是B，则看下一个，若下一个存在且是I，则跳过，否则，换成S
    #3、若是I，则看下一个，若下一个存在且是I，则不变，否则，换成E
    for index,tag in enumerate(tags):
        if tag == "O":
            continue
        if tag == "B":
            if index != len(tags)-1 and tags[index+1].split("-")[0] == "I":
                continue
            else:
                tags[index] = "S" + tags[index][1:]
        else:
            if index != len(tags)-1 and tags[index+1].split("-")[0] == "I":
                continue
            else:
                tags[index] = "E" + tags[index][1:]

def update_tag_scheme(sentences, tag_schema):
    "输入的是BIO格式，先check，后转换"
    #sentences格式：[[['偏', 'B-SGN'], ['斜', 'I-SGN']],[['偏', 'B-SGN'], ['斜', 'I-SGN']]]
    for index,sentence in enumerate(sentences):
        #sentence则是每一句话了
        tags = [ each[-1] for each in sentence] #['B-SGN', 'I-SGN', 'O']
        if not check_BIO(tags):
            strs = [ "".join(each) for each in sentence]
            raise Exception("Sentence should be IOB format," +
                            "please check the line %i:\n %s"%(index,strs))
        if tag_schema == "bio":
            for word,new_tag in zip(sentence,tags):
                word[-1] = new_tag
        elif tag_schema == "bioes":
            BIOES_format(tags)
            for word,new_tag in zip(sentence,tags):
                word[-1] = new_tag
        else:
            raise Exception("No such format")

def char_mapping(sentences):
    #输入的sentences如：[[['偏', 'B-SGN'], ['斜', 'I-SGN']],[['偏', 'B-SGN'], ['斜', 'I-SGN']]]"
    #输出时三个字典，一个是拍好序的词频字典，一个是字-ID，一个是ID-字，ID是根据词频字典来的
    pre_dict = {}
    for index,sentence in enumerate(sentences):
        for word in sentence:
            if word[0] not in pre_dict:
                pre_dict[word[0]] = 1
            else:
                pre_dict[word[0]] += 1

    # 因为可能存在没有字向量文件的情况，也就不需要进行字典丰富了
    word_id_dict, id_word_dict = generate_ID(pre_dict)

    return pre_dict, word_id_dict, id_word_dict

def rich_dict(dictionary, ext_emb_path, chars, lower, zero):
    #dictionary为训练集的词频字典，ext_emb_path为字向量文件，chars为测试集所有字组成的列表
    #若无测试集，则将字向量文件中内容补充到训练字典里
    #若有测试集，则只将训练集与字向量文件中交集部分补充到训练字典里
    #另外，为防止测试集中的字在字典中仍可能不存在，故而添加一个<unknown>
    #同时，处于做膨胀卷积神经网络，再添加一个<padding>，用于后续填充使用
    assert os.path.isfile(ext_emb_path)
    with open(ext_emb_path,"r",encoding="utf-8") as file:
        words_vec = [ word.split()[0] for word in file.readlines()]

    if len(chars) == 0:
        for each in words_vec:
            if each not in dictionary:
                dictionary[each] = 0
    else:
        for each in chars:
            all_each = [
                each, each.lower(), re.sub("\d","0",each)
            ]
            if any(char in words_vec for char in all_each
                   ) and all( char not in dictionary for char in all_each):
                each = each.lower() if lower else each
                each = re.sub("\d","0",each) if zero else each
                dictionary[each] = 0

    dictionary["<UNK>"] = 0
    dictionary["<padding>"] = 0

    #因为ID-字是根据词频字典排序来的，所以获取词频字典和获取ID-字，以及字-ID就放在一个函数里
    word_id_dict, id_word_dict = generate_ID(dictionary)

    return dictionary,word_id_dict,id_word_dict

def tag_mapping(sentences): #类似char-mapping，只是提取的内容不一样
    res_dict = {}
    for index, sentence in enumerate(sentences):
        for word in sentence:
            if word[1] not in res_dict:
                res_dict[word[1]] = 1
            else:
                res_dict[word[1]] += 1

    tag_id_dict, id_tag_dict = generate_ID(res_dict)

    return res_dict, tag_id_dict, id_tag_dict

def generate_ID(dictionary):
    res_dict = sorted(dictionary.items(), key=itemgetter(1, 0), reverse=True)
    key_id_dict = {word[0]: index for index, word in enumerate(res_dict)}
    id_key_dict = {value: key for key, value in key_id_dict.items()}
    return key_id_dict,id_key_dict

def prepare_dataset( sentences, char_to_id, tag_to_id ):
    #要求返回是[chars, chars_id, segs, tags_id]，分别对应所有字，字对应ID，切词信息，标签ID
    data = []

    for sentence in sentences:
        words = [ word[0] for word in sentence ]
        tags = [ word[1] for word in sentence ]

        seg_nums = get_seg_num("".join(words))

        chars_id = []
        for word in words:
            chars_id.append(char_to_id[word]) if word in char_to_id else chars_id.append(char_to_id["<UNK>"])

        tags_id = [ tag_to_id[tag] for tag in tags ]

        data.append([words,chars_id,seg_nums,tags_id])
    return data

def get_seg_num(words_str):
    #切割词，得到的进行bies标注，s：0，b：1，i：2，e:3
    seg = jieba.cut(words_str)
    seg_nums = []
    for each in seg:
        if len(each) == 1:
            seg_nums.append(0)
        else:
            temp = [2] * len(each)
            temp[0] = 1
            temp[-1] = 3
            seg_nums.extend(temp)
    return seg_nums

def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)

def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line) #全字符转半字符，全角转半角
    line = replace_html(line) #替换html中的特殊字符
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_num(line)])
    inputs.append([[]]) #最后一个列表用于存放结果
    #最后inputs即为[[“所有字”],[[每个字对应的ID]],[[每个字对应的切词数字]],[[]]]
    #比如inputs：[['冠心病与心脏病的关系'],[[179,19,12,,429,17,542,749,63,168]],[[1,2,3,0,1,3,0,1,3]],[[]]]
    return inputs

class BatchManager:
    def __init__(self, data,  batch_size):
        #data格式为多个[string, chars, segs, tags]，每个代表一句话中的所有字，字ID，切词信息，标签ID
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        #对句子长度小于3的直接滤掉,因为句子都还有句号，去掉句号最多就两字的。。。真心不想要，主要是得不到啥有用信息
        data = [ each for each in data if len(each[0]) > 3 ]
        #对数据进行按批次padding，padding成一样大小的。为减少padding量，可以先排序。
        batch_num = math.ceil(len(data)/batch_size)

        sorted_data = sorted(data,key=lambda x: len(x[0])) #按句子长短进行排序
        batch_data = []
        for i in range(batch_num):
            batch_data.append(self.pad_data(sorted_data[i*batch_size:(i+1)*batch_size]))
        return batch_data

    def pad_data(self,data):
        #按本批次最大进行padding
        #传进来的数据格式是一批次的[string, chars, segs, tags]，返回的是整体归整的
        #即从batch_size个[string, chars, segs, tags]，变成一个[string, chars, segs, tags]，里面每个列表都是汇总的
        #所有pading操作均为补0
        batchsize_string,batchsize_chars,batchsize_segs,batchsize_tags = [],[],[],[]
        max_len = max( len(sentence[0]) for sentence in data)
        for sentence in data:
            current_len = len(sentence[0])
            padding = [0] * (max_len-current_len)
            batchsize_string.append(sentence[0]+padding)
            batchsize_chars.append(sentence[1]+padding)
            batchsize_segs.append(sentence[2]+padding)
            batchsize_tags.append(sentence[3]+padding)

        pad_data = [batchsize_string,batchsize_chars,batchsize_segs,batchsize_tags]
        return pad_data

    def iter_batch(self, shuffle=False):
        if shuffle :
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]