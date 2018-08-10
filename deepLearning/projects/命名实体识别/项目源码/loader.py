import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in open(path, 'r',encoding='utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:#以空行作为一句话结束的标志
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
        else:
            if line[0] == " ": #可能会有写错的情况，即第一个汉子写成了空格，初期可以不考虑
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split( )
            assert len(word) == 2  #断言确保一下确实是两个
            sentence.append(word)
    if len(sentence) > 0:
        #if 'DOCSTART' not in sentence[0][0]:
        sentences.append(sentence)
    return  sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences] #每个字转小写
    dico = create_dico(chars) #统计词频，生成词频字典
    dico["<PAD>"] = 10000001  #对于padding给予一个很大的数，为了后面padding用
    dico['<UNK>'] = 10000000  #对于未知的字符，也给一个很大的数
    char_to_id, id_to_char = create_mapping(dico)
    #print("Found %i unique words (%i in total)" % (
    #    len(dico), sum(len(x) for x in chars)
    #))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    
    f=open('tag_to_id.txt','w',encoding='utf8')
    f1=open('id_to_tag.txt','w',encoding='utf8')
    tags=[]
    
    for s in sentences:
    #s类似于[['去',O],['来',B-Tes]]这样
        ts=[]
        for char in s:
            tag=char[-1] #取出标签
            ts.append(tag)
        tags.append(ts)
    #上面这段可以写成一句话
    #tags = [[char[-1] for char in s] for s in sentences]
    
    #tags1 = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)  #统计词频，创建词频字典
    tag_to_id, id_to_tag = create_mapping(dico)
    #print("Found %i unique named entity tags" % len(dico))
    #下面只是为了写到文件里看看
    for k,v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k,v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        #print(sentences)
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string] #对每个字进行判断是否在词频字典里，如果有，返回字，没有，返回自己添加的<UNK>
        #即将每个字都进行ID化，即找到每个字对应的ID，对于那些不在字典里面的字，返回的是字典中<UNK>的ID
        #这个主要是针对那些不在训练字典里的字的情况，因为新的训练样本中很可能包含新的字，并不在训练字典里，即那些未登录的词
        segs = get_seg_features("".join(string))  #获取每个字的切词信息
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags]) #[一句话的每个字,字对应的ID,字对应的切词信息,标签ID]

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    #dictionary为训练集的词频字典，ext_emb_path为字向量文件，chars为测试集所有字组成的列表
    #主要是为了确保字向量文件的所有字都在词频字典中有，若无，则添加，并赋0
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    #print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    #ext_emb_path是字向量文件，即一个字加一个空格，后面跟着这个字的向量表示
    #下面即将该文件中非空行的第一个汉子都拿出来，放到一个set里
    #故而pretrained则是从字向量文件里取出了所有的字
    pretrained = set([
        line.rstrip().split()[0].strip() #先消除每行最后的换行符
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        #if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    #如果没有测试集，则把只存在于所有字向量，而不存在于词频字典中的，都加到词频字典中完善。
    #如果有测试集，则遍历一遍测试集，把测试集中那些存在于字向量，但不存在于训练集词频字典的，加到字典里完善。
    #故而这里首先，训练集中存在的字肯定都需要，其次，就只关心字向量中存在的那些字，但是不一定需要对字向量的字全看一遍，如果测试集中有这个字的话，就加到训练集里，如果没有，就不需要了。
    #完善了词频字典之后，则是确定ID关系
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0  #如果字在字向量文件里，但不在词频字典里，就添加上并赋值0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary: #如果当前的字、其小写、或其数字改写三种形态中有某种形态存在于字向量中，但是又不存在于词频字典里的
                dictionary[char] = 0  #就添加上，并赋值0
            #这里说明如果测试集中的字有在字向量文件里不存在的，就不添加到词频字典里了。

    word_to_id, id_to_word = create_mapping(dictionary)
    #这个时候的id对应是比较完整的
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

