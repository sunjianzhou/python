import os
import json
import logging
import tensorflow as tf
import numpy as np
import shutil

def prepare_files(params):
    #准备result文件、checkpoint文件和日志文件
    if not os.path.isdir(params.result_path):
        os.mkdir(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.mkdir(params.ckpt_path)
    if not os.path.isdir("log"):
        os.mkdir("log")

def load_config(config_file):
    assert os.path.isfile(config_file)
    with open(config_file,"r",encoding="utf-8") as file:
        return json.load(file)

def save_config(config, config_file):
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4) #ensure_ascii=False是为了避免中文读取是ascii格式，indent=4表示缩进4格

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def print_config(config, logger):
    for key,value in config.items():
        logger.info("{}:\t{}".format(key.ljust(15),value))

def create_model(session, Model_class, path, load_vec, config, id_to_char, logger):
    model = Model_class(config)
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # 如果ckpt文件夹以及文件夹底下的文件都存在，则加载模型
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value()) #也是先拿随机初始化的值
            emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)  #再更新
            session.run(model.char_lookup.assign(emb_weights)) #更新完后赋值给总表char_lookup
            logger.info("Load pre-trained embedding.")
    return model


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    #即将old_weights里面的进行更新，即看看emb_path里有的内容，则替换掉old_weights里面的内容，是根据id_to_word来寻找的
    #遍历一遍emb_path文件，将字放入字典，并统计长度，若每行内容长度不是word_dim+1（字向量+字）,则统计出数目并打出来即可。
    #有pretrain的前提下，如果pretrain里有，则用pretrain里的向量，没有，则不替换
    new_weights = old_weights
    if not os.path.isfile(emb_path):
        return new_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    file = open(emb_path,"r",encoding="utf-8")
    for line in file:
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(i) for i in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    words_len = len(id_to_word)
    for idx in range(words_len):
        word = id_to_word[idx] #第idx个字对应下面第idx个字向量
    if word in pre_trained:
        new_weights[idx] = pre_trained[word]
    file.close()
    return new_weights

def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item