# encoding=utf8

import codecs
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
root_path=os.getcwd()+os.sep
flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")#接收切词信息，seg_dim为切词长度
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters") #每个字用100维数据来表示
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN") #lstm的维度
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")#梯度截断，梯度大于5的就截断
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    60,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.test"),   "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    #三个参数，一个是训练文件，一个是是否要对英文进行转换小写操作，一个是是否要对数字进行变0操作。
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    #对IOB类型转换成IOBES类型
    update_tag_scheme(train_sentences, FLAGS.tag_schema) #传的是一个列表，直接对列表修改
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            #返回有三个，词频字典，char_to_id，id_to_char其中id都是自己随意生成的
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),#词频字典，从训练样本里获得
                FLAGS.emb_file, #vec.txt 字向量文件，这里已经有了现成的字向量文件了，是提前训练好的，这里只是为了填充字，确保都加到词频字典里
                #这里的字向量文件，只是拿来丰富了词频字典的字，而词频字典也只是用了固定创建字对应ID使用。
                #不用字向量文件也是可以的，但是因为有这个字向量文件，想用起来，所以这里就这么用了。
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences]) #将双层list里每个文字拉平后的一个list
                )#此时char_to_id和id_to_char会比较完整，因为会将字向量文件中存在，而词频字典中不存在的字也添加到词频字典里，然后创建与ID的对应关系
            ) # 这个词频字典dico_chars会被更新，然后也比较全面，后面用作训练用的
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)
            #对标签进行词频字典，char_to_id, id_to_char的创建
            #实际上词频字典后面并没有去使用，只是收集了可以观察了解一下情况
            #但是这个字对应的ID，和ID对应字的字典是根据词频来确定的

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        #with open('maps.txt','w',encoding='utf8') as f1:
            #f1.writelines(str(char_to_id)+" "+id_to_char+" "+str(tag_to_id)+" "+id_to_tag+'\n')
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    #数据预处理，得到的即[每个字,字对应的ID,字对应的切词信息,标签ID]，这里的切词信息有和没有，最后的准确度能差10%以上
    train_data = prepare_dataset( 
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))
    
    #BatchManager处理之前内容格式[[一句话],[一句话],...,[一句话]]
    #BatchManager处理之后内容格式[[一批次],[一批次],...,[一批次]]
    train_manager = BatchManager(train_data, FLAGS.batch_size) #对数据先排序，后每批次数据进行padding好
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True #允许GPU的使用率上涨
    #tf_config.gpu_options.per_process_gpu_memory_fraction=0.6 #设置整个服务器上GPU的使用率
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        #创建模型
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        with tf.device("/gpu:0"):
        #没有GPU，纯CPU时还得拿掉
            for i in range(100):
                #迭代100次
                for batch in train_manager.iter_batch(shuffle=True):
                    #每次随机地取一批次的数据
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                        
                #best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
                if i%7==0: #若把上面的best打开的话，这里就换成if best：
                    save_model(sess, model, FLAGS.ckpt_path, logger)
            #evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                #此时即为最终的转成了json格式的输出了
                print(result)


def main(_):

    if FLAGS.train:
    #要么自己完整的训练一遍获得最佳的参数,即FLAGS.train=True
    #要么就用现成的模型参数，即已经训练好的：
    #即令FLAGS.train=False的同时，还要把ckptbak的文件覆盖复制到ckpt中，同时将config里的两个文件覆盖复制到最外层目录下即可
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)



