import tensorflow as tf
import os
root_path=os.getcwd()+os.sep
from collections import OrderedDict
from deepLearning.命名实体识别.dataProcess import *
import pickle
from deepLearning.命名实体识别.utils import *
from deepLearning.命名实体识别.model import *

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.test"),   "Path for test data")
flags.DEFINE_boolean("lower",       False,       "Wither lower case")
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("tag_schema", "bioes", "tagging schema iobes or iob")
flags.DEFINE_integer("batch_size",    60,         "batch size")
flags.DEFINE_string("ckpt_path", "ckpt", "the path for ckpt")
flags.DEFINE_string("result_path", "result", "the path for result")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string( "config_file", "config_file", "file for config")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters") #每个字用100维数据来表示
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")#接收切词信息，seg_dim为切词长度
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN") #lstm的维度
flags.DEFINE_float("clip",          5,          "Gradient clip")#梯度截断，梯度大于5的就截断
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")


FLAGS = tf.app.flags.FLAGS

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

def prepare():
    #1、加载训练文件，测试文件以及验证文件，同时要指定是否字母转小写，数字转成0，返回列表
    # 训练样本在当前目录下的data底下，FLAGS.train_file=os.path.join(root_path+"data", "example.train")
    # 文件内容每行一个字加一个标签，中间以空格隔开，遇到一个空行表示一句话结束。
    # 要求"返回格式：[[['偏', 'B-SGN'], ['斜', 'I-SGN']],[['偏', 'B-SGN'], ['斜', 'I-SGN']]]"
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    #2、对IOB数据类型转IOBES类型，转换之前做IOB类型检查，操作列表
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    #3、先看看有无FLAGS.map_file='maps.pkl'，若有，直接获得"字-ID"、"ID-字","标签-ID"和"ID-标签"四个字典
    # 若无，则自己生，自己生的话，则用训练集来生词频字典，若有字向量文件，则以字向量文件进行补充
    # 根据词频字典按每个字出现次数从高到低排列，生成字对应的ID，则最终获取到字-ID和ID-字这两个字典。
    # 同样，对于标签也是，需要标签-ID，ID-标签，通过标签的词频字典来生成
    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb: #如果有字向量文件，则先提取训练集词频字典，再用测试集合自向量文件进行丰富
            dict_chars_train = char_mapping(train_sentences)[0]
            print("len(dict_chars_train):",len(dict_chars_train))

            # 提取测试集中所有的字，将双层list里每个文字拉平后的一个list
            chars = []
            for sentence in test_sentences:
                for each in sentence:
                    chars.extend(each[0])
            print("len(chars):",len(chars))

            dict_chars, char_to_id, id_to_char = rich_dict(
                dict_chars_train.copy(),  # 词频字典，从训练样本里获得
                FLAGS.emb_file,
                chars,
                FLAGS.lower,    #添加进字典时也按整体格式来，即是否变成小写,此处若不加lower和zeros限制，则会多出很多
                FLAGS.zeros     #是否将数字变0
                )
        else: #即没有字向量文件
            dict_chars, char_to_id, id_to_char = char_mapping(train_sentences)

        #对标签数据也做字典生成，包括标签词频字典，ID-标签，标签-ID
        dict_tag, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        #保存到map文件里
        with open(FLAGS.map_file, "wb") as file:
            pickle.dump([char_to_id,id_to_char,tag_to_id,id_to_tag],file)

    else: #即直接有map文件
        with open(FLAGS.map_file,"rb") as file:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(file)

    #4、通过prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)获取预备好的数据
    #得到的结果是一个列表[string, chars, segs, tags]，四个元素又都是一个等大小的列表
    #其中string为所有的字，chars为每个字对应ID，segs为切词信息，tags为每个字对应标签ID
    train_data = prepare_dataset( train_sentences, char_to_id, tag_to_id )
    dev_data = prepare_dataset( dev_sentences, char_to_id, tag_to_id )
    test_data = prepare_dataset( test_sentences, char_to_id, tag_to_id )

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    #5、创建一个BatchManager类，对上步得到的数据进行sort和分批padding，初始化传进来即[string, chars, segs, tags]和batch_size
    # 对每一批次数据做最后一次归整处理，并将每句话规整到每一批次里
    train_manager = BatchManager(train_data, FLAGS.batch_size) #对数据先排序，后每批次数据进行padding好
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    #到这里，数据预处理即OK了。

    #6、准备存储文件,加载参数配置文件，若没有，则存储起来
    prepare_files(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    #配置模型，开始迭代
    steps_per_epoch = train_manager.len_data
    with tf.Session() as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []
            if i % 7 == 0:
                save_model(sess, model, FLAGS.ckpt_path, logger)

def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session() as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            try:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                #此时即为最终的转成了json格式的输出了
                print(result)
            except Exception as e:
                 logger.info(e)

def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
            prepare()
    else:
        evaluate_line()

if __name__=="__main__":
    tf.app.run(main)