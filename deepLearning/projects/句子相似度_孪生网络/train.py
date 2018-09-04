#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from siamese_network_semantic import SiameseLSTMw2v
from tensorflow.contrib import learn
import gzip
from random import random
# Parameters
# ==================================================

tf.flags.DEFINE_boolean("is_char_based", True, "is character based syntactic similarity. "
                                               "if false then word embedding based semantic similarity is used."
                                               "(default: True)")
#is_char_based若为True，则基于字符的语法相似，若为False，则基于字符的语义相似
#这里的wiki.simple.vec即对应该目录下wiki.simple.vec.zip，以文件夹的方式能看到
tf.flags.DEFINE_string("word2vec_model", "wiki.simple.vec", "word2vec pre-trained embeddings file (default: None)")
tf.flags.DEFINE_string("word2vec_format", "text", "word2vec pre-trained embeddings file format (bin/text/textgz)(default: None)")

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
#词向量维度需要和word2vec里面结果对应
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")  #for sentence semantic similarity use "train_snli.txt"
#指定训练文件
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units (default:50)")
#lstm里面的维度数


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
#表示每1000次评估一次
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print("Input Files List is empty. use --training_files argument.")
    exit()


max_document_length=15 #最大句子长度为15
inpH = InputHelper()
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.training_files,max_document_length, 10,
                                                                         FLAGS.batch_size, FLAGS.is_char_based)
#is_char_based若为True，则基于字符的语法相似，若为False，则基于字符的语义相似
#若为True，则自己手动添加负样本，也就是可以直接去从正样本中寻找不相关答案，从而语法基本一致。

trainableEmbeddings=False
if FLAGS.is_char_based==True:
    FLAGS.word2vec_model = False  #若基于语法相似，则不用现成的word2vec，因为现成的word2vec是基于语义相似生成的
else: #表明负样本是要“基于语义相似”的，上面也不会去生成负样本，当然，有可能本身具备正负样本
    if FLAGS.word2vec_model==None: #表明没有现成word2vec文件可以用
        trainableEmbeddings=True   #这个True是说对样本也训练
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "You are using word embedding based semantic similarity but "
          "word2vec model path is empty. It is Recommended to use  --word2vec_model  argument. "
          "Otherwise now the code is automatically trying to learn embedding values (may not help in accuracy)"
          "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:#有现成的word2vec，加载也只是加载到辅助类里,故而并不会有返回值
        inpH.loadW2V(FLAGS.word2vec_model, FLAGS.word2vec_format) 
        #word2vec_model对应文件名，word2vec_format为文件存储格式
        #这里表示有现成的word2vec了，就不再对样本的数据进行训练了

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement, #True
      log_device_placement=FLAGS.log_device_placement) #False
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        if FLAGS.is_char_based:  #表明基于语法相似，所谓语法，即和给定的正负样本语法基本一致的，也不会加载现成的字向量
            siameseModel = SiameseLSTM(   #这个用的是双向lstm
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size
            )
        else:  #表明基于语义相似
            siameseModel = SiameseLSTMw2v( #这个用的是单向的lstm
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size,
                trainableEmbeddings=trainableEmbeddings #
            )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)#对loss求梯度
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step) #更新参数
    #global_step初值为0，设置了以后后面每次更新都会自动加1,从而打印时可以看到效果，即每次的变化
    #当然，因为这里也没有做梯度截断操作，纯粹地只是分开写了一下，故而也可以直接一句话
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(siameseModel.loss)
    
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    #tf.summary模块代码主要即为了可视化，故而下面主要都是可视化部分内容
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100) #最多存最近的100个,相当于有一个队列，容量就100。
    #若设置为None或者0，则每次都保存。若设置为1，则只保存最后一个。一般设置为5

    # Write vocabulary
    vocab_path=os.path.join(checkpoint_dir, "vocab")
    vocab_processor.save(vocab_path)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    #将图的整体结构信息存储下来
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    if FLAGS.word2vec_model :
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        #先随机生成总表，一次随机效率高一些
        #initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        print("initializing initW with pre-trained word2vec embeddings")
        #下面即将语料库中的字用word2vec里有的去代替掉，没有的，则还是原来的随机初始化的
        for w in vocab_processor.vocabulary_._mapping:
            arr=[]
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            #这里即将非数字字母的都转成空
            #可以修改成s = re.sub("[^\w]",'',w),即非数字字母下划线或中文的 才换成空
            #这里可以改进，从而可以处理中文汉字
            if w in inpH.pre_emb:
                arr=inpH.pre_emb[w]
            elif w.lower() in inpH.pre_emb:
                arr=inpH.pre_emb[w.lower()]
            elif s in inpH.pre_emb:
                arr=inpH.pre_emb[s]
            elif s.isdigit():  #这里的是数字，即非0的数字
                arr=inpH.pre_emb["zero"]
            if len(arr)>0:
                idx = vocab_processor.vocabulary_.get(w)
                initW[idx]=np.asarray(arr).astype(np.float32)
        print("Done assigning intiW. len="+str(len(initW)))
        inpH.deletePreEmb()
        gc.collect()
        sess.run(siameseModel.W.assign(initW))
        #这里即再把initW赋值给siameseModel.W，这个技巧很重要
        #这块可以单写成一个函数

    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }  #通过随机函数来抉择x1和x2哪个放问句，哪个答句，即更加随机化，从而增强学习能力，能同时处理问答
        _, step, loss, accuracy, dist, sim, summaries = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim, train_summary_op],  feed_dict)
        #喂入需要的数据，将网络中所有想要的值都执行一遍
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step) #这个是用于画图用的
        print(y_batch, dist, sim)

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """ 
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim, summaries = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim, dev_summary_op],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        dev_summary_writer.add_summary(summaries, step)
        print (y_batch, sim)
        return accuracy

    # Generate batches
    batches=inpH.batch_iter(
                list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)
    #即取一批次的数据
    #这里list(zip(train_set[0], train_set[1], train_set[2]))很巧妙，即将三部分数据绑在一起，一个数据即对应三个数据相应数据
    #但是这里train_set好像直接读进去也行，就不需要list(zip)了不过没有去验证
    
    ptr=0
    max_validation_acc=0.0
    for nn in range(sum_no_of_batches*FLAGS.num_epochs): #sum_no_of_batches为训练集每批次对应的长度
        batch = next(batches) #取下一批，因为前面是从迭代器里面取的，这里面已经处理了本批次样本数不足的情况
        if len(batch)<1: #这句话只是为了完整一点，但本质上没必要因为都会是3个
            continue
        x1_batch,x2_batch, y_batch = zip(*batch)  #即将上面的zip(train_set[0], train_set[1], train_set[2])解压缩
        if len(y_batch)<1:   #同样的，这里对标签值长度判断，为了鲁棒性，但实际上感觉应该可以不要
            continue
        train_step(x1_batch, x2_batch, y_batch) #取到一批次数据后进行train，train则会进行损失求梯度，参数更新等
        current_step = tf.train.global_step(sess, global_step)  #获取到当前对应的步子
        #因为前面调了一次train_step，而train_step里又run了tr_op_set
        #而train_step里又会有一次tr_op_set = optimizer.apply_gradients(grads_and_vars,global_step) #更新参数
        #故而global_step会发生变化
        sum_acc=0.0
        if current_step % FLAGS.evaluate_every == 0: #这里evaluate_every为1000，即每1000步进行一次交叉验证
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0],dev_set[1],dev_set[2])), FLAGS.batch_size, 1)
            #这里的dev_set是交叉验证集，在训练集中（训练集比验证集9:1）取出来的，为了做一轮交叉验证
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc  #加和准确度，这里准确度计算的是对了多少个
            print("")
        if current_step % FLAGS.checkpoint_every == 0: #每一百次迭代进行一次模型保存
            if sum_acc >= max_validation_acc:  #前提是当前的准确度大于目前最好的交叉验证准确度
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))
