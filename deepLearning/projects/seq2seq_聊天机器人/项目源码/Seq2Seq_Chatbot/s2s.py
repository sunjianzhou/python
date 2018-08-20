#!/usr/bin/env python3
#encoding=utf8
import os
import sys
import math
import time

import numpy as np
import tensorflow as tf

import data_utils
import s2s_model

tf.app.flags.DEFINE_float(
    'learning_rate',
    0.0003,
    '学习率'
)
tf.app.flags.DEFINE_float(
    'max_gradient_norm',
    5.0,
    '梯度最大阈值'
)
tf.app.flags.DEFINE_float(
    'dropout',
    1.0,
    '每层输出DROPOUT的大小'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    64,
    '批量梯度下降的批量大小'
)
tf.app.flags.DEFINE_integer(
    'size',
    512,
    'LSTM每层神经元数量'
)
tf.app.flags.DEFINE_integer(
    'num_layers',
    2,
    'LSTM的层数'
)
tf.app.flags.DEFINE_integer(
    'num_epoch',
    5,
    '训练几轮'
)
tf.app.flags.DEFINE_integer(
    'num_samples',
    512,
    '分批softmax的样本量'
)
tf.app.flags.DEFINE_integer(
    'num_per_epoch',
    1000,
    '每轮训练多少随机样本'
)
tf.app.flags.DEFINE_string(
    'buckets_dir',
    './bucket_dbs',
    'sqlite3数据库所在文件夹'
)
tf.app.flags.DEFINE_string(
    'model_dir',
    './model',
    '模型保存的目录'
)
tf.app.flags.DEFINE_string(
    'model_name',
    'model3',
    '模型保存的名称'
)
tf.app.flags.DEFINE_boolean(
    'use_fp16',
    False,
    '是否使用16位浮点数（默认32位）'
)
tf.app.flags.DEFINE_integer(
    'bleu',
    -1,
    '是否测试bleu'
)
tf.app.flags.DEFINE_boolean(
    'test',
    False,
    '是否在测试'
)

FLAGS = tf.app.flags.FLAGS
buckets = data_utils.buckets

def create_model(session, forward_only):
    #forward_only即代表是否是训练还是预测，因为预测时只做向前传播训练时双向传播
    """建立模型"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = s2s_model.S2SModel(
        data_utils.dim, #字典中字的总数
        data_utils.dim,
        buckets, #即那四个问答字数的桶，[(5,15),(10,20),(15,25),(20,30)]
        FLAGS.size,
        FLAGS.dropout, 
        FLAGS.num_layers, #纵向上2个lstm，横向上是不同时间状态的变化。
        FLAGS.max_gradient_norm,  #最大的梯度截断
        FLAGS.batch_size,   #64
        FLAGS.learning_rate,  #0.01
        FLAGS.num_samples,   #512
        forward_only,   #True则为训练，False则为测试
        dtype
    )
    return model

def train():
    """训练模型"""
    # 准备数据
    print('准备数据')
    #数据预处理有两步：1、decode_conv 2、data_utils
    #原始数据集不是很好的问答式数据集。用decode_conv处理的数据，假定有ABC三个句子，则处理成两句问答：A:B,B:C，然后都插入到sqlite3里
    #生成一个conversion.db文件，然后使用data_utils来进行语句处理，即对这个db文件做进一步处理
    #对应四种格式，5_15,10_20,15_25,20_30,分别代表问句和回答句的字数上限。比如5_15即问句不超过5个字且答句不超过15个字。
    #这种方式也和命名实体识别的一个性质，是为了能最小padding，进行局部padding，如果有句子太长的，但是不太多，那么可以滤掉。
    #因为一般的对话都不会太长
    bucket_dbs = data_utils.read_bucket_dbs(FLAGS.buckets_dir)
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    #到这里为止还只是拿到四个bucket里的数据,并统计了一下总的数据条数
    # 开始建模与训练
    with tf.Session() as sess:
    #整体流程即：1、创建模型 2、接收数据，并转换成模型可接收的类型 3、放入模型，计算损失 4、更新参数
        #　构建模型
        model = create_model(sess, False)
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        buckets_scale = [ sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]#i=0,1,2,3==>bucket_sizes[: 1],
        # 开始训练
        metrics = '  '.join([
            '\r[{}]',
            '{:.1f}%',
            '{}/{}',
            'loss={:.3f}',
            '{}/{}'
        ])
        bars_max = 20
        with tf.device('/gpu:0'):
            for epoch_index in range(1, FLAGS.num_epoch + 1600):
                print('Epoch {}:'.format(epoch_index))
                time_start = time.time()
                epoch_trained = 0
                batch_loss = []
                while True:
                    # 选择一个要训练的bucket
                    random_number = np.random.random_sample()
                    #tmp=[]
                    #for i in range(len(buckets_scale)):
                        #if buckets_scale[i] > random_number:
                            #tmp.append(i)
                    #bucket_id = min(tmp)
                    bucket_id = 1 if random_number<=0.25 else 2 if random_number>0.25 and random_number<=0.5 else 3 if random_number>0.5 and random_number<0.75 else 4
                    bucket_id -= 1
                    #先选择对应的问答对长度，因为后面无论是padding还是生结果，都是根据这个位数来的
                    #bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
                    #拿出64个问答对，data 和data_in 问答倒转
                    data, data_in = model.get_batch_data(
                        bucket_dbs,
                        bucket_id
                    )#先获取到问答对和答问对
                    encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                        bucket_dbs,
                        bucket_id,
                        data
                    )#再得到padding后的encoder_inputs,decoder_inputs和新生成的权重decoder_weights
                    #而这里的encoder_inputs，decoder_inputs都只是对应的字ID信息，而decoder_weights则是1和0组成的,也是和字位置一对一对应
                    #通过源码可以看出，ID只是初步信息，随机初始化一个embedding是embedding_attention_seq2seq内部会有的
                    _, step_loss, output = model.step(
                        sess,
                        encoder_inputs,
                        decoder_inputs,
                        decoder_weights,
                        bucket_id,
                        False
                    )#给定需要喂入的参数，即encoder、decoder、weights以及选择的bucket_id
                    #根据训练和测试状态，获取输出结果。
                    epoch_trained += FLAGS.batch_size
                    batch_loss.append(step_loss) #为了计算损失用
                    time_now = time.time()
                    time_spend = time_now - time_start
                    time_estimate = time_spend / (epoch_trained / FLAGS.num_per_epoch)
                    percent = min(100, epoch_trained / FLAGS.num_per_epoch) * 100
                    bars = math.floor(percent / 100 * bars_max)
                    sys.stdout.write(metrics.format(
                        '=' * bars + '-' * (bars_max - bars),
                        percent,
                        epoch_trained, FLAGS.num_per_epoch,
                        np.mean(batch_loss),
                        data_utils.time(time_spend), data_utils.time(time_estimate)))
                    sys.stdout.flush()
                    if epoch_trained >= FLAGS.num_per_epoch:
                        break
                print('\n')

        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        if epoch_index%800==0:
            model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))


def test_bleu(count):
    """测试bleu"""
    from nltk.translate.bleu_score import sentence_bleu
    from tqdm import tqdm
    # 准备数据
    print('准备数据')
    bucket_dbs = data_utils.read_bucket_dbs(FLAGS.buckets_dir)
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    # bleu设置0的话，默认对所有样本采样
    if count <= 0:
        count = total_size
    buckets_scale = [
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    with tf.Session() as sess:
        #　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        sess.run(tf.initialize_variables())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

        total_score = 0.0
        for i in tqdm(range(count)):
            # 选择一个要训练的bucket
            random_number = np.random.random_sample()
            bucket_id = min([
                i for i in range(len(buckets_scale))
                if buckets_scale[i] > random_number
            ])
            data, _ = model.get_batch_data(
                bucket_dbs,
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                bucket_dbs,
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ask, _ = data[0]
            all_answers = bucket_dbs[bucket_id].all_answers(ask)
            ret = data_utils.indice_sentence(outputs)
            if not ret:
                continue
            references = [list(x) for x in all_answers]
            score = sentence_bleu(
                references,
                list(ret),
                weights=(1.0,)
            )
            total_score += score
        print('BLUE: {:.2f} in {} samples'.format(total_score / count * 10, count))


def test():
    class TestBucket(object):
        def __init__(self, sentence):
            self.sentence = sentence
        def random(self):
            return sentence, ''
    with tf.Session() as sess:
        #　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            #获取最小的分桶id
            bucket_id = min([
                b for b in range(len(buckets))
                if buckets[b][0] > len(sentence)
            ])
            #输入句子处理,获取问答对和答问对
            data, _ = model.get_batch_data(
                {bucket_id: TestBucket(sentence)},
                bucket_id
            ) #正常是bucket_dbs, bucket_id，即主要为了bucket_dbs[bucket_id]
            #而这里主要是为了能构建一个空的答案，故而第一个参数制造一个字典，从而也可以使用bucket_dbs[bucket_id]
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                {bucket_id: TestBucket(sentence)},
                bucket_id,
                data
            ) #得到encoder_inputs, decoder_inputs, decoder_weights
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            ) #输出为None，loss和outputs,这里只取了outputs
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            #对每个输出选择最大维度的那个
            ret = data_utils.indice_sentence(outputs)
            print(ret)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.bleu > -1:
        test_bleu(FLAGS.bleu)
    elif FLAGS.test:
        test()
    else:
        train()

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.app.run()
