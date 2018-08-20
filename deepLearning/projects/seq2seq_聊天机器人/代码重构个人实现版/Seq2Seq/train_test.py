import tensorflow as tf
import deepLearning.Seq2Seq.dataProcess as dataProcess
import deepLearning.Seq2Seq.seq2seq_model as seq2seq_model
import numpy as np
import time,sys,os

flags = tf.app.flags
flags.DEFINE_integer("lstm_size",512,"lstm每层神经元数目")
flags.DEFINE_float('dropout',1.0,'每层输出DROPOUT的大小')
flags.DEFINE_integer('num_layers',2,'LSTM的层数')
flags.DEFINE_float('clip_norm',5.0,'全局梯度正则比例')
flags.DEFINE_integer('batch_size',64,'批量梯度下降的批量大小')
flags.DEFINE_float('learning_rate',0.0003,'学习率')
flags.DEFINE_integer('num_samples',512,'分批softmax的样本量')
flags.DEFINE_integer('num_per_epoch',1000,'每轮训练多少随机样本')
flags.DEFINE_integer('num_epoch', 50, '训练几轮')
flags.DEFINE_string('model_dir','./model','模型保存的目录')
flags.DEFINE_string('model_name', 'model3','模型保存的名称')
flags.DEFINE_boolean("forTrain",True,"是否用于训练")
FLAGS = tf.app.flags.FLAGS

buckets = dataProcess.buckets

def create_model(session, forTrain):
    return seq2seq_model.Seq2Seq(
        encoder_vocb_size = dataProcess.dim,
        decoder_vocb_size = dataProcess.dim,
        buckets = buckets,
        lstm_size = FLAGS.lstm_size,
        dropout = FLAGS.dropout,
        num_layers = FLAGS.num_layers,
        clip_norm = FLAGS.clip_norm,
        batch_size = FLAGS.batch_size,
        learning_rate = FLAGS.learning_rate,
        num_samples = FLAGS.num_samples,
        forTrain = forTrain
        )

def train():  #跑了十轮后，loss能从8降到4，后面就降的很慢了
    print("准备数据")
    totalNum = 0
    for fileName,wordNum in dataProcess.words_dict.items():
        print("bucket {} 中有{}条数据".format(fileName,wordNum))
        totalNum += wordNum
    print("Total numbers of ask_answer is:",totalNum)

    with tf.Session() as sess:
        model = create_model(sess,forTrain=FLAGS.forTrain)
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.num_epoch):
            time_start = time.time()
            print("Epoch:", epoch)
            total_loss = []
            epoch_trained = 0
            while True:
                bucket_id = int(np.floor(np.random.random()*4))
                data,_ = model.get_batch_data(FLAGS.batch_size,bucket_id)
                batch_encoder_inputs, batch_decoder_inputs, batch_weights = model.get_batch(bucket_id,data)
                loss,outputs = model.step(sess,batch_encoder_inputs,batch_decoder_inputs,batch_weights,bucket_id,forTrain=FLAGS.forTrain)
                epoch_trained += FLAGS.batch_size
                total_loss.append(loss)  # 为了计算损失用
                time_spend = time.time() - time_start
                percent = min(100, epoch_trained / FLAGS.num_per_epoch * 100)
                sys.stdout.write(
                    "percent:{} trained:{} epoch:{} loss:{} spend time:{} \n".format(
                    percent, epoch_trained, FLAGS.num_per_epoch,np.mean(total_loss),dataProcess.time(time_spend)))
                sys.stdout.flush()
                if epoch_trained >= FLAGS.num_per_epoch:
                    break

            if not os.path.exists(FLAGS.model_dir):
                os.makedirs(FLAGS.model_dir)
            if epoch % 10 == 0:
                model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

def test():
    with tf.Session() as sess:
        model = create_model(sess, FLAGS.forTrain)
        model.batch_size = 1
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] >= len(sentence) ])
            data = [(sentence.strip(),"")]
            batch_encoder_inputs, batch_decoder_inputs, batch_weights = model.get_batch(bucket_id, data)
            _, outputs = model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_weights, bucket_id,
                                       forTrain=FLAGS.forTrain)

            outputs = [int(np.argmax(logit, axis=1)) for logit in outputs]
            ret = dataProcess.sentence_decode(outputs)
            print(ret)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.forTrain:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()