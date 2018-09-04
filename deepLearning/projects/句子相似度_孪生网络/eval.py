#! /usr/bin/env python

#这个即负责测试的，应该对应着模型上线

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "validation.txt0", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1535127935/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1535127935/checkpoints/model-5000", "Load trained model checkpoint (Default: None)")
#这个要对应上路径，不然找不到文件就尴尬了

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)
#这里是从eval_filepath文件中读取的词语对，可以改成输入读取的。
#然后因为这个文件中读取到的还是有标签的,而真实预测数据可能是没有标签的,而网络中对应需要标签的主要即准确度.
#故而若是没有标签的真实数据，则需要将准确度去掉
#同时，对于这里，则一方面需要大致修改一下getTestDataSet
#即令x1为接收到的请求词语，x2即自己语料库中所有词。也不需要label。
#当然，因为x1与x2是会去计算对比损失的，也就是x1与x2需要等大小，x2为整个语料库，故而x1也要复制成那么大的
#当然，由于后面肯定是用的x1，x2的ID，所以可以在转换成ID之后，对x1的ID直接复制成语料库大小的份数，这样会快一些
#测试时也不用去获取accuracy即可

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print (checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))  #选择meta来加载，可以不用重新定义网络结构
        sess.run(tf.initialize_all_variables()) #直接加载了整个图结构后，这句话完全可以不要(已验证)，加载过程本质就相当于初始化过程
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0] #这里加载的变量都是训练时定义的
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
            #因为对应的只有batch_acc需要真实标签的，故而如果是真实预测，没有标签,那么需要把batch_acc去掉
            #然后稍微改一下dataset
            batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
            #测试中如果没有真实数据的标签的话，那么后两个其实都可以去掉
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print(batch_predictions)
            #故而如果是真实数据没有标签的话，这后面的内容则都可以去掉了
            all_d = np.concatenate([all_d, batch_sim])
            print("DEV acc {}".format(batch_acc))
        for ex in all_predictions:
            print (ex )
        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions))
