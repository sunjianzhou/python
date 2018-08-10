import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.crf import crf_log_likelihood
import numpy as np
from deepLearning.命名实体识别.utils import *

class Model(object):
    def __init__(self,config):
        self.config = config

        self.lr = config["lr"]
        self.char_dim = config["char_dim"]  # 字设定100维
        self.lstm_dim = config["lstm_dim"]  # 如果没有lstm，则代表卷积核的个数，复用了一下变量，也是100个
        self.seg_dim = config["seg_dim"]  # 切词信息给定20维

        self.num_tags = config["num_tags"]  # 每个字可能对应的标签数 51  即那些B-Tes，I-Tes，E—Tes，S-Tes等等
        self.num_chars = config["num_chars"]  # 样本中总字数
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)  # trainable=False的表明不需要用梯度去更新
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)  # 验证集上的f1值，因为只是最后评价一下用，故而也不需要去梯度更新它
        self.best_test_f1 = tf.Variable(0.0, trainable=False)  # 测试集上的f1值
        self.initializer = tf.contrib.layers.xavier_initializer()  # 这种初始化能保证输出和输入尽可能地服从相同的概率分布

        # 输入的每个字
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],  # 一个是batchsize，一个是每句话中字的长度，即外面大列表中的第1项
                                          name="ChatInputs")
        # 切词信息
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],  # 一个是batchsize，一个是每句话中字的长度，即外面大列表中的第2项
                                         name="SegInputs")
        # 标签
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],  # 一个是batchsize，一个是每句话中字的长度，即外面大列表中第3项
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        length = tf.reduce_sum(tf.sign(self.char_inputs),reduction_indices=1)
        self.lengths = tf.cast(length,tf.int32)

        self.batch_size = tf.shape(self.char_inputs)[0]  # 因为self.char_inputs为batch_size*每句话中字的个数,故而0即batch_size
        self.num_steps = tf.shape(self.char_inputs)[-1]  # 一句话中字的个数

        self.model_type = config['model_type']

        self.layers = [{ 'dilation': 1 }, { 'dilation': 1 }, { 'dilation': 2 }]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.char_dim + self.seg_dim

        self.repeat_times = 4
        self.cnn_output_width = 0

        #确定总的额embedding
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':  # 双向lstm
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)  # 使用双向lstm，指定内部维度数和解码长度

            self.logits = self.project_layer_bilstm(model_outputs)

        elif self.model_type == 'idcnn': #膨胀卷积神经网络
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            model_outputs = self.IDCNN_layer(model_inputs)

            self.logits = self.project_layer_idcnn(model_outputs)

        else:
            raise KeyError

        #计算损失
        self.loss = self.loss_layer(self.logits, self.lengths)

        #设置优化器
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]

            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.loss) #根据损失计算梯度
            capped_grads_vars = [[tf.clip_by_value(g,-self.config['clip'],self.config['clip']),v] for g,v in grads_vars]#梯度截断
            self.train_op = self.opt.apply_gradients(capped_grads_vars,self.global_step) #更新参数

        #保存模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
            embedding = []
            self.char_inputs_test = char_inputs  # batch_size*num_steps 即多少句话*每句话多少字 是2维的
            self.seg_inputs_test = seg_inputs  # 即多少句话*每句多少个切词信息(字)

            with tf.variable_scope("char_embedding"):
                self.char_lookup = tf.get_variable(
                    name = "char_embedding",
                    shape = [self.num_chars,self.char_dim],
                    initializer=self.initializer)
                embedding.append(tf.nn.embedding_lookup(self.char_lookup,char_inputs))
                if config["seg_dim"]:
                    with tf.variable_scope("seg_embedding"):
                        self.seg_lookup = tf.get_variable(
                            name = "seg_embedding",
                            shape = [self.num_segs,self.seg_dim],
                            initializer=self.initializer)
                        embedding.append(tf.nn.embedding_lookup(self.seg_lookup,seg_inputs))
                embed = tf.concat(embedding,axis = -1)

            self.embed_test = embed
            self.embedding_test = embedding
            return embed

    def IDCNN_layer(self, model_inputs,name=None):
    # 模型的输入是[batch_size,num_steps,num_tags],即对应着NWC三个维度的数据
    # 故而先expand上H维度，然后进入网络
    # 网络流程为：conv -> "atrons_conv+bias_add+relu"*4,四次循环中每次结果作为下一次输出，并append到finalOutFromLayers里
    # 四次循环结束后，对finalOutFromLayers进行第三维的concat，finalOutFromLayers：[batch_size,H=1,num_steps,num_tags],后两个为字数和标签数
    # concat完之后对finalOutFromLayers进行dropout模型压缩，然后squeeze消去expand上的那个维度
    # 再对当前结果做全连接层dense操作，即Y=XW+b
    # 最后将结果从 [batch_size*每句话字数,结果类别数] reshape 为 [batch_size,每句话字数,结果类别数]
        model_inputs = tf.expand_dims(model_inputs,1)
        self.model_inputs_test = model_inputs
        reuse = True if self.dropout == 1.0 else False

        with tf.variable_scope("idcnn" if not name else name):
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape = [1,self.filter_width,self.embedding_dim,self.num_filter],
                initializer=self.initializer)
            layerInput = tf.nn.conv2d(model_inputs,filter_weights,strides=[1,1,1,1],padding="SAME",name="init_layer")
            self.layerInput_test = layerInput
            finalOutFromLayers = []

            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers)-1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" %i,reuse = True if (reuse or j > 0) else False):
                        w = tf.get_variable("filterW",shape=[1,self.filter_width,self.num_filter,self.num_filter],initializer=self.initializer)
                        if j == 0 and i == 0:
                            self.w_test_1 = w
                        if j == 1 and i ==0:
                            self.w_test_2 = w
                        b = tf.get_variable("filterB",shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,w,rate=dilation,padding="SAME")
                        self.conv_test = conv
                        conv = tf.nn.bias_add(conv,b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv

            finalOut = tf.concat(finalOutFromLayers,axis=3)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut,keepProb)
            finalOut = tf.squeeze(finalOut,[1])
            finalOut = tf.reshape(finalOut,[-1,totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W",shape=[self.cnn_output_width,self.num_tags],dtype=tf.float32,initializer=self.initializer)
                b = tf.get_variable("b",initializer=tf.constant(0.001,shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(idcnn_outputs,W,b)
        return tf.reshape(pred,[-1,self.num_steps,self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        #project_logits: [batch_size, num_steps, num_tags]
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            pad_logits = tf.cast(small*tf.ones([self.batch_size,self.num_steps,1]),tf.float32)
            logits = tf.concat([project_logits,pad_logits],axis = -1)
            start_logits = tf.concat([small*tf.ones(shape=[self.batch_size,1,self.num_tags]),tf.zeros(shape=[self.batch_size,1,1])],axis=-1)
            logits = tf.concat([start_logits,logits],axis = 1)
            #此时logits:[batch_size, num_steps+1, num_tags+1]

            targets = tf.concat([tf.cast(self.num_tags*tf.ones([self.batch_size,1]),tf.int32),self.targets],axis = -1)

            self.trans = tf.get_variable("transitions",shape=[self.num_tags+1,self.num_tags+1],initializer=self.initializer)

            log_likelihood,self.trans = crf_log_likelihood(
                inputs = logits,
                tag_indices= targets,
                transition_params= self.trans,
                sequence_lengths= lengths+1
            )

            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            # 把所有可能需要的值都记录下来，后面要debug时可以直接查看
            global_step, loss,_,char_lookup_out,seg_lookup_out,char_inputs_test,seg_inputs_test,embed_test,embedding_test,\
                model_inputs_test,layerInput_test,conv_test,w_test_1,w_test_2,char_inputs_test= sess.run(
                [self.global_step, self.loss, self.train_op,self.char_lookup,self.seg_lookup,self.char_inputs_test,self.seg_inputs_test,\
                 self.embed_test,self.embedding_test,self.model_inputs_test,self.layerInput_test,self.conv_test,self.w_test_1,self.w_test_2,self.char_inputs],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            #即得到句子的长度，以及整个网络最后连接层的总输出
            return lengths, logits

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        #即拿到状态转移矩阵
        lengths, scores = self.run_step(sess, False, inputs)
        #得到最终句子长度，以及句子信息经过网络最后一层全连接层之后的输出信息
        batch_paths = self.decode(scores, lengths, trans)
        #进行解码操作，用的是维特比算法，得到的即标签对应的数字
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        #转成相应的标签
        return result_to_json(inputs[0][0], tags) #转成json字符串的形式输出

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            # 用维特比算法解码

            paths.append(path[1:])
        return paths
