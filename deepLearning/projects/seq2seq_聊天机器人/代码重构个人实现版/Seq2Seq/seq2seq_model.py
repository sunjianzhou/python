import tensorflow as tf
import copy
import numpy as np
import deepLearning.Seq2Seq.dataProcess as dataProcess

class Seq2Seq:
    def __init__(self,
                 encoder_vocb_size,
                 decoder_vocb_size,
                 buckets,
                 lstm_size,
                 dropout,
                 num_layers,
                 clip_norm,
                 batch_size,
                 learning_rate,
                 num_samples,
                 forTrain=True,
                 dtype = tf.float32
                 ):
        self.encoder_vocb_size = encoder_vocb_size #主要用于embedding，构造总矩阵
        self.decoder_vocb_size = decoder_vocb_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        output_projection = None
        #判断是否开启投影，即字典中字数是否太多，多于预估的类别数
        if num_samples >0 and num_samples < decoder_vocb_size:
            print("开启投影")
            w = tf.get_variable("projection_w",[lstm_size,decoder_vocb_size],dtype=dtype)
            b = tf.get_variable("projection_b",[decoder_vocb_size],dtype=dtype)
            output_projection = (w,b)

        #模型：LSTM三层定义，因为是seq2seq，即训练时可以双向，测试时只能单向，故而这里暂时先都单向
        cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob = dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="encoder_inputs_{}".format(i)))
        for i in range(buckets[-1][1]+1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="decoder_inputs_{}".format(i)))
            self.decoder_weights.append(tf.placeholder(dtype=dtype,shape=[None],name="decoder_weights_{}".format(i)))
        targets = [ self.decoder_inputs[i+1] for i in range(buckets[-1][1])] #真实结果把起始符GO去掉了

        def sampled_loss(labels,logits):
            labels = tf.reshape(labels,[-1,1]) #转成1列
            return tf.nn.sampled_softmax_loss(
                weights=tf.transpose(w),biases=b,labels=labels,inputs=logits,num_sampled=num_samples,num_classes=self.decoder_vocb_size
            )

        def seq2seqFunction(encoder_inputs,decoder_inputs,use_prev):
            temp_cell = copy.deepcopy(cell)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs = encoder_inputs,
                decoder_inputs = decoder_inputs,
                cell = temp_cell,
                num_encoder_symbols = encoder_vocb_size,
                num_decoder_symbols = decoder_vocb_size,
                embedding_size = lstm_size, #这个和上两个变量主要用于embedding_lookup，即从总的里寻找出这么些个
                output_projection = output_projection,
                feed_previous = use_prev,
                dtype = dtype
            )

        if not forTrain:
            self.outputs,self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                encoder_inputs = self.encoder_inputs,
                decoder_inputs = self.decoder_inputs,
                targets = targets,
                weights = self.decoder_weights,
                buckets = self.buckets,
                seq2seq = lambda x,y: seq2seqFunction(x,y,False),
                softmax_loss_function = sampled_loss
            )
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                encoder_inputs=self.encoder_inputs,
                decoder_inputs=self.decoder_inputs,
                targets=targets,
                weights=self.decoder_weights,
                buckets=self.buckets,
                seq2seq=lambda x, y: seq2seqFunction(x, y, True),
                softmax_loss_function=sampled_loss
            )

        if output_projection is not None:
            for bucket in range(len(buckets)):
                self.outputs[bucket] = [ tf.matmul(output,output_projection[0]) + output_projection[1]
                                         for output in self.outputs[bucket]]

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if forTrain:
            self.gradient_norms = []
            self.updates = []
            for output,loss in zip(self.outputs,self.losses):
                gradients = tf.gradients(loss,params)
                list_clipped_gradients, use_norm = tf.clip_by_global_norm(gradients,clip_norm)
                self.gradient_norms.append(use_norm)
                self.updates.append(opt.apply_gradients(zip(list_clipped_gradients,params)))

        self.saver = tf.train.Saver(tf.global_variables())

    def get_batch_data(self,batch_size,bucket_id):
        #获取batch_size大小的问答对
        #return data,data_in
        return dataProcess.get_batch_data(batch_size=batch_size,bucket_id=bucket_id)

    def step(self,session,encoder_inputs,decoder_inputs,decoder_weights,bucket_id,forTrain):
        #确定训练和测试时feed_input和feed_output的对应值
        encoder_size,decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,%d != %d." % (len(encoder_inputs), encoder_size)
            )
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket, %d != %d." % (len(decoder_inputs), decoder_size)
            )
        if len(decoder_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket, %d != %d." % (len(decoder_weights), decoder_size)
            )

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]
        last_decoder = self.decoder_inputs[decoder_size].name
        input_feed[last_decoder] = np.zeros([self.batch_size],dtype = np.int32) #因为EOS是对应的0

        if forTrain:
            output_feed = [self.gradient_norms[bucket_id],self.updates[bucket_id],self.losses[bucket_id]]
            for i in range(self.buckets[bucket_id][1]):
                output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed = [self.losses[bucket_id]]
            for i in range(self.buckets[bucket_id][1]):
                output_feed.append(self.outputs[bucket_id][i])

        outputs = session.run(output_feed,feed_dict = input_feed)
        if forTrain:
            return outputs[2],outputs[3:]
        else:
            return outputs[0],outputs[1:]

    def get_batch(self,bucket_id,data):
        encoder_size,decoder_size = self.buckets[bucket_id]

        _, word_id, _ = dataProcess.load_dict()
        GO,PAD,EOS = '<go>','<pad>','<eos>'

        encoder_inputs,decoder_inputs = [],[]  #batch_size * steps
        print(data)
        for ask,answer in data:
            encoder_input = dataProcess.get_id_by_dict(ask)
            decoder_input = dataProcess.get_id_by_dict(answer)

            encoder_pad = [word_id[PAD]]  * (encoder_size-len(encoder_input))
            encoder_input = encoder_pad + encoder_input

            decoder_pad = [word_id[PAD]]  * (decoder_size-len(decoder_input)-2)
            decoder_input = [word_id[GO]] + decoder_input + [word_id[EOS]] + decoder_pad

            encoder_input = encoder_input[:encoder_size]
            decoder_input = decoder_input[:decoder_size]

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        weights = []
        for idx in range(self.batch_size):
            batch_weight = np.ones(decoder_size, dtype=np.float32)
            for j in range(decoder_size):
                if j == decoder_size-1 or decoder_inputs[idx][j] == word_id[PAD]:
                    batch_weight[j] = 0
            #其实也可以直接一句话，就是不太好看
            # batch_weight = [ 0.0 if (j == decoder_size-1 or decoder_inputs[idx][j] == word_id[PAD]) else 1.0 for j in range(decoder_size)]
            weights.append(batch_weight)

        #由于前面定义encoder_inputs、decoder_inputs和weights时都是按照字数来定义的，故而也就是第一维度需要是对应的第几个字。
        #所以这里需要转置
        batch_encoder_inputs = np.array(encoder_inputs,dtype=np.int32).transpose()
        #batch_encoder_inputs = [ batch_encoder_inputs[i] for i in range(batch_encoder_inputs.shape[0]) ]

        batch_decoder_inputs = np.array(decoder_inputs,dtype=np.int32).transpose()
        #batch_decoder_inputs = [ batch_decoder_inputs[i] for i in range(batch_decoder_inputs.shape[0]) ]

        batch_weights = np.array(weights,dtype=np.float32).transpose()
        # batch_weights = [ batch_weights[i] for i in range(batch_weights.shape[0]) ]

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights



