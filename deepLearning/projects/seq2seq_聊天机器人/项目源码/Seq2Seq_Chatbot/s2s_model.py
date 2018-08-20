#encoding=utf8
import pdb
import random
import copy

import numpy as np
import tensorflow as tf

import data_utils

class S2SModel(object):
    def __init__(self,
                source_vocab_size,
                target_vocab_size,
                buckets,
                size,
                dropout,
                num_layers,
                max_gradient_norm,
                batch_size,
                learning_rate,
                num_samples,
                forward_only=False,
                dtype=tf.float32):
        # init member variales
        self.source_vocab_size = source_vocab_size   #源句子中可能用到的字的总个数，也就是字典中字的总数
        self.target_vocab_size = target_vocab_size   #目标句子中可能的字的总个数，因为都是中文，故而用的一个字典，故而字数一致
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # LSTM cells
        cell = tf.contrib.rnn.BasicLSTMCell(size) #指定lstm内部维度数：512，即代表网络接收数据的维度数,相当于卷积核个数，控制着输出维度
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout) #即添加一个dropout
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers) 
        #纵向上创建2层lstm，2层只是为了能提取到一些深层的特征信息,再往上，3层4层，也可以，但是要看效果

        output_projection = None
        softmax_loss_function = None

        # 如果vocabulary太大，我们还是按照vocabulary来sample的话，内存会爆
        if num_samples > 0 and num_samples < self.target_vocab_size:
            print('开启投影：{}'.format(num_samples))
            w_t = tf.get_variable(
                "proj_w",
                [self.target_vocab_size, size], #字的长度*512,因为从lstm输出出来的即为每个字512维
                dtype=dtype
            )
            w = tf.transpose(w_t) #转置成 512*字的长度
            b = tf.get_variable(
                "proj_b",
                [self.target_vocab_size],
                dtype=dtype
            )
            output_projection = (w, b) #因为得到的数字也是512维中的，若字典太大，则需要进行往更大的字典维度上投影
            #w:512*字典总字数

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # 因为选项有选fp16的训练，这里统一转换为fp32
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),
                    dtype
                )
            softmax_loss_function = sampled_loss
            #这里相当于只是起了一个别名，用了自己定义的损失

        # seq2seq_f
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # Encoder.先将cell进行deepcopy，因为seq2seq模型是两个相同的模型，但是模型参数不共享，所以encoder和decoder要使用两个不同的RnnCell
            tmp_cell = copy.deepcopy(cell) # 这里的cell即上面定义的双层lstm，这里编码器和解码器都是用的双层lstm，即cell结构一致
            
                #cell:                RNNCell常见的一些RNNCell定义都可以用.
                #num_encoder_symbols: source的vocab_size大小，用于embedding矩阵定义
                #num_decoder_symbols: target的vocab_size大小，用于embedding矩阵定义
                #embedding_size:      embedding向量的维度
                #num_heads:           Attention头的个数，就是使用多少种attention的加权方式，用更多的参数来求出几种attention向量
                #output_projection:   输出的映射层，因为decoder输出的维度是output_size，所以想要得到num_decoder_symbols对应的词还需要增加一个映射层，参数是W和B，W:[output_size, num_decoder_symbols],b:[num_decoder_symbols]
                #feed_previous:       是否将上一时刻输出作为下一时刻输入，一般测试的时候置为True，此时decoder_inputs除了第一个元素之外其他元素都不会使用。
                #initial_state_attention: 默认为False, 初始的attention是零；若为True，将从initial state和attention states开始。
            #tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                #encoder_inputs, 
                #decoder_inputs, 
                #cell, 
                #num_encoder_symbols, 
                #num_decoder_symbols, 
                #embedding_size, 
                #num_heads=1, 
                #output_projection=None, 
                #feed_previous=False, 
                #dtype=None, 
                #scope=None, 
                #initial_state_attention=False)    
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq( #seq2seq with attention的接口
                encoder_inputs,# tensor of input seq  #10
                decoder_inputs,# tensor of decoder seq  #20
                tmp_cell,#自定义的cell,可以是GRU/LSTM, 设置multilayer等
                num_encoder_symbols=source_vocab_size,# 词典大小 40000  #6865
                num_decoder_symbols=target_vocab_size,# 目标词典大小 40000  #6865
                embedding_size=size,# embedding 维度  #512
                output_projection=output_projection,# 输出的映射层，因为decoder输出的维度是output_size，所以想要得到num_decoder_symbols对应的词还需要增加一个映射层
                feed_previous=do_decode,  #False  #训练的时候是否用上一个输出来喂入，训练时有真实值故而不需要这个，必须为False，即对应于y[i]值
                dtype=dtype
            )

        # inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_weights = []
        #encoder_inputs 这个列表对象中的每一个元素表示一个占位符，其名字分别为encoder0, encoder1,…,encoder39，encoder{i}的几何意义是编码器在时刻i的输入。
        # buckets中的最后一个是最大的（即第“-1”个）
        for i in range(buckets[-1][0]): #buckets即[(5,15),(10,20),(15,25),(20,30)]，故而buckets[-1][0]即为20
            #这里即创建20个占位符，都放到encoder_inputs里，因为输入的问答对长度里问句最多是20个字
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='encoder_input_{}'.format(i) ))
        # 输出比输入大 1，这是为了保证下面的targets可以向左shift 1位
        for i in range(buckets[-1][1] + 1):
        #输出最多30个字，故而这里创建30+1个占位符，这个1是为了告诉解码器遇到它就可以停止了，所以也不需要额外的输出什么了属于解码器最后一个输入
        #解码要对应解码维度和权重维度，权重维度和解码字数一一对应的。即attention里的权重矩阵个数,是与解码的字数一样的
            self.decoder_inputs.append(tf.placeholder( #decode_inputs即解码器的输入
                tf.int32,
                shape=[None],
                name='decoder_input_{}'.format(i)
            ))
            self.decoder_weights.append(tf.placeholder(
                dtype,
                shape=[None],
                name='decoder_weight_{}'.format(i)
            ))
            #target_weights 是一个与 decoder_outputs 大小一样的 0-1 矩阵。该矩阵将目标序列长度以外的其他位置填充为标量值 0。
                # Our targets are decoder inputs shifted by one.
            #target是训练集中解码器的应该的输出，也就是输入给解码器的y[i]，因为decoder_inputs是GO+decoder_inputs+EOS+Padding的。
            #所以真实的，或者说目标的即从第1个开始
        targets = [self.decoder_inputs[i + 1] for i in range(buckets[-1][1])]
        #这里的targets是24*64 decoder_inputs[0]全是3，因为做了转置后，第0号元素是所有句子的第一个字，因为第一个字都是GO

# 跟language model类似，targets变量是decoder inputs平移一个单位的结果，
    #encoder_inputs: encoder的输入，一个tensor的列表。列表中每一项都是encoder时的一个词（batch）。
        #decoder_inputs: decoder的输入，同上
        #targets:        目标值，与decoder_input只相差一个<EOS>符号，int32型
        #weights:        目标序列长度值的mask标志，如果是padding则weight=0，否则weight=1
        #buckets:        就是定义的bucket值，是一个列表：[(5，10), (10，20),(20，30)...]
        #seq2seq:        定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq，embedding_rnn_seq2seq，basic_rnn_seq2seq等
        #softmax_loss_function: 计算误差的函数，(labels, logits)，默认为sparse_softmax_cross_entropy_with_logits
        #per_example_loss: 如果为真，则调用sequence_loss_by_example，返回一个列表，其每个元素就是一个样本的loss值。如果为假，则调用sequence_loss函数，对一个batch的样本只返回一个求和的loss值，具体见后面的分析
        #name: Optional name for this operation, defaults to "model_with_buckets".

        if forward_only:# 测试阶段 测试阶段没有真实decoder_input了，只能将前一次输出作为这一次输入
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.decoder_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, True), 
                #测试和训练时这个第三个参数是主要差别。即decode时是否需要前面一个的输出。
                #因为是测试，此时decoder_input还没有内容，因此输入的y即padding后的decoder_input为[GO]+decoder_input+[EOS]+[Padding]
                #这里因为没有decoder_input，故而数字矩阵即[3,0,2,2,2,2,2...]
                #True则代表会使用前一次的输出作为本次的输入，但是第一次时会使用y的输入值，因为y的第一个为GO，代表开始解码，后面就不需要y了
                softmax_loss_function=softmax_loss_function
            )
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(
                            output, #因为output都是从lstm网络里出来的，故而输出的最后一个维度为512
                            output_projection[0]  # 512*6865
                        ) + output_projection[1]  # 6865
                        for output in self.outputs[b]
                    ]  # 不为空，说明需要做投影操作
        else:#训练阶段
            #将输入长度分成不同的间隔，这样数据的在填充时只需要填充到相应的bucket长度即可，不需要都填充到最大长度。
            #比如buckets取[(5，10), (10，20),(20，30)...]（每个bucket的第一个数字表示source填充的长度，
            #第二个数字表示target填充的长度，eg：‘我爱你’-->‘I love you’，应该会被分配到第一个bucket中，
            #然后‘我爱你’会被pad成长度为5的序列，‘I love you’会被pad成长度为10的序列。其实就是每个bucket表示一个模型的参数配置），
            #这样对每个bucket都构造一个模型，然后训练时取相应长度的序列进行，而这些模型将会共享参数。
            #其实这一部分可以参考现在的dynamic_rnn来进行理解，dynamic_rnn是对每个batch的数据将其pad至本batch中长度最大的样本，
            #而bucket则是在数据预处理环节先对数据长度进行聚类操作。明白了其原理之后我们再看一下该函数的参数和内部实现：
            #encoder_inputs: encoder的输入，一个tensor的列表。列表中每一项都是encoder时的一个词（batch）。
            #decoder_inputs: decoder的输入，同上
            #targets:        目标值，与decoder_input只相差一个<EOS>符号，int32型
            #weights:        目标序列长度值的mask标志，如果是padding则weight=0，否则weight=1
            #buckets:        就是定义的bucket值，是一个列表：[(5，10), (10，20),(20，30)...]
            #seq2seq:        定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq，embedding_rnn_seq2seq，basic_rnn_seq2seq等
            #softmax_loss_function: 计算误差的函数，(labels, logits)，默认为sparse_softmax_cross_entropy_with_logits
            #per_example_loss: 如果为真，则调用sequence_loss_by_example，返回一个列表，其每个元素就是一个样本的loss值。如果为假，则调用sequence_loss函数，对一个batch的样本只返回一个求和的loss值，具体见后面的分析
            #name: Optional name for this operation, defaults to "model_with_buckets".            
            #tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, 
                                                        #decoder_inputs, 
                                                        #targets, 
                                                        #weights, 
                                                        #buckets, 
                                                        #seq2seq, 
                                                        #softmax_loss_function=None, 
                                                        #per_example_loss=False, 
                                                        #name=None)
            
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets( #seq2seq的分桶算法，每个bucket不同的问答长度
                self.encoder_inputs,     #编码器的输入 #20
                self.decoder_inputs,     #解码器的输入 #31
                targets,                 #编码器的预测输出 #30
                self.decoder_weights,    #解码器的权重，即每个解码输出字对应的所有encoder的权重 #31
                buckets,                 #问答长度的桶
                lambda x, y: seq2seq_f(x, y, False),        
                # 传进来的一个匿名函数，内部处理，x，y实际上一个是encoder_input,另一个是decoder_input，第三个是是否需要decodder的前一个输出作为这一次的输入
                #因为是训练阶段，所以不需要前面的输出作为这次的输入，直接拿真实语句输入，更准确
                softmax_loss_function=softmax_loss_function # 损失函数
            )
            #这里的outputs应该也需要判断一下是否需要投影，如果是，也需要往高维映射。因为训练和测试都需要，所以可以把那个挪出判断条件

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate )

        if not forward_only:# 只有训练阶段才需要计算梯度和参数更新
            self.gradient_norms = []
            self.updates = []
            for output, loss in zip(self.outputs, self.losses):# 用梯度下降法优化每一个桶的参数
                gradients = tf.gradients(loss, params) 
                #通过损失对需要更新的参数进行求梯度，更新。这里的损失是总损失
                #tf.gradients最重要的即两个参数，一个即ys，这里即loss，另一个即xs，这里即要更新的params，即用ys对xs进行求偏导
                clipped_gradients, norm = tf.clip_by_global_norm( gradients,   max_gradient_norm   ) #进行梯度截断，以5为上界，防止梯度爆炸
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params)  ))  #反向传播，更新参数
        # self.saver = tf.train.Saver(tf.all_variables())
        self.saver = tf.train.Saver(tf.global_variables(),write_version=tf.train.SaverDef.V2)

    
    
    def step(self,session,encoder_inputs,decoder_inputs,decoder_weights,bucket_id,forward_only):
        #根据模式，定义好需要喂入的参数以及对应输出的内容
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size)
            )
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size)
            )
        if len(decoder_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_weights), decoder_size)
            )

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i] #放到对应的名字中去
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]  #decoder_weights:25*64的列表
        #将参数做成feed_dict形式
        
        # 理论上decoder inputs和decoder target都是n位
        # 但是实际上decoder inputs分配了n+1位空间
        # 不过inputs是第[0, n)，而target是[1, n+1)，刚好错开一位
        # 最后这一位是没东西的，所以要补齐最后一位，填充0
        last_target = self.decoder_inputs[decoder_size].name #按名字赋值
        #因为前面decode的占位符定义是从0到decoder_size+1,是为了给target多腾出一个位置可以对应到结尾
        #而这里喂值的时候还只是从0到decoder_size,因为weight只需要那么多位，decoder_inputs和weight先一起喂了那么多，故而还差了一位没给
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id], self.gradient_norms[bucket_id],self.losses[bucket_id]]
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed = [self.losses[bucket_id]]
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])

        
        outputs = session.run(output_feed, feed_dict=input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:]  
            #训练模式下，第0个只是为了更新参数，没必要输出，第1个是更新的梯度，第二个是损失，后面是outputs，故而主要要后两个
        else:
            return None, outputs[0], outputs[1:]
            #测试模式下，第0个即为损失，后面的即为outputs，但是为了保证和上面一样是三个输出，故而第一个为None,也是没啥实际作用

    def get_batch_data(self, bucket_dbs, bucket_id):
        #随意获取batch_size个问答对，以及答问对
        data = []
        data_in = []
        bucket_db = bucket_dbs[bucket_id]
        for _ in range(self.batch_size):
            ask, answer = bucket_db.random()
            data.append((ask, answer))
            data_in.append((answer, ask))
        return data, data_in

    def get_batch(self, bucket_dbs, bucket_id, data):
        #将encoder_input和decoder_input转成ID，并打上padding
        #其中encoder的padding打在前面，decoder的padding打在后面
        #同时decoder需要加上开始和结束标志，即GO和EOS
        #为后面解读方便，将64句话*15个字，转成15个字*64句话，这样，按15这个维度遍历时，就能依次取到每个句子的第i个字了
        #然后设置权重，如果是padding的位置，或者是句子的最后一个字，就设置为0，其它的为1
        encoder_size, decoder_size = self.buckets[bucket_id]
        # bucket_db = bucket_dbs[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for encoder_input, decoder_input in data:
            # encoder_input, decoder_input = random.choice(data[bucket_id])
            # encoder_input, decoder_input = bucket_db.random()
            #把输入句子中每个字转化为字典id
            encoder_input = data_utils.sentence_indice(encoder_input)
            decoder_input = data_utils.sentence_indice(decoder_input)
            # Encoder
            encoder_pad = [data_utils.PAD_ID] * (
                encoder_size - len(encoder_input)
            )
            #空格放前面，增强句子向量的语义信息
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            #因为lstm随着长度增长，最前面的信息会有部分丢失，而这里加的padding，是属于噪音数据，让它放前面即相当于有效信息增强了
            #这个操作能提升准确度
            #但是这里是倒着放句子的，不然就可以直接list(encoder_pad + encoder_input)
            #之所以倒着放，是因为谷歌有一篇机器翻译的论文，上面说将句子倒置能提升准确度，实际上那应该是针对于英文有效果。
            #因为英文的疑问词都是放句首，翻转后即将疑问词等放在句尾，信息丢失。而中文疑问词一般放句尾，所以这里不翻转应该效果会更好。
            # Decoder
            decoder_pad_size = decoder_size - len(decoder_input) - 2
            #减2是为了空出两个标志位，一个是开始解码标志 GO_ID,另一个是停止解码标志 EOS_ID，因此到了EOS_ID则解码停止
            decoder_inputs.append(
                [data_utils.GO_ID] + decoder_input +
                [data_utils.EOS_ID] +
                [data_utils.PAD_ID] * decoder_pad_size
            )
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # batch encoder
        for i in range(encoder_size):
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
        #把encoder_inputs转换成ndarray格式，同时行列互换。batch_encoder_inputs：15*64的list encoder_inputs：64*15的list
        #故而实际上可以两句话搞定，这里64表示一次64句话，15表示每句话15个字
        #temp = np.array(encoder_inputs,dtype=np.int32).transpose()
        #batch_encoder_inputs = [ temp[i] for i in range(temp.shape[0]) ] 
        #这里虽然行列互换了，但是因为本身是字对字的预测，所以每个字有对应的权重，并且encoder和decoder也是每个字相对应的
        #故而测试时换不换也无所谓，但作为统一操作，故而测试时因为会调用函数get_batch故而也会换
        # batch decoder
        for i in range(decoder_size): #decoder_size即解码的长度，即解码后句子的长度
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for j in range(self.batch_size): #batch_size即句子个数，即按批次操作，每次操作多个句子，每次对所有句子的某个字进行操作
                if i < decoder_size - 1:
                    target = decoder_inputs[j][i + 1]
                    #进行位置对应
                if i == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[j] = 0.0
                    #若是最后一个字或者是padding的，则置权重为0
                    #这里用np.zeros来初始化能少些代码
            batch_weights.append(batch_weight)
            #64*10  ,64*20 ,64*20
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
