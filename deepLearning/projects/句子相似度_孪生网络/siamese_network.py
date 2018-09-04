import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn 
class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
    #从名字上看即可以看出是双向RNN，这里的RNN用的是lstm
        n_hidden=hidden_units #lstm内部维度数，可以理解为神经元个数，即输出的维度数，这里是50，有点少，一般100到300
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))  #默认unstack的axis=0
        #起初x是?*15*300,即batch_size*句子长度*每个字的维度。转置之后即15*?*300
        #unstack即进行横向拆解,即变成15*[?*300],相当于拆成15个元素，也就是每批次都逐字来进行网络训练
        print(x)
        
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope): #这里两个同时写，即扁平化，能高效一些
        #否则分两层写，则属于双层结构了。效率会低一点
        #正常而言定义了variable_scope是因为要去使用tf.getVariable(),但这里并没有，故而其实写不写无所谓，只是为了好管理
            stacked_rnn_fw = []
            for _ in range(n_layers):#3层lstm网络其实可以加上残差连接了
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)#初始化一个向前的lstm
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)#使用一个dropout
                stacked_rnn_fw.append(lstm_fw_cell) #添加到列表里，因为是3层，故而循环三次
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)#多层lstm
           #这里即先定义向前的lstm,然后下面再定义向后的lstm，虽然向前和向后结构一样，但是作用不一样。
           #这里用MultiRNNCell组合多个lstm，从而得到多层lstm
        
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #这里通过static_bidirectional_rnn即构成静态的双向lstm拼接，也可以使用动态的
            #重点：这里用的是静态的双向lstm
            #带有static前缀的api要求输入的序列具有固定长度。即所有批次，不同batch_size的句子长度也都需要一样
            #带有dynamic前缀的api可以选择输入一个sequence_length（可以是一个list）参数。
            #即每batch_size内部句子长度一样，但不同batch_size的句子长度可以不一样，这种就可以使用局部padding了
            #该参数对应的是输入sequence的序列长度，用来动态处理sequence的长度
            #因此这里显然可以提升，因为所有长度一样就意味着要全局padding，没有局部padding来得信息有效率高
            #但是用动态的因为每batch_size处理完的数据长度不一样，故而还要做一些处理，比如concat起来。
            #因为RNN都是要取的最后一个字处理完后的总的结果输出，故而句子长度即使不一样，出来的结果长度还是一样的
            #比如直接用条件随机场来作为分类器，处理损失，因为条件随机场不仅会考虑状态转移，而且可以指定长度
            #但使用softMax则不行
            #序列标注需要看所有的字，所以需要看所有字最后输出的维度。而这个项目是文本相似度，用的lstm，看最后一个字结果输出。
            #故而若最后一个字输出的维度一致，也可以。即使用局部padding之后也能获取到等长结果
            
            #正常返回是三个值：outputs, output_state_fw, output_state_bw
            #第0个为纵向的结果输出，第一个为向前的横向的状态输出，第二个是向后的横向的状态的输出
            #纵向上的结果输出，即可以直接拿过来解码获得答案的
            
            top_states = [ tf.reshape(e,[-1,1,100]) for e in outputs ]
            attention_states = tf.concat(top_states,1)
            attn_length = attention_states.get_shape()[1].value
            if attn_length is None:
                attn_length = tf.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value
            
            hidden = tf.reshape(attention_states,[-1,attn_length,1,attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size #512
            w = tf.get_variable("AttnW",[1,1,attn_size,attention_vec_size])
            
        hidden_features.append(tf.nn.conv2d(hidden,w,[1,1,1,1],"SAME"))  #1*1的卷积核表示对每个字进行一下特征抽取
        #后面再加一层3*3的卷积，然后将1*1的卷积结果和3*3的卷积结果拼接起来，应该会效果更好。因为3*3的卷积即将前后几个字的关联性特征也抽取出来了
        #这里一定要试一下
        hidden_features = tf.reshape(hidden_features,[-1,100*15])
            
        return hidden_features
        #return outputs[-1]
        #这里直接双向lstm之后网络最后出的结果测试准确度大概只有66%多
        #改进：对outputs[-1]进行一次卷积操作,能提升效果至90多
    
        #因为上面已经做过转置，所以最后的输出即最后一个字对应的时间维度输出
        #因为这里的outputs是时间步长为15的总输出，取最后一个表示只要最后一个的结果。
        #其实这里可以加上attention信息来提升准确度
    
    def contrastive_loss(self, y,d,batch_size):
        #性质上讲和逻辑回归的损失有点像，即都考虑了两种取值的情况。
        #孪生网络的对比损失函数：L = 1/2N * sum(y*d^2+(1-y)max(margin-d,0)^2)
        # 在损失函数中，y代表是否匹配的标签，y=0表示不匹配，y=1表示匹配，故而这里的损失需要考虑实际中y的情况来决定是否要换
        # 当y=0时，表示不相似，此时前半段为0，若两个样本特征空间距离d越小的话，代表损失应该要越大，
        # 前面之所以对欧式距离d做了除总模长的操作(要保证值最大也就为1)，是因为这里的阈值是要用1去减。
        # 当y=1时，表示样本特征相似，此时后半段为0，此时d越大，则损失越大因为d在前面做了处理，肯定小于1，故而平方一下则缩小点，加快收敛
        # margin为设定的阈值，d为两个样本特征输出的欧氏距离
        # 这里margin设为了1
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):
        #先初始化网络
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,name="W")  
            #这里的W用Variable来初始化，然后trainable=True(默认也是True)，表明是会被更新的
            #这个w并不是lstm内部的权重参数，而是总表，因为没有使用现成的word2vec(基于语义训练的),所以就随机初始化了
            #虽然这里先随机初始化了，但是外面加载表依然可以把数据加载进来
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            #input_x1为?*15大小，w是15381*300，故而embedded_chars1则为?*15*300
            #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)
            #这里input_x2大小与input_x1一样，故而结果也和上面一样大小
            #这里因为是走的语法相似的，没有去获取字向量文件，因为字向量文件是基于语义训练出来的，故而这里都是用的随机初始化

            #这里显然有个缺点，即所有句子都是15*300维，故而也可以理解是定义了一个max_length后，进行了全局的句子截断和padding
            #即最多15个字，少了也补成15个字，每个字300维向量，可以改进，改进成分批等长,参考命名实体识别
            
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
            self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
            #这里因为是构建的孪生网络,故而两个网络结构一致,只是最后对应的权重参数等会不一样
            #这里两个LSTM网络，其w和b由BasicLSTMCell内部随机初始化了，故而两个网络的初始W和b应该就不一样
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            #即求两个向量的欧氏距离,keep_dims=True表示保持维度，因为它本身是二维的矩阵，而不是一维的向量
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            #这里是难点，也是和孪生网络损失函数有关系的重点
            #这一步感觉完全只是为了缩小一下范围，让其整体肯定小于1。因为两个向量连接后最长也就是两个模长相加，即头尾相连的情况
            #因为求损失是孪生网络的对比损失，涉及到一个margin-d的操作，margin是自己设定的阈值，故而对d做一下范围缩小操作
            #如果不做范围缩小操作的话，那么两个向量的距离则是[0,两个模长之和]，故而直接在这里除一下，也方便一点后面设置阈值
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            #原本是?*1，现在reshape成1维的
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
            #求损失，这里的损失即孪生网络的对比损失函数
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            #tf.rint代表四舍五入，tf.ones_like代表全一，大小为self.
            #原本欧式距离是越小越相似，四舍五入后再用1去减，则变成越相似越大了，这里只有01两个值，故而相似的即为1了。
            #但是这里相当于以0.5作为区分，也可以精度提升一点，比如以0.6,0.7等作为界限，大于该值，则置为1，即欧式距离小于0.4,0.3才认为相似
            #应该能提升一点准确度
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
