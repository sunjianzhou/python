# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob,iob_iobes


class Model(object):
    #初始化模型参数
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"] #字设定100维
        self.lstm_dim = config["lstm_dim"] #如果没有lstm，则代表卷积核的个数，复用了一下变量，也是100个
        self.seg_dim = config["seg_dim"]  #切词信息给定20维
 
        self.num_tags = config["num_tags"] #每个字可能对应的标签数 51  即那些B-Tes，I-Tes，E—Tes，S-Tes等等
        self.num_chars = config["num_chars"]#样本中总字数
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)  #trainable=False的表明不需要用梯度去更新
        self.best_dev_f1 = tf.Variable(0.0, trainable=False) #验证集上的f1值，因为只是最后评价一下用，故而也不需要去梯度更新它
        self.best_test_f1 = tf.Variable(0.0, trainable=False)  #测试集上的f1值
        self.initializer = initializers.xavier_initializer() #这种初始化能保证输出和输入尽可能地服从相同的概率分布

        # add placeholders for the model
        
        #输入的每个字
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None], #一个是batchsize，一个是每句话中字的长度，即外面大列表中的第1项
                                          name="ChatInputs") 
        #切词信息
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],#一个是batchsize，一个是每句话中字的长度，即外面大列表中的第2项
                                         name="SegInputs")
        #标签
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],#一个是batchsize，一个是每句话中字的长度，即外面大列表中第3项
                                      name="Targets") 
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs)) #都转换成-1和1
        #因为char_inputs为batch_size*每句话的长度，故而两个数都是正的，加一个abs更保证一下，然后sign即让所有位置的数变成1了
        length = tf.reduce_sum(used, reduction_indices=1) #根据第1维度求和，此时即得到了本batch_size中每句话的总字数了，即句子长度
        self.lengths = tf.cast(length, tf.int32) #类型转换
        self.batch_size = tf.shape(self.char_inputs)[0]  #因为self.char_inputs为batch_size*每句话中字的个数,故而0即batch_size
        self.num_steps = tf.shape(self.char_inputs)[-1]  #一句话中字的个数
        
        
        #Add model type by crownpku bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        #以下三个是膨胀卷积网络的系数，1表示不插孔，和正常的卷积网络一样。
        #2表示插一个孔
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3 #卷积核大小
        self.num_filter = self.lstm_dim  #卷积核个数 = lstm的维度数 100 因为使用lstm时不需要卷积核，使用卷积核时不需要lstm维度，故而复用
        self.embedding_dim = self.char_dim + self.seg_dim  #嵌入的维度数 = 字的维度数 + 分词的维度数
        #这里的seg_dim是20，所以分词的维度应该是被映射压缩了。
        self.repeat_times = 4   #使用多少个分支
        self.cnn_output_width = 0
        
        # embeddings for chinese character and segmentation representation
        # 这里转换获得embedding，即为网络的输入数据，即所有的文本矩阵
        # 故而这个转换很重要，即将切词信息提取嵌入到字信息中
        # 字信息，切词信息，和特征信息
        #得到的结果是 batch_size * 该batch_size中每句话的字数 * 每个字对应的120维数字信息
        #就相当于 batch_size * width * channel，即N*W*C,和正常的卷积相比缺少了H维度信息
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        
        #下面的整体步骤流程即：定义模型，计算损失，定义优化器，优化器根据损失更新参数，保存模型
        
        if self.model_type == 'bilstm':  #双向lstm
            # apply dropout before feed to lstm layer
            #输入层进行dropout，这里dropout只是模型压缩，即神经元随机抽取，但是对数据不会改变
            model_inputs = tf.nn.dropout(embedding, self.dropout) 

            # bi-directional lstm layer
            #中间层进行双向lstm
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths) #使用双向lstm，指定内部维度数和解码长度

            # logits for tags
            # 线性映射，输出得到预测的结果
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            #和lstm一样先进行dropout
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            # 放入到膨胀卷积神经网络,获得输出
            model_outputs = self.IDCNN_layer(model_inputs)
            #得到的输出为[batch_size*每句中字数，每字的特征数字(400维)]
            #400维是四次重复，每次进行三次卷积，每次卷积结果得到100维数据，最后将第三维数据tf.concat,总共4次，即变成400
            #然后reshape，合并前两维数据，返回回来

            # logits for tags
            # 通过全连接得到预测结果 logits = WX+b 得到具体的类别
            #因为实际是X*W+B，所以W的维度即[400,label_nums],label_nums为所有的标签类别数，即self.num_tags
            #故而得到[batch_size*每句中字数,label_nums]的结果，然后reshape成[batch_size,每句中字数,label_nums]
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        # loss of the model
        #计算损失,使用条件随机场计算
        #因为我们这里处理的时候是每个batch_size内的所有句子是等长的，但是不同batch_size之间的句子不一定等长，所以需要用条件随机场来处理
        #其中，logits是网络计算后的结果:[batch_size,每句中字数,label_nums]
        #而self.lengths即本次batch_size中的句子长度。
        self.loss = self.loss_layer(self.logits, self.lengths)
        
        #计算出损失后，就可以通过优化器去优化参数了，这里提供了常见的三种优化器
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

            # apply grad clip to avoid gradient explosion
            #下面三步即相当于优化器以一个学习率minimize(loss)的过程，因为优化器minimize(loss)的过程即将计算梯度和更新参数合在一起
            #分开写，能单独看到每次的梯度更新情况，其变化状态，并且能加进来一步梯度截断的过程，故而拆开三步
            grads_vars = self.opt.compute_gradients(self.loss) #根据损失，计算梯度
            #返回的是 A list of (gradient, variable) pairs.故而grads_vars是一个列表
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars] #截断梯度，比如大于5的梯度都设置为5
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step) #更新参数
            #即将梯度变化更新到参数上
            
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        #到这里，main2里的create_model(sess,Model,FLAGS.ckpt_path,load_word2vec,config,id_to_char,logger)也就ok了

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        #比如当前传过来的这一批次是10句话*每句50个字，那么char_inputs, seg_inputs都是10*50
        #然后自己分别对字数据和切词数据初始化了一个总数据集，现在是随机初始化的，通过initializers.xavier_initializer()这个初始化方式
        #即一个总字数*100维和一个总切词状态数*20维的数据集。后面如果有word2vec的字向量信息，则相当于能把这部分替换了应该。
        #这里用总字数和总切词状态数是为了保证后面查询都是有对应的数据可以查的
        #当然，加载已经训练好的字向量其准确率应该会高一些，但是随机初始化数字也可以，随机初始化依然能保证每个字都有唯一的120维向量来表示。 
        #如果有已经训练好的字向量的话，即拿着这些ID的数字去字向量文件里查即可。咱这里好像还真有字向量文件，只是这个字向量文件不一定匹配，对于找不到的也得随机初始化，故而整体随机初始化得了。
        #然后拿着char_inputs, seg_inputs两个10*50的具体数字去总数据集里取相应位置的数字，也就能得到10*50*100和10*50*20的两个随机数据集
        #由于这两个随机数据集都是列表embedding通过append的方式添加的，所以列表大概长这样[[10*50*100],[10*50*20]],这里10为batch_size，50为当前每个句子的字数
        #然后将两个随机数据集进行最后一个维度的concat，即变成了10*50*120，也就得到了当前这一批次，有着若干句话的数字信息了。
        
        #所以这里相当于我们对每个字信息随机初始化了一个总字数*100维的随机数，训练集大的时候可以调整成150维，200维。
        #每次对每批次的字，则通过ID信息去这总字数*100维里抽取相应位置的数。
        #然后再随机初始化了总切词状态数*20维的随机数。看起来这里只占了字信息维度的1/5。
        #然后通过每批次的字的切词信息数字，去这总切词状态数（4）*20里抽取相应位置的数。
        #char_inputs即对应每个字的ID，seg_inputs即对应每个字的切词信息，bies共四个状态，config即复合完整后的配置文件
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test=char_inputs #batch_size*num_steps 即多少句话*每句话多少字 是2维的
        self.seg_inputs_test=seg_inputs #即多少句话*每句多少个切词信息(字)
        #接下来是要将每个字变成一个向量
        with tf.variable_scope("char_embedding" if not name else name):
            #创建的char_lookup即为一个字的数字向量
            self.char_lookup = tf.get_variable( #char_lookup即为总表
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],  #初始化总表的维度大小 即总的字的个数*每个字的维度，这里还没有切词信息
                    initializer=self.initializer)           #字的长度根据外面获取，字的维度自己定义的100
                    #这里初始化以xavier_initializer来进行初始化，即尽可能保证输入输出是属于同一个概率分布
            #输入char_inputs='常' 对应的字典的索引/编号/value为：8
            #self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            #先用字对应的ID去获取数字信息
            #embedding_lookup即查询，char_lookup即总表,char_inputs即多少句话*每句话中所有字对应的ID,即外面传进来预处理数据里对应的第1个维度
            #self.char_lookup是2维的，假定为a*b，char_inputs也是2维的，假定为c*d，那么查询后的维度即为c*d*b的维度
            #由于定义的每个字的维度是100维的，所以这个查询之后即c*d*100
            #这个100，有一篇谷歌的论文上说，一般而言，embedding的维度是总字数的开四次方根差不多会比较合理，故而10^8(1亿)个字，差不多就100维
            #self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        #因为每个切词后的字向量最多会有四个状态，即SBIE，若每个状态用20维的向量来表示，则字的切词信息总大小即为4*20
                        #这里切词信息维度用10也可以，但是20维相对特征会丰富一点，具体情况看哪个准确度高用哪个
                        #因为这四个状态也只是用随机数生成的
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)#随机初始化一个4*20的向量总表,后面取的切词信息即从这里面去取
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
                    #如果有切词信息的话，再用每个字对应的切词信息去获取数字信息，这里设定切词信息就20维
                    #这里的seg_lookup即总的数据表，seg_inputs即多少句话*每句话中所有字对应的切词信息，即外面传进来预处理数据里对应的第2个维度
                    #预处理完的数据共四维，第0维即本句话中所有字，第1维即每个字对应ID，第2维即切词信息，第3维即标签ID。
                    #这里总表是4*20,然后seg_inputs维度是c*d,则最后维度为c*d*20
            embed = tf.concat(embedding, axis=-1) #即按最后一个维度拼接
            #一个是c*d*100,另一个是c*d*20,因为每一批次，每句话中对应所有字的ID数和切词信息数大小一致，故而这两个cd是相同的cd。
            #故而按最后一个维度拼接后即c*d*120，这里c即batch_size,d即每句话的长度。因为每个batch_size里其每句话长度是一致的，即做过padding
            #因此这里对字信息提取，主要即使用了“字ID”+“切词信息”，并没有加标签信息，然后使用过程即将“字ID”和“切词信息”对应的数字去各自初始化的总表中去取，分别对应预处理完数据的第1维和第2维。
        self.embed_test=embed
        self.embedding_test=embedding
        return embed

    
    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, 
                    name=None):
    #模型的输入是[batch_size,num_steps,num_tags],即对应着NWC三个维度的数据
    #故而先expand上H维度，然后进入网络
    #网络流程为：conv -> "atrons_conv+bias_add+relu"*4,四次循环中每次结果作为下一次输出，并append到finalOutFromLayers里
    #四次循环结束后，对finalOutFromLayers进行第三维的concat，finalOutFromLayers：[batch_size,H=1,num_steps,num_tags],后两个为字数和标签数
    #concat完之后对finalOutFromLayers进行dropout模型压缩，然后squeeze消去expand上的那个维度
    #再对当前结果做全连接层dense操作，即Y=XW+b
    #最后将结果从 [batch_size*每句话字数,结果类别数] reshape 为 [batch_size,每句话字数,结果类别数]
    
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        #这里的输入即model_inputs是三维的，即batch_size*每批次字的总数*120,120即字信息100维+切词信息20维
        #正常卷积都是四个维度的，即NHWC四个维度，故而这里需要插入一个维度的数
        #当然，这里也可以用conv1d来进行卷积，即没有H高度这个维度了。conv1d的本质也是自己插了一个维度。
        #tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。
        model_inputs = tf.expand_dims(model_inputs, 1) #插到了第一维上，也就是H所在的维度
        #此时model_inputs维度即变成了 batch_size * 1 * 每批次的字数 * 120
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #卷积核大小[H,W,C,N] C即输入通道数，N即卷积核个数，也是输出通道数
            #shape=[1*3*120*100]
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)
            self.layerInput_test=layerInput
            finalOutFromLayers = []
            
            totalWidthForLastDim = 0
            for j in range(self.repeat_times): #有点像分叉口一样，分四个分支，类似于GoogleNet
                #但是这里每一次分叉卷积完了以后都会作为下一次的输入，故而又有点像VGG的感觉
                #即每次分叉的输入都不一样的
                for i in range(len(self.layers)):
                #相当于四个分支，每个分支进行三次卷积，第一个分支的三次卷积分别初始化其自己的w和b，命名空间为atrous-conv-layer-%d
                #后面三个分支每次的卷积核第一个分支一样
                #也就是有四个分支，每个分支都有三次卷积，对应三个命名空间和相应的w、b
                #atrous-conv-layer-0，atrous-conv-layer-1，atrous-conv-layer-2下分别都有各自的filterW，filterB
                    #1,1,2
                    dilation = self.layers[i]['dilation'] #即膨胀系数，分别为1,1,2、
                    #第0次卷积时，且是训练状态时，reuse是false，因为是第一次创建并使用这个命名空间，所以不需要reuse
                    #如果不是第一次卷积，后面要用时，因为变量名字一样，所以一旦get_variable时需要使用reuse。
                    #当然如果是dropout为1，即测试时，也要reuse，因为训练时已经产生了这个命名空间
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                    #reuse=False时会创建新的变量，若变量已存在，则报错。 reuse=True时，则会重用已有的变量。
                    #大家比较常用也比较笨的一种方法是，在重复使用（即 非第一次使用）时，
                    #设置 reuse=True 来 再次调用 该共享变量作用域（variable_scope）。    
                    #更简洁的方法是直接使用  with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                    
                        #w 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        #下面只是为了记录一下卷积核，观察一下不同分支下，相同次数的卷积对应的W和B是不是一样的。
                        #事实上是一样的，即后面三个分支下，每个分支的三次卷积中每一次的W，b都和第一个分支下三次卷积中每一次的W和b是一样的。
                        #但是三次卷积，每次卷积各自的w和b不一样
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        #初始化偏置，偏置个数和卷积核个数要保持一致
                        b = tf.get_variable("filterB", shape=[self.num_filter])
#tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
    #除去name参数用以指定该操作的name，与方法有关的一共四个参数：                  
    #value： 
    #指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数] 
    #filters： 
    #相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
    #rate： 
    #要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，
    #这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入
    #图像上卷积时的采样间隔，你可以理解为卷积核当中穿插了（rate-1）数量的“0”，   插孔操作
    #把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
    #具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，
    #此时这个函数就变成了普通卷积。  
    #padding： 
    #string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。
    #ok，完了，到这就没有参数了，或许有的小伙伴会问那“stride”参数呢。其实这个函数已经默认了stride=1，也就是滑动步长无法改变，固定为1。
    #结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor，这个结果怎么得出来的？先不急，我们通过一段程序形象的演示一下空洞卷积。                        
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv 
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)   #如果是最后一个，则把前面的卷积结果添加进来
                            totalWidthForLastDim += self.num_filter  #输出通道数相加
                        layerInput = conv
            #这里也可以自己修改成resnet那种残差结构连接，因为每次repeat都有三次卷积，而残差结构都是跨两层结构进行相加
            #因此可以将第1次的卷积结果残差连接到第3次卷积之后，relu之前
            #即在 self.conv_test=conv 这行之前先记录本次卷积结果，即resident = layerInput if i == 0
            #然后在conv = tf.nn.relu(conv) 之前进行跨两层的残差连接，即 conv = (resident + conv) if i == 2
            #可以试试这个残差连接后跑的效果
                
            finalOut = tf.concat(axis=3, values=finalOutFromLayers) #concat后第3维的大小即为totalWidthForLastDim
            keepProb = 1.0 if reuse else 0.5  #即只在第0次卷积时会有dropout现象，其它的时候包括后面卷积以及测试时都不会dropout
            finalOut = tf.nn.dropout(finalOut, keepProb)
            #Removes dimensions of size 1 from the shape of a tensor. 
                #从tensor中删除所有大小是1的维度
            
                #Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don’t want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims. 
            
                #给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
            finalOut = tf.squeeze(finalOut, [1]) #把自己添加进去的第1维重新给去掉
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim]) 
            #合并前两个维度，即将[batch_size,每句中字数,每字的特征数] 修改成[batch_size*每句中字数，每字的特征数]
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags]) 
    
    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags]) #即返回batch_size * 每句话字数 * 结果类别数

    def loss_layer(self, project_logits, lengths, name=None):
        #这里之所以不用softMax是因为softMax只是考虑本样本在整体中占的比例，并不会去考虑上下文关系。
        #而条件随机场还会关系状态转移情况，即会关系上下文关系来计算损失，但这个值也是通过随机初始化得到的。
        #这里既给定该batch_size的结果数据，也给出句子长度length，所以最后求tf.reduce_mean时，即能得到本批次所有的误差。
        #因而也就能进行优化了，也就可以是每批次都不等长也可以了
        # project_logits：[batch_size,每句中字数,label_nums],这里的每句中字数即num_steps,label_nums即num_tags
        # lengths本批数据中所有字的长度，即num_steps
        #条件随机场主要需要两个矩阵，一个特征矩阵，另一个是状态转移矩阵
        #不仅要看上一个字的特征(字信息+切词信息)是什么，还需要看上一个字对应的标签是什么
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            # 让最后一个维度num_tags多了1维  start_logits：[batch_size,1,num_tags+1]
            #这里让最后一个维度和zeros进行拼接是为了说明是添加的，只是维度增加，而不增加数字信息
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            #这里以网络输出结果project_logits作为基础，让最后一个维度num_tags变大了1维  
            #得到logits: [batch_size,num_steps,num_tags+1]
            logits = tf.concat([start_logits, logits], axis=1)
            #此时主要是中间维度num_step变大1维   logits：[batch_size,num_step+1,num_tags+1]
            #此时的logits即为所需要的特征矩阵
            #也就是以网络输出结果project_logits作为基础，先标签数据num_tags多一维度，然后再句子字数num_step多一维度
            
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
            #原本targets为[batch_size,num_steps],即对应本批次所有字的标签
            #此时大小为[batch_size,num_steps+1],targets本身是本批次每个字对应的标签值
            #故而这里的targets为标签矩阵
            
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            #这里的trans即是条件随机场所需要的两矩阵之一：特征转移矩阵，通过随机初始化得到值
            #crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            #inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            #一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入. 
            #tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签. 
            #sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度. 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵    
            #log_likelihood: 标量,log-likelihood 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            
            #故而条件随机场总共需要四个参数：
            #特征矩阵logits：[batch_size,num_step+1,num_tags+1]
            #标签矩阵targets: [batch_size,num_steps+1]
            #状态转移矩阵trans: [num_tags+1,num_tags+1]
            #本批次句子长度: length+1
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
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
            #这里只是把所有可能需要的值都记录下来，后面要debug时可以直接查看
            #一次把所有需要run的都给run了，只拿回自己最想要的，故而这里其它的可以删除掉不需要也可以
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
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            #用维特比算法解码

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                #gold = iob_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                #pred = iob_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])                
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

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
