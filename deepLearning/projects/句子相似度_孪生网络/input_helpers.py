#encoding=utf8
import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys


class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None
    def cleanText(self, s):
        s = re.sub(r"[^\x00-\x7F]+"," ", s)
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",s)
        s = re.sub(r'( [0-9,\.]+)',r"\1 ", s)
        s = re.sub(r'\$'," $ ", s)
        s = re.sub('[ ]+',' ', s)
        return s.lower()

    def getVocab(self,vocab_path, max_document_length,filter_h_pad):
        if self.vocab_processor==None:
            print('locading vocab')
            vocab_processor = MyVocabularyProcessor(max_document_length-filter_h_pad,min_frequency=0)
            self.vocab_processor = vocab_processor.restore(vocab_path)
        return self.vocab_processor

    def loadW2V(self,emb_path, type="bin"): #type默认为bin，但现在传进来的是text
    #基于语义相似的前提下，加载word2vec
        print("Loading W2V data...")
        num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        if type=="text":
            # this seems faster than gensim non-binary load
            #该文件第一行表示有多少个字，每个字对应多少维度数字,第二行开始即每个字对应的向量了
            #所以这里应该要先对第一行做特殊的操作,去掉，或者获取内容
            for line in open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()
        #因为这个字典里存的内容比较多，故而单独清除一下

    def getTsvData(self, filepath):
        #这个即表示语料库中本身就具备正样本和负样本，不需要自己额外去创造或者添加
        #这个函数和下一个函数有common的部分，其实可以写成一个函数
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if random() > 0.5:
                x1.append(l[0].lower())
                x2.append(l[1].lower())
            else:
                x1.append(l[1].lower())
                x2.append(l[0].lower())
            #y.append(int(l[2]))  #这里直接转int肯定有问题，除非本身就是数字，只不过是字符型
            #等价于
            y.append(1) if l[2].lower() == 'y' else y.append(0)
        return np.asarray(x1),np.asarray(x2),np.asarray(y)

    def getTsvDataCharBased(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        #先获取正样本，但是将问答对随机地放在两个数组里
        #只是两个数组x1，x2不定是哪个放问句，哪个放答句，故而两个数组里应该都会既存在问句，也存在答句
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            #对于孪生网络，输入的两个句子需要是有序的，这里是打乱顺序随机放的
            #这里的l[0]即问句，l[1]即对应的答句
            if random() > 0.5:
               x1.append(l[0].lower())
               x2.append(l[1].lower())
            else:
               x1.append(l[1].lower())
               x2.append(l[0].lower())
            #if len(l)>2 and l[2].lower() == 'y':
                #y.append(1)#np.array([0,1]))
            #else:
                #y.append(0)
            y.append(1) #这里明确写1则需要保证前面添加进来的都是正样本,用上面注释的更严谨一些
        # generate random negative samples
        combined = np.asarray(x1+x2)
        shuffle_indices = np.random.permutation(np.arange(len(combined)))
        combined_shuff = combined[shuffle_indices] #打乱顺序
        #其实这里一句话就可以了，即combined_shuff = np.random.permutation(combined)
        
        #接下来即添加负样本,这里的负样本即将正样本中问答对中的答案打乱顺序，从而获得负样本
        #这里有一点bug，即哪怕打乱了答案的顺序了，结果仍然有可能会有找到对应的答案，虽然是小概率事件，但是仍有可能
        #故而这里可能并不是严格地正负比例一比一
        for i in range(len(combined)):
            x1.append(combined[i]) # 将所有的问答句都添加进来
            x2.append(combined_shuff[i]) #将打乱顺序的问答句也添加进来
            y.append(0) #np.array([1,0]))
        #这里通过对问答句打乱顺序的方式添加负样本，能够达到正负样本比例1:1的效果
        
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath,"r",encoding="utf-8"):
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[1].lower())
            x2.append(l[2].lower())
            y.append(int(l[0])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)  
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        #这里data即数据，batch_size即一批次大小,num_epochs即总共要执行多少次循环
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1   #计算出每批次的数据量大小
        #这里加1是因为不一定能整除，取整后就会抹掉了小数，即相当于抹掉了最后一批数目不够一批次的样本，故而加1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):  #对每一批次进行yield
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)  
                #因为最后一批可能不足一批次，所以这里作个限制
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w',encoding="utf-8") as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, is_char_based):
        if is_char_based:
            x1_text, x2_text, y=self.getTsvDataCharBased(training_paths)
        else:
            x1_text, x2_text, y=self.getTsvData(training_paths)
        #得到的x1_text和 x2_text首先是等长的，其次不定是问句或者答句，但是两者对应位置是匹配的
        #if和else的区别仅在于语料库中是否有负样本，如果没有，则自己生成负样本
        # Build vocabulary
        print("Building vocabulary")
        #vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0,is_char_based=is_char_based)
        
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        #源码中参数里可以设置最大长度(必须设置)，以及最低频率(默认为0)，超过最大长度的会截断，低于最大长度的会补0。低于限制频率的会舍弃
        #这里面的分词器主要是对应英文的，对于中文，可以自己写一个tokenizer_fn的函数，然后传进去
        #只要tokenizer_fn的结果是一个分词后的列表的生成器，然后列表中所有元素即对应分好的词即可。所谓生成器，即最后不将列表return
        #而类似于这样  for value in iterator:
        #                 yield TOKENIZER_RE.findall(value)
        
        vocab_processor.fit_transform(np.concatenate((x2_text,x1_text),axis=0))
        #这里有点sklearn里的写法的意思，fit是统计词频，transform是返回字对应的ID
        #这里其实主要使用了其fit功能，即统计出词频
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))  #transform对应的是一个生成器,这里有点奇妙
        #x1是对x1_text的ID提取，有一个max_length，超出这个长度，则截断。低于这个长度的，补0，对应unknown的其id也是0
        #x1_text是所有句子，因为限定最长长度为15，故而每个句子能得到对应15个字母(英文文档)的ID
        #x1_text是一个39594长度的列表，x1是一个39594*15的数字矩阵 
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        #这两步只用了transform功能，即获取到对应的ID
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y))) #permutation和shuffle一个作用，但它不影响原数据
        #这里只为了一个随机序列
        x1_shuffled = x1[shuffle_indices] #x1是问答句对应的ID
        x2_shuffled = x2[shuffle_indices] #x2也是问答句对应的ID，与x1一一对应,只是有些是对的，有些是错的，一比一
        y_shuffled = y[shuffle_indices] #y是标签，表明x1,x2的对应关系是否正确
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        #这里传进来的percent_dev是10，即后面要取整体的十分之一的内容
        #因此dev_idx为-3960，后面取数据即从这个位置取到结尾，也就是取了整体的十分之一
        #删除x1,x2,主要是为了省内存
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:] #这里为了做交叉验证，训练集和测试集比例1:9
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        #所以可以理解为sum_no_of_batches即为训练集每批次对应的数据长度
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect() #垃圾回收一下，释放一些内存空间
        return train_set,dev_set,vocab_processor,sum_no_of_batches 
        #vocab_processor是辅助类对象,因为里面还存有一些内容
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)

        # Build vocabulary
        #这个如果在测试时还不能这么写，因为如果测试的是5句话，每次一句话，那么这里就相当于每次都会加载一次字典。
        #会很影响效率，因此可以单独先加载好训练的字典，然后去接收等待新来的测试词语
        #这里如果用于聊天机器人，寻找最匹配的问句的话，此处必须要改改，即加载训练的词典需要先单拎出来，转成ID，直接加载
        
        #这里加载的字典是训练时训练好的，即用的训练集的字典，来对测试集进行测试,内容更完整一些
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path) #这里相当于保存了完整的实例,实际上可以做一些精简，不需要读取那么多
        print (len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp))) #这里即进行了ID化操作
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y

