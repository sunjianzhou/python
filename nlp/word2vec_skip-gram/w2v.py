#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
filename = 'text8.zip'

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        #当然这里现在是词，也可以加一句改成字
        #data = [ word for words in data for word in words] #两层for，需要满足先大后小
    return data

#所以这里的filename是需要做成字/词向量的源数据文件
#要做命名实体识别的话，则将所有的内容按空格隔开，成一个字一个字，即可用这个代码生成word2Vec总表
words = read_data(filename)#先获取到切好词的文件中所有的词语，放到一个列表中，这里有内存溢出的可能，当文件上百G的时候就不适应了


# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = len(words)
vocabulary_size = len(set(words))
print('Data size', vocabulary_size)
def build_dataset(words):
    count = [['UNK', -1]]
    #collections.Counter(words).most_common
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1000))
    #统计各词的频数，这里只是简单粗暴地把频数最小的1000个去掉了
    #这里可以对低频词和高频词做一个处理，即首先除上总词数得到频率，然后设置低频域值，低于阈值直接舍弃
    #这里也可以简单按频数进行处理一下，比如 count = [each for each in count if each[1]>1],如果这样的话得把UNK那行放在这个后面
    #再处理高频词，设置t=0.9，target = t/词频 + sqrt(t/词频), ,随机获得一个[0,1)的数，若大于target，则舍弃该词，否则保留。
    dictionary = dict()
    for word, _ in count: 
        dictionary[word] = len(dictionary)
    #上面这个就等价于
    #word,idx in enumerate(count)：
    #    dictionary[word] = idx + 1
    data = list()
    unk_count = 0
    data=[dictionary[word]  if  word in dictionary else 0 for word in words] #将所有词典中每个词对应的ID存起来
    #for word in words:
        #if word in dictionary:
            #index = dictionary[word]
        #else:
            #index = 0  # dictionary['UNK']
            #unk_count += 1
        #data.append(index)
    count[0][1] = unk_count  #简直多此一举，直接在最初赋值为0即可
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #逆置key,value
    return data, count, dictionary, reverse_dictionary #故而这里data即为id，count即词频列表，dictionary即字-id

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory. #看起来这个地方确实能释放许多资源
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window): 
    global data_index #全局变量，每次会改变，从而后面每次获得batch_size的内容时都会更新
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)#会先随机得到一堆数，取决于当前在内存中开辟位置上的01数据,因为是int,32位，故而最小为0，最大是2的31-1
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] 即前后各两个字另加中间目标词。
    buffer = collections.deque(maxlen=span)
    for _ in range(span):  #将data中最开始的5个词语对应的ID添加到双端队列里，其实即相当于初始化队列元素
        buffer.append(data[data_index])  
        data_index = (data_index + 1)
    for i in range(batch_size // num_skips):#i取值0,1 每个batch为8个词，这里因为每个中心词有上下两个词，所以一共循环两次
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        #所以这里直接用shuffle，然后把下面的while循环替代了也可以
        #index_num = list(range(skip_window*1+1))
        #index_num.pop(skip_window) #去掉中心词 列表的remove是删除值，pop是删除索引，但这里值和索引是一致的,故而哪个都可以
        #random.shuffle(index_num)即将对应的序号打乱了，然后用队列去取即可
        for j in range(num_skips): #j取值0,1,2,3
            while target in targets_to_avoid: #targets_to_avoid存的是已经拿到过的数，拿到过了，后面就不能再拿了
                target = random.randint(0, span - 1) # [0,4]随机抽取一个数，还不能是中间的那个目标数
            targets_to_avoid.append(target) 
            batch[i * num_skips + j] = buffer[skip_window] #batch放的都是中心词的ID
            labels[i * num_skips + j, 0] = buffer[target] #labels即放的中心词对应的前后两个上下文ID，并且乱序，不包括中心词
            #这里的labels里每组存四个，分别是中心词的前两个词和后两个词，不过是打乱了顺序，并且保证都有。
            #那么其实可以直接取出前后各两个凑到一个temp的列表里，然后将列表进行shuffle即可
        buffer.append(data[data_index]) 
        data_index = (data_index + 1) % len(data)
    return batch, labels

#下面两行只是为了展示一下效果
batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2) #skip_window表示上下两个字
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    #这里打印出来是这个效果
    #483 三 -> 4491 西元前
    #483 三 -> 14880 欧几里得
    #483 三 -> 70 世纪
    #483 三 -> 1 的
    #70 世纪 -> 1 的
    #70 世纪 -> 483 三
    #70 世纪 -> 1009 希腊
    #70 世纪 -> 4491 西元前    

# Step 4: Build and train a skip-gram model.

# hyperparameters
batch_size = 128
embedding_size = 300 # dimension of the embedding vector
#字/词的维度,即最终字/词的维度，可以自己调整一下
skip_window = 2 # how many words to consider to left and right
num_skips = 4 # how many times to reuse an input to generate a label

# we choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are also the most frequently occurring words
valid_size = 16 # size of random set of words to evaluate similarity on
valid_window = 100 # only pick development samples from the first 'valid_window' words
valid_examples = np.random.choice(valid_window, valid_size, replace=False) #即从[0,100)里随机不重复挑选16个数
#从[0,valid_window]里随机挑选了valid_size个数
#效果等价于np.random.randint(0,valid_window,(valid_size)),但是上面的replace=False则意味着不会有重复的值
num_sampled = 64 # number of negative examples to sample

# create computation graph
graph = tf.Graph()

with graph.as_default():
    # input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # operations and variables
    # look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) #初始化总表
    #这里的embeddings是tf.Variable的类型，并且没有设置trainable=False，所以在网络训练的时候也会发生变化！！！
    #我们最终也是要得到想要的词向量,所以这里只是先初始化一下
    #像命名实体识别中的embedding则是一个列表,本身并没有trainable的属性,网络训练的时候也不会发生改变
    #vocabulary_size即字典中总的字数，embedding_size表示每个单词会有embedding_size那么多维
    #生成一个符合均匀分布的矩阵，取值范围在-1.0到1.0之间的数，矩阵大小为[vocabulary_size, embedding_size]
    #故而这里的词向量是随机初始化的
    embed = tf.nn.embedding_lookup(embeddings, train_inputs) #即对应训练集所有词的ID

    # construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    #对于词向量，如果用正常的神经网络计算会有问题，正常的即包含卷积，激活等，这里采用nce_loss避开了，故而速度上会比较快
    #因为正常的网络输出是会与目标结果类别数一致的，而词向量这里，会以词的总数作为总类别数，故而这样的话就太多了
    #所以就会有部分类别采样的做法
    # compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    ncs_loss_test=tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                   labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size)
    #以中心词作为正样本，中心词对应的上下文作为负样本（还不是严格的随机负采样），从而计算也更便捷快速，类别数少，好计算
    #word2vec里则是随机负采样，同时为了照顾到低频词，还对词频做了四分之三的幂次处理
    #从而实现将一个多分类问题转换为一个二分类问题，这里inputs即对应的各个中心词，即为正样本，labels即对应的两个上下文，即负样本
    #这个tf.nn.nce_loss是核心，这是一个做候选类别采样时计算损失的API
    #候选采样函数，即从巨大的类别库中，按照一定原则，随机采样出类别子集。
    #tf中候选采样损失函数主要有两个：tf.nn.sampled_softmax_loss 和 tf.nn.nce_loss 
    #区别是前者只支持单分类，而后者支持多分类，所以用softmax会特别耗费时间，故而一般会用nce_loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                     labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    #这里即求一下损失均值，num_sampled是负样本噪声单词数量
    # construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #梯度下降最小化损失,来更新参数

    # compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) #计算每个向量的模长
    normalized_embeddings = embeddings / norm #单位化,即将每行向量转成单位向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    #在总表里查了16个数
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    #normalized_embeddings：199247 * 300，转置之后即300 * 199247
    #valid_embeddings：16 * 300    故而两者矩阵相乘即得到 16 * 199247
    # transpose_b=True 表示对b向量先进行转置，transpose_a = True，则表示对a先进行转置
    #正常两个向量a和b求余弦，即向量a与b的点积除以a与b的模长之积，因为这里都先转成单位向量了，故而模长都为1，模长之积也为1
    #而向量a与b的点积，即对应元素相乘后再相加，比如a=(x_1,y_1),b=(x_2,y_2)，a点积b = x_1*x_2 +y_1*y_2
    #故而如果这里是两个向量想要做点积的话，则应该是tf.multiply(a,b)
    #但是这里是用16个词去与所有的词依次做一次点积，每个词对应300个维度
    #得到的结果矩阵similarity每一个元素都是16个元素中的某个与199247个元素中某个进行的点积
    #故而若similarity矩阵中第i行第j列的结果值为1，则代表原先两个矩阵对应的元素是相似的。
    #即valid_embeddings中的第i行对应的元素，和normalized_embeddings转置后的第j列个元素两者最相近，其它的，越相近，则越大，也就是越接近1
    #故而这句话本身的意思即从199247个词中寻找与这16个词最接近的词，越接近，则值越大，对应关系即上一行的分析
    
    # add variable initializer
    init = tf.initialize_all_variables()
#5
num_steps = 1000

with tf.Session(graph=graph) as session:
    # we must initialize all variables before using them
    init.run()
    print('initialized.')
    
    # loop through all training steps and keep track of loss
    average_loss = 0
  
    for step in range(num_steps):
        # generate a minibatch of training data
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        #batch_inputs：(128,)   batch_labels:(128,1)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # we perform a single update step by evaluating the optimizer operation (including it
        # in the list of returned values of session.run())
        _, loss_val,ncs_loss_ = session.run([optimizer, loss,ncs_loss_test], feed_dict=feed_dict)
        #执行优化器，即更新参数
        average_loss += loss_val
        
        # print average loss every 2,000 steps
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # the average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        
        # computing cosine similarity (expensive!)
        #看一下每个词最相似的词
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                # get a single validation sample
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbors
                top_k = 8
                # computing nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary.get(nearest[k],None)
                    #close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        
    final_embeddings = normalized_embeddings.eval() #执行总表单位化
    print(final_embeddings)  #看看最终的词向量
    
    fp=open('vector.txt','w',encoding='utf8')
    for k,v in reverse_dictionary.items():
        t=tuple(final_embeddings[k])
        #t1=[str(i) for i in t]
        s=''
        for i in t:
            i=str(i)
            s+=i+" "
            
        fp.write(v+" "+s+"\n")

    fp.close()
## Step 6: Visualize the embeddings.

#下面是一些可视化的内容
#def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    #assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #plt.figure(figsize=(18, 18))  # in inches
    #for i, label in enumerate(labels):
        #x, y = low_dim_embs[i, :]
        #plt.scatter(x, y)
        #plt.annotate(label,
                 #xy=(x, y),
                 #xytext=(5, 2),
                 #textcoords='offset points',
                 #ha='right',
                 #va='bottom')

    #plt.savefig(filename)

#try:
    #from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt

    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #plot_only = 500
    #low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    #plot_with_labels(low_dim_embs, labels)

#except ImportError:
    #print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

#画图展示
#tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) #TSNE降维函数
#plot_only = 100
#low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#labels = [reverse_dictionary[i] for i in range(plot_only)]
#plot_with_labels(low_dim_embs, labels)