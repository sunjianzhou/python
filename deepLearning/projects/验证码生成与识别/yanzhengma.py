#encoding=utf8
#本代码是半成品，仿LetNet，识别四位验证码图片
#能训练，不能预测。根本原因是设计的矩阵存在问题，暂留后解。
#因为属于第一个试手内容，也从中能学到一些，故而暂留

#由于自己的初始设定，y不是一个只存在一个1的向量，而是多个数，这多个数代表的是一个二进制数，故而有多个位置上结果需要是1
#故而也就没法是通过max_idx_p = tf.argmax(predict,2)这种只招最大数所在的维度来判断了，因为只会找第一个位置
#但预测的结果会类如[-6.514113 1.6827217 0.2300825 0.11542906 -0.42399892 -0.18963598 0.05467165 0.04269369]，并非是01形式
#这就导致后面没法进一步进行预测，没法将其转成十进制的ASCII码值获取text，故而目前暂时卡住，以后有机会再回来看

#这个对于y的设定即对与text转矩阵的设定，自己认为比较有意思，所以不舍得改掉这块
#即四个验证码（虽然训练测试都是数字、大小写字母、下划线，但实际可以是ACSII表所有元素），因为2的8次为256，可以包含所有ASCII码值
#故而每个验证码给8位，四个一共32位，即可以通过32位0、1数字来唯一确定任意顺序的验证码

#如果将y改成一个位置代表一个字符，故而只要一个1即能确定元素的话，应该可以让网络通顺，因为可以使用max_idx_p = tf.argmax(predict,2)
#对于预测结果[-6.514113 1.6827217 0.2300825 0.11542906 -0.42399892 -0.18963598 0.05467165 0.04269369]也能得到最大值所在位置
#从而判定是该位置为1的对应元素即可。
#其他方式暂时没想到，先留着。

from captcha.image import ImageCaptcha #需要pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image #需要pip install pillow
import random
import tensorflow as tf

number =list(map(str,range(0,10)))
print(number)
alphabet = [ chr(each+ord('a')) for each in range(26) ]
print(alphabet)
ALPHABET = [ chr(each+ord('A')) for each in range(26) ]
print(ALPHABET)

def random_captcha_text(char_set=number+alphabet+ALPHABET,captcha_size=4):
#captcha_size表示验证码位数
    captcha_text=[]
    for i in range(captcha_size):
        c = random.choice(char_set) #随机从列表中选出一项
        captcha_text.append(c)
    return captcha_text #生成一个四个随机得到的元素的列表

#生成验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha() #实例化验证码类
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    #把字符生成图像验证码
    captcha = image.generate(captcha_text)#此时是一个二进制流
    #image.write(captcha_text,captcha_text + 'jpg')
    
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    #得到label和图像
    return captcha_text,captcha_image

#图片展示
text,image = gen_captcha_text_and_image()
f = plt.figure()
ax = f.add_subplot(111)
plt.xlim(0,120)
ax.text(0.1,0.9,text,transform=ax.transAxes)#只是为了把内容写出来
plt.imshow(image)
plt.show()

MAX_CAPTCHA = 4 #验证码共4位
CHAR_LENGTH = 8 
#CHAR_SET_LEN = 10 + 26 + 26 + 1 #先数字，后小写字母，后大写字母，后下划线，另外还有四位通过append来添加，用来记录顺序，用于文本转数字矩阵

def mybin(num):#十进制转成8位的二进制,因为2的8次为256，正好整个ASCII表，足以表示各种字符
    if not str(num).isdigit():
        raise{"input Error"}
    res = []
    while num != 0:
        res.append(num%2)
        num = num // 2
    while len(res) != CHAR_LENGTH:
        res.append(0)
    return res[::-1]
res = mybin(13)
print(res)
def mydec(num_list):#二进制数转十进制,num为整型列表
    print(num_list)
    if len(num_list) != CHAR_LENGTH:
        raise ValueError("the length of binary num should be 8")
    num = "".join(map(str,map(int,num_list)))
    return int(str(num),2)
print(mydec(res))

#文本转换，文本转换为向量,按顺序获取ASCII码值，将ASCII码转二进制数据，不足八位补到八位
def text2vec(text):
    if len(text) != MAX_CAPTCHA:
        raise ValueError("验证码需要%d个字符"%MAX_CAPTCHA)
    #生成一行MAX_CAPTCHA * CHAR_SET_LEN的都是0的矩阵
    vector = np.zeros(MAX_CAPTCHA * CHAR_LENGTH)
    for i,c in enumerate(text):
        binary = mybin(ord(c))
        vector[i*8:i*8+8] = binary
    return vector

#向量转回文本
def vect2text(vec):
    text = []
    for i in range(MAX_CAPTCHA):
        each = chr(mydec(vec[i*8:i*8+8]))
        text.append(each)
    return ''.join(text)   

vec = text2vec("F5Sd")
print(vec)
text = vect2text(vec)
print(text)
vec = text2vec("SFd5")
text = vect2text(vec)
print(text)

text,image = gen_captcha_text_and_image()
print(image.shape)

#转换为灰度图,即将三筒单变成单通道，压缩数据，减少数据量
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img,-1)
        #这里的颜色不会影响到图片的text，故而取快速的
        #上面的转法较快，正规转法如下：
        #r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
        #gray = 0.2989*r + 0.5870*g + 0.1140*b
        return gray
    else:
        return img
    
#生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size,MAX_CAPTCHA*CHAR_LENGTH])

    def wrap_gen_captcha_text_and_image():
        while True: #image尺寸大小必须一样大，因为偶尔不是这个尺寸，pass掉
            text,image = gen_captcha_text_and_image()
            #print(image.shape)
            if image.shape == (60,160,3):
                return text,image

    for i in range(batch_size):
        text,image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)#转换为灰度图，减少计算量

        #rgb(255,255,255)
        batch_x[i,:] = image.flatten()/255 #归一化到零一之间
        #(image.flatten()-128)/128则是归一化到-1到+1之间
        #这两种方式主要看激活函数是sigmoid还是tanh，relu则随意
        batch_y[i,:] = text2vec(text)

    return batch_x,batch_y

def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1): #w_alpha和b_alpha是调参调出来的
    #仿照LeNet的结构
    x = tf.reshape(X,shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1]) #NHWC,做灰度图故而通道数为1，不做则为3
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    #卷积核一般在3到四分之一原图大小，去尝试，通道数一般为2的平方的倍数，即一般为4的倍数
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    #conv1d一般用于一维矩阵处理，比如文本处理会用conv1d
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    out = tf.nn.dropout(pool1,keep_prob)
    
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    out = tf.nn.dropout(pool2,keep_prob)
    
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    out = tf.nn.dropout(pool3,keep_prob)
    
    #Fully connected layer
    #三次池化(padding=SAME) 60*160 -> 30*80 -> 15*40 ->8*20
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64,1024]))
    #后面维度是自己设的，但是最好是前面的十分之一或九分之一，但是需要是2的幂次方
    #8*20*64=10240
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(out,[-1,w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)
    
    w_out = tf.Variable(w_alpha*tf.random_normal([1024,MAX_CAPTCHA*CHAR_LENGTH]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_LENGTH]))
    out = tf.add(tf.matmul(dense,w_out),b_out)
    return out
    
#对于CNN而言，1、定义网络2、定义损失3、定义优化器4、执行训练计算精确度5、保存模型
best_acc = 0 
def train_crack_captcha_cnn():
    global best_acc
    batch_size = 64
    #CNN 训练过程
    output = crack_captcha_cnn()
    #损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))
    #Adam函数 梯度下降优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    #接下来是为了判断正确率
    #转换矩阵形状
    predict = tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_LENGTH])#MAX_CAPTCHA即最大验证码长度，为4
    target = tf.reshape(Y,[-1,MAX_CAPTCHA,CHAR_LENGTH])
    ##这里因为不是只有一个位置有1，故而没法取argmax然后比较是不是相同的位置
    ##取第2个维度即在向量中具体位置进行比较
    ##max_idx_p = tf.argmax(predict,2)
    ##max_idx_l = tf.argmax(tf.reshape(Y,[-1,MAX_CAPTCHA,CHAR_LENGTH]),2)
    ##相等的判断
    ##correct_pred = tf.equal(max_idx_p,max_idx_l)
    ##accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))#精确正确率
    
    result = tf.subtract(predict,target)
    expect = tf.zeros((batch_size,MAX_CAPTCHA,CHAR_LENGTH),dtype=tf.int32)
    correct = tf.equal(tf.cast(result,tf.int32),expect)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))#这个比较方式可能会过于精细导致精确度上不去，因为是所有元素进行整体比较
    #比如5*4*8，总共160个元素，其中140个元素一致，则它的比较结果则是140/160，而不是5个里面对了几个。
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#初始化session的变量
        
        step = 1
        epoch = 1
        while epoch < 500 :
            batch_x,batch_y = get_next_batch(batch_size)
            _,loss_ = sess.run([optimizer,loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            
            #每10step进行一次准确率计算
            if step % 10 ==0:
                batch_x_test,batch_y_test = get_next_batch(batch_size)
                test_acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1})
                #测试时不需要有丢弃，故而全部保持
                print(step,test_acc)
                #如果准确率提升了
                is_best = test_acc > best_acc
                if is_best:
                    best_acc = test_acc
                    print("The current is the best")
                    saver.save(sess,"./model/crack_capcha.model",global_step=step)
                    writer = tf.summary.FileWriter("./tf",tf.get_default_graph())
                    #保存视图
                    writer.close()
                    #break
            
            step += 1
            epoch += 1
                
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./model/crack_capcha.model-450")
        #predict = tf.argmax(tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_LENGTH]),2)
        #text_list = sess.run(predict,feed_dict={X:[captcha_image],keep_prob:1})
        #text = text_list[0].tolist()
        #对输出的向量先计算精确率，后获取结果值
        predict = tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_LENGTH])
        vec = sess.run(predict,feed_dict={X:[captcha_image],keep_prob:1})
        vec = tf.reshape(vec,[-1])        
        return vec
        
if __name__=="__main__":
    train = 0
    if train == 0:
        number =list(map(str,range(0,10)))
        alphabet = [ chr(each+ord('a')) for each in range(26) ]
        ALPHABET = [ chr(each+ord('A')) for each in range(26) ]
        
        #生成验证码值和图片
        text,image = gen_captcha_text_and_image()
        print("验证码图像channel：",image.shape) 
        #图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本字符数",MAX_CAPTCHA)
        CHAR_LENGTH = 8
        
        X = tf.placeholder(tf.float32,[None,IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_LENGTH])
        keep_prob = tf.placeholder(tf.float32)
        
        train_crack_captcha_cnn()
    if train == 1:
        number = list(map(str,range(0,10)))
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_LENGTH = 8
        
        text,image = gen_captcha_text_and_image()
        
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1,0.9,text,transform=ax.transAxes)
        plt.imshow(image)
        
        plt.show()
        
        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten()/255
        
        X = tf.placeholder(tf.float32,[None,IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_LENGTH])
        keep_prob = tf.placeholder(tf.float32)
        #调用模型预测
        predic_text = crack_captcha(image)
        predic_text = vect2text(tf.Session().run(predic_text))
        print("正确：{}预测：{}".format(text,predic_text))
        
