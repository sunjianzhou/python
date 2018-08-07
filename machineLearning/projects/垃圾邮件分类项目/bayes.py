import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import sys
import time
 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# mpl.rcParams['font.sans-serif'] = [u'simHei']
# mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/result_process02.csv', sep =',')
# print(df.head(5))
df.dropna(axis = 0, how ='any', inplace = True) #按行删除Nan 确保数据安全
# print(df.head(5))
# print(df.info())

x_train, x_test, y_train, y_test = train_test_split(df[['has_date','jieba_cut_content',\
                                                        'content_length_sema']],df['label'],\
                                                    test_size = 0.2, random_state = 0)

# print("训练数据集大小：%d" % x_train.shape[0])
# print("测试集数据大小：%d" % x_test.shape[0])
# print(x_train.head(10))
# print(x_test.head(10)) #注意前面索引
#================================================================================================
print('='*30 + '下面开始tf-idf权重计算' + '='*30)
jieba_cut_content = list(x_train['jieba_cut_content'].astype('str'))
transformer = TfidfVectorizer(norm = 'l2', use_idf = True)#逆向文件频率  默认即这两个参数
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)# fit_transform(jieba_cut_content)
# df1 = transformer.fit_transform(jieba_cut_content)

print('='*30 + '下面开始svd分解降维计算' + '='*30)
svd = TruncatedSVD(n_components=20)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)

print('='*30 + '重新构建矩阵开始' + '='*30)
data['has_date'] = list(x_train['has_date'])
data['content_length_sema'] = list(x_train['content_length_sema'])

#以上完成了数据清洁工作-------->tf-idf文本转换和svd降维

print('='*30 + '构建伯努利贝叶斯模型' + '='*30)  
nb = BernoulliNB(alpha = 1.0, binarize = 0.0005)#二值转换阈值
model = nb.fit(data, y_train)

print('='*30 + '构建测试集' + '='*30)    
jieba_cut_content_test = list(x_test['jieba_cut_content'].astype('str'))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test['has_date'] = list(x_test['has_date'])
data_test['content_length_sema'] = list(x_test['content_length_sema'])

print('='*30 + '开始测试集预测' + '='*30)
start = time.time()  
y_predict = model.predict(data_test)
end = time.time()
print('朴素贝叶斯预测共耗时%.2f秒'%(end-start))

print('='*30 + '开始评估预测模型' + '='*30)   
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1mean = f1_score(y_test, y_predict)

print('='*30 + '打印预测结果' + '='*30)   
print('精确率为：%0.5f' % precision)
print('召回率：%0.5f' % recall)
print('F1均值为：%0.5f' % f1mean)






