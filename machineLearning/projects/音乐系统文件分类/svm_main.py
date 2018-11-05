#coding:utf-8
#from music_category import svm
#from music_category import feature
import feature
import pandas as pd
import numpy as np
from sklearn.externals import joblib#保存模型模块
import sys
import time

数值化标签路径 = './data/music_index_label.csv'
# svm.cross_validation(data_percentage=0.99)
def 加载模型(model_f = None):#load_model
    if not model_f:
        model_f = 模型保存路径
    clf = joblib.load(model_f)
    return clf
    
def 歌曲标签数值化():#fetch_index_label
    """
     从文件中读取index和label之间的映射关系，并返回dict
    """
    data = pd.read_csv(数值化标签路径, header=None, encoding='utf-8')
    name_label_list = np.array(data).tolist()
    index_label_dict = dict(map(lambda t: (t[1], t[0]), name_label_list))
    return index_label_dict
index_lable_dict = 歌曲标签数值化()


def 预测(clf, X):
    label_index = clf.predict([X])
    label = index_lable_dict[label_index[0]]
    return label 


if __name__ == '__main__':
    数值化标签路径 = './data/music_index_label.csv'
    模型保存路径 = './data/music_model.pkl'
    clf = 加载模型()
    #svm.多次训练并保存模型(train_percentage = 0.9, fold = 1000)
    path = './data/test/50 Cent - Ready For War.mp3'#兴奋
#     path = './data/test/A＊Teens - Floorfiller.mp3'#流行
#     path = './data/test/Beyond - 大地.mp3'#怀旧
    music_feature = feature.获取单首歌曲特征(path)
    label = 预测(clf, music_feature)
    print('预测标签为：%s'% label)
