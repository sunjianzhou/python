import pandas as pd
import numpy as np
import matplotlib as mpl
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import re
import time
import jieba
import sys

# mpl.rcParams['font.sans-serif'] = [u'simHei']#ָ改为指定字体“黑体”
# mpl.rcParams['axes.unicode_minus'] = False #使得坐标轴保存负号变更为方块，用来正常显示负号
# plt.title(u'我是中文')
# get_ipython().magic(u'matplotlib tk')

df = pd.read_csv('./data/result_process01', sep = ',', header = None, names= ['from','to', 'date', 'content','label'])
# print(df.head(10))

def 获取邮件收发地址(strl):#发送接收地址提取
    it = re.findall(r"@([A-Za-z0-9]*\.[A-Za-z0-9\.]+)", str(strl))#正则匹配
    result = ''
    if len(it)>0:
        result = it[0]
    else:
        result = 'unknown'
    return result

df['from_address'] = pd.Series(map(lambda str : 获取邮件收发地址(str), df['from']))#map映射并添加
df['to_address'] = pd.Series(map(lambda str: 获取邮件收发地址(str), df['to']))
print("========to address=======================")
print("="*10 + 'to address' + "="*20)#也可以这样写
print(df.to_address.value_counts().head(5))#
print("总邮件接受服务器类别数量为：" + str(df.to_address.unique().shape))#计算服务器的个数
print("="*10 + 'from address' + "= "*20)
print(df.from_address.value_counts().head(5))
print("邮件发送服务器类别数量为：" + str(df.from_address.unique().shape))
from_address_df = df.from_address.value_counts().to_frame()#转为结构化的输出,带出列名
len_less_10_from_address_count = from_address_df[from_address_df.from_address<=10].shape
print("发送邮件数量小于10封的服务器数量为：" + str(len_less_10_from_address_count))
# from_address_df[from_address_df.from_address<=10].to_csv('./data/fromToResult.csv')
# df.from_address.value_counts().to_csv('./data/fromToResultNoneFrame.csv')
#===================================================================================================
np.unique(list(map(lambda t: len(str(t).strip()), df['date'])))#转换为list再去做
print(np.unique(list(map(lambda t: len(str(t).strip()), df['date']))))
# np.unique(list(filter(lambda t: len(str(t).strip())==30, df['date'])))
print((list(filter(lambda t: len(str(t).strip())==7, df['date']))))
print((list(filter(lambda t: len(str(t).strip())==16, df['date']))))
print((list(filter(lambda t: len(str(t).strip())==19, df['date']))))
print((list(filter(lambda t: len(str(t).strip())==21, df['date']))))
print((list(filter(lambda t: len(str(t).strip())==23, df['date']))))
print((list(filter(lambda t: len(str(t).strip())==24, df['date']))))

def 根据日期长度提取日期特征(str1):
    '''
    24~8=3;8~13=0;13~19=1;19~24=2;
    '''
    if not isinstance(str1, str):#如果不是字符串
        str1 = str(str1)
         
    str_len = len(str1)
    week = ""
    hour = ""
    time_quantum = ""      
    if str_len < 10:
        week = "unknown"
        hour = "unknown"
        time_quantum = "unknown"
        pass
    elif str_len == 16:
        rex = r"(\d{2}):\d{2}"#只取冒号前面的
        it = re.findall(rex, str1)
        if len(it) == 1:
            hour = it[0]
        else:
            hour = "unknown"
        week = "Fri"
        time_quantum = "0"
        pass
    elif str_len == 19: #['Sep 23 2005 1:04 AM']
        week = "Sep"
        hour = "01"
        time_quantum = "3"
        pass
    elif str_len == 21: #['August 24 2005 5:00pm'
        week ="Wed"
        hour = "17"
        time_quantum = "1"
        pass
    else:               #'Fri 2 Sep 2005 08:17:50'  Wed 31 Aug 2005 15:06:36 
        rex = r"([A-Za-z]+\d?[A-Za-z]*) .*?(\d{2}):\d{2}:\d{2}.*"# 加问号保险些# 'Fri 23 Sep 2005 09:39:39 +0800 X-Priority: 3 X-Mailer: FoxMail'
        it = re.findall(rex, str1)
        if len(it) == 1 and len(it[0]) ==2:
            week = it[0][0][-3:]#it是list
            hour = it[0][1]
            int_hour = int(hour)
            if int_hour <8:
                time_quantum = "3"
            elif int_hour <13:
                time_quantum = "0"
            elif int_hour <19:
                time_quantum = "1"
            else:
                time_quantum = "2"
            pass
        else:
            week = "unknown"
            hour = "unknown"
            time_quantum = 'unknown'
     
    week = week.lower()
    hour = hour.lower()
    time_quantum = time_quantum.lower()
    return(week, hour, time_quantum)
#数据转换
date_time_extract_result = list(map(lambda st: 根据日期长度提取日期特征(st), df['date']))
df['date_week'] = pd.Series(map(lambda t: t[0], date_time_extract_result))#匿名函数传出的是最后结构里面的值，是子集
df['date_hour'] = pd.Series(map(lambda t: t[1], date_time_extract_result))
df['date_time_quantum'] = pd.Series(map(lambda t: t[2], date_time_extract_result))
print("======星期属性字段的描述==========")
print(df.date_week.value_counts().head(3))
print(df[['date_week', 'label']].groupby(['date_week', 'label'])['label'].count())#先取data_week 和 label，然后按照label去排
print("======小时属性字段的描述==========")
print(df.date_hour.value_counts().head(3))
print(df[['date_hour', 'label']].groupby(['date_hour', 'label'])['label'].count())
print("======时间段属性字段的描述==========")
print(df.date_hour.value_counts().head(3))
print(df[['date_time_quantum', 'label']].groupby(['date_time_quantum', 'label'])['label'].count())                   
df['has_date'] = df.apply(lambda c: 0 if c['date_week'] == 'unknown' else 1, axis=1)#这里的1是按照行
# 开始分词==============================================
print('='*30 + '现在开始分词，请耐心等待5分钟。。。' + '='*20)
df['content'] = df['content'].astype('str')#确保类型，类型转换
df['jieba_cut_content'] = list(map(lambda st: "  ".join(jieba.cut(st)), df['content']))
# df.head(4)
            
#特征工程之四 长度提取  
def 邮件长度统计(lg):
    if lg <= 10:
        return 0
    elif lg <= 100:
        return 1
    elif lg <= 500:
        return 2
    elif lg <= 1000:
        return 3
    elif lg <= 1500:
        return 4
    elif lg <= 2000:
        return 5
    elif lg <= 2500:
        return 6
    elif lg <=  3000:
        return 7
    elif lg <= 4000:
        return 8
    elif lg <= 5000:
        return 9
    elif lg <= 10000:
        return 10
    elif lg <= 20000:
        return 11
    elif lg <= 30000:
        return 12
    elif lg <= 50000:
        return 13
    else:
        return 14
 
df['content_length'] = pd.Series(map(lambda st:len(st), df['content']))
df['content_length_type'] = pd.Series(map(lambda st: 邮件长度统计(st), df['content_length']))
# print(df.head(10))  #如果不count就按照自然顺序排      
df2 = df.groupby(['content_length_type', 'label'])['label'].agg(['count']).reset_index()#agg 计算并且添加count,类似于eval
df3 = df2[df2.label == 1][['content_length_type', 'count']].rename(columns = {'count' : 'c1'})
df4 = df2[df2.label == 0][['content_length_type', 'count']].rename(columns = {'count' : 'c2'})
df5 = pd.merge(df3, df4)#注意pandas中merge与concat的区别
df5['c1_rage'] = df5.apply(lambda r: r['c1'] / (r['c1'] + r['c2']), axis = 1)#1所占百分比
df5['c2_rage'] = df5.apply(lambda r: r['c2'] / (r['c1'] + r['c2']), axis = 1)#0所占百分比
print(df5) 
#画图出来观测为信号添加做准备
plt.plot(df5['content_length_type'], df5['c1_rage'], label = u'垃圾邮件比例')#长度与概率的图像
plt.plot(df5['content_length_type'], df5['c2_rage'], label = u'正常邮件比例')
plt.grid(True)
plt.legend(loc = 0)#加入图例
plt.show()
#添加信号量,数值分析模拟回归方程
def 长度信息量计算(x):
    '''返回值介于0~1之间'''
    if x > 10000:
        return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1) - np.log(abs(x - 10000)) + 1
    else:
        return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1)
a = np.arange(1, 20000)
plt.plot(a, list(map(lambda t: 长度信息量计算(t) ,a)), label = u'信息量')
# plt.plot(df['content_length'], list(map(lambda t: 长度信息量计算(t) ,df['content_length'])), label = u'信息量')
plt.grid(True)
plt.legend(loc = 0)
plt.show()
df['content_length_sema'] = list(map(lambda st: 长度信息量计算(st), df['content_length']))          
# print(df.dtypes) #可以查看每一列的数据类型，也可以查看每一列的名称
   
df.drop(['from', 'to', 'date', 'from_address', 'to_address', \
         'date_week','date_hour', 'date_time_quantum', 'content', \
         'content_length', 'content_length_type'], 1, inplace=True) 
# print(df.info())
# print(df.head(10)) 
  
# df.to_csv('./data/result_process02', encoding='utf-8', index = False)
df.to_csv('./data/result_process02.csv', encoding='utf-8', index = False)

        
          
    