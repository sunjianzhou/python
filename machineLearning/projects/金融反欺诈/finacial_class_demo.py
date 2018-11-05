import pandas as pd
import numpy as np
import sys

df = pd.read_csv("./data/LoanStats3a.csv",skiprows=1)
#print(df.head())
# print(df.info())

df2 = df
df2.drop('id',axis=1,inplace=True)
df2.drop('member_id',1,inplace=True)
#print(df2.head())

#清洗数据，去除特征中的某些特殊字符
#term列需要删除非数字部分的month和空格
#int_rate需要删除%
df2.term.replace(to_replace='[^\d]+',value='',inplace=True,regex=True)
df2.int_rate.replace(['%'],'',inplace=True,regex=True)
# print(df2.head().int_rate)

#对于grade和sub_grade,删除sub_grade
#对于em_title，职业title，可以删掉
df2.drop('sub_grade',axis=1,inplace=True)
df2.drop('emp_title',1,inplace=True)

#对于emp_length，先统计一下,发现有存在"n/a"、10+ years, < 1years,然后1到9 years
#故而对于n/a替换成nan，对于<1 years替换成1,对于10+ years替换成10，非数字都删掉
#print(df.emp_length.value_counts())
df2.emp_length.replace('n/a',value=np.nan,inplace=True)
df2.emp_length.replace('[^\d]+','',inplace=True,regex=True)
# print(df.emp_length.value_counts())

#print(df2.shape)
#删除所有值都为空的行和列
df2.dropna(how='all',axis=0,inplace=True)
df2.dropna(how='all',axis=1,inplace=True)

#若要用SelectKBest来查看X与Y的关系从而进行数据特征的筛选，比如使用卡方检验，则需要所有的X是各种统计频数，至少是数值型。
#而目前数据中包含有数值型和浮点型的，故而需要先对各种数字类型进行处理。但可以先尝试获取所有数值型数据进行卡方检验试试,也只能是说删除卡方值特别小的。
#结果发现，卡方值小的，往往unique后的值也少。这里按照特征unique后端额数量先考虑，故而此段不要。
#print(df2.info())
# pre_x = df2[df2.select_dtypes(['float']).columns]
# pre_x = pre_x.fillna(pre_x.mean()+0.001)
# pre_y = pd.Categorical(df2['loan_status']).codes
# from sklearn.feature_selection import SelectKBest,chi2
# sk = SelectKBest(chi2)
# sk.fit(pre_x,pre_y)
# importance = sorted(sk.scores_[np.isnan(sk.scores_) == False],reverse=True) #由大到小排列
# print(importance)
# less_importance = []
# for num,score in enumerate(sk.scores_):
#     if np.isnan(score) or score < 200:
#         print('column name:{}'.format(pre_x.columns[num]),end=' ')
#         print('chi2 score:{}'.format(score),end=' ')
#         print('unique number:{}'.format(len(pre_x[pre_x.columns[num]].unique())))
#         less_importance.append(pre_x.columns[num])
# df2.drop(less_importance,axis=1,inplace=True)

#print(df2.shape) #(42535, 61)
#print(df2.info()) #在删除了空行空列之后，则可以列出各个字段得数据信息了
#查看到df2的info信息之后，观察每个特征变量的数据条数：
#1、首先粗看，对于最后几个字段，如debt_settlement_flag_date等都只有一百多条数据，故而直接先删。
# debt_settlement_flag_date     98 non-null object
# settlement_status             155 non-null object
# settlement_date               155 non-null object
# settlement_amount             155 non-null float64
# settlement_percentage         155 non-null float64
# settlement_term               155 non-null float64
df2.drop(['debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term'],axis=1,inplace=True)
#print(df2.shape) #(42535, 55)

#删除float类型中重复值较多的特征
#先查看每个float类型数据中都有多少个不同得值，对于那种只有很少的，需要看看，对于那些最多只有一百多个的，也先删掉
# col delinq_2yrs has 13
# col inq_last_6mths has 29
# col mths_since_last_delinq has 96
# col mths_since_last_record has 114
# col open_acc has 45
# col pub_rec has 7
# col total_acc has 84
# col out_prncp has 1
# col out_prncp_inv has 1
# col collections_12_mths_ex_med has 2
# col policy_code has 1
# col acc_now_delinq has 3
# col chargeoff_within_12_mths has 2
# col delinq_amnt has 4
# col pub_rec_bankruptcies has 4
# col tax_liens has 3
# for col in df2.select_dtypes(include=['float']).columns:
#     print('col {} has {}'.format(col,len(df2[col].unique())))
df.drop(['delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc',\
         'pub_rec','total_acc','out_prncp','out_prncp_inv','collections_12_mths_ex_med','policy_code',\
         'acc_now_delinq','chargeoff_within_12_mths','delinq_amnt','pub_rec_bankruptcies','tax_liens'],axis=1,inplace=True)

#删除object类型中重复值较多的特征
# for col in df2.select_dtypes(include=['object']).columns:
#     print("col {} has {}".format(col,len(df2[col].unique())))
#逐个分析，对于数量很少的进行过滤，这里的loan_status不能删，因为它是Y值
#例如col term has 2，这里是只有分两期，一期是36，一期是60，从业务角度上看，对结果应该有影响
#但从机器学习角度考虑，这里可以看看删掉有没有影响，也就是最后准确度的问题
#例如等级grade，虽然只有7个不同的值，从业务的角度上看，对结果是有用的，应该留，但从纯机器学习的角度上看，可以删了试试
#其它的，数量少的，则都删除掉，因为数据太少，没法用
# col term has 2
# col int_rate has 394   #这个讲课中没删
# col grade has 7
# col emp_length has 11
# col home_ownership has 5
# col verification_status has 3
# col issue_d has 55
# # 这个是Y，不能删 col loan_status has 4
# col pymnt_plan has 1
# col purpose has 14
# col zip_code has 837 邮政编码 虽然八百多，但是具体数据中还有掩藏了部分数据，故而也不要
# col addr_state has 50
# col earliest_cr_line has 531 #最早还款日期，主要是贷款机构的约束条件，也没啥用
# col initial_list_status has 1
# col last_pymnt_d has 113
# col next_pymnt_d has 99
# col last_credit_pull_d has 125
# col application_type has 1
# col hardship_flag has 1
# col disbursement_method has 1
# col debt_settlement_flag has 2
df2.drop(['term','int_rate','grade','emp_length','home_ownership','verification_status','issue_d','pymnt_plan',
          'purpose','zip_code','addr_state','earliest_cr_line','initial_list_status','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d','application_type','hardship_flag','disbursement_method','debt_settlement_flag'],axis=1,inplace=True)

#最后一次筛选
#print(df2.info()) #只剩19列了
#总共19列，逐个去撸，desc比其它少很多，一看是描述性文档，删除掉
#title，一看也没啥用，就不要了
df2.drop(['desc','title'],axis=1,inplace=True)

#目前从初步上X处理了一遍,现在先处理Y值
# print(df2.info())
# y = df2.loan_status
# print(y.value_counts())
# Fully Paid                                             34116
# Charged Off                                             5670
# Does not meet the credit policy. Status:Fully Paid      1988
# Does not meet the credit policy. Status:Charged Off      761
#后两种是不能确定的状态，因为数量也比较少，故而可以把后两种删除掉
df.loan_status.replace('Fully Paid',int(1),inplace=True)
df.loan_status.replace('Charged Off',int(0),inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid',np.nan,inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off',np.nan,inplace=True)
print(df2.loan_status.value_counts())
#删除掉Y值中存在有nan的所在行
df2.dropna(subset=['loan_status'],how='any',axis=0,inplace=True)

#再查看表数据信息，发现revol_util的数据量少于其它列的数据
#故而填充缺失数据
# print(df2.info())
df2.fillna(0.0,inplace=True)
# print(df2.info())

#检测清洁后样本特征的相关性，删除掉线性相关的特征
# cor = df2.corr()
# cor.iloc[:,:] = np.tril(cor,k=-1)
# cor = cor.stack()
# del_list = cor[(cor>0.9)|(cor<-0.9)]
# del_list = cor[(cor>0.9)|(cor<-0.9)].index.tolist()
# print(del_list)#只看第二列，大于0.95以上的选择去掉
# funded_amnt      loan_amnt          0.981544      #去掉loan_amnt
# funded_amnt_inv  loan_amnt          0.940157
#                  funded_amnt        0.958564      #去掉funded_amnt
# installment      loan_amnt          0.930209
#                  funded_amnt        0.956108      #已去funded_amnt
#                  funded_amnt_inv    0.905098
# total_pymnt      funded_amnt        0.901775
# total_pymnt_inv  funded_amnt_inv    0.911798
#                  total_pymnt        0.971633      #去掉total_pymnt
# total_rec_prncp  total_pymnt        0.972251      #已去total_pymnt，故而共需去掉三个
#                  total_pymnt_inv    0.941208
#删除特征变量中线性相关系数大于0.95的特征
df2.drop(['loan_amnt','funded_amnt','total_pymnt'],axis=1,inplace = True)

#再次打印信息，查看是否有非float类型数据，将其做哑变量处理。
# print(df2.info())
# print(df2.head(10))
df2 = pd.get_dummies(df2)
df2.to_csv('./data/feature711.csv')
print(df2.info())




