#!/usr/bin/env python3
#encoding=utf8
import os
import re
import sys
import sqlite3
from collections import Counter

from tqdm import tqdm

def file_lines(file_path):
    #读取文件中所有句子，以E打头的直接添加一个空，以E打头的从第2个开始，读进整个句子。去句号，有空格的转成逗号。
    with open(file_path, 'rb') as fp:
        b = fp.read() #以二进制的方式读取能加快速度
    content = b.decode('utf8', 'ignore')
    lines = []
    for line in tqdm(content.split('\n')):
        try:
            #原本的句子都是类似这样的：“ M 就/因/为/没/穿/红/让/人/赏/咱/一/纸/枷/锁/ ” 或者是本行所有内容就单单一个E
            line = line.replace('\n', '').strip() #\n应该没了，所以这里主要即删除前后空格
            if line.startswith('E'):
                lines.append('')
            elif line.startswith('M '):
                chars = line[2:].split('/') #得到后面所有的字
                while len(chars) and chars[len(chars) - 1] == '.': 
                    chars.pop() #如果最后一个字是“。”,则把“。”删掉
                if chars:
                    sentence = ''.join(chars) #将多个字拼接成句子
                    #re.sub用于把sentence中空格 ' ' 替换成 '，' : 
                    sentence = re.sub('\s+', '，', sentence) #将多个空格替换成逗号
                    lines.append(sentence)  #得到所有句子
        except:
            print(line)
            return lines
        
        #lines.append('')
    return lines

def contain_chinese(s):
    #中文字符的匹配
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def valid(a, max_len=0):
    #1、只有中文才合法。2、如果没设最大长度，则直接通过，如果有最大长度，则需要当前句子要比最大长度小才通过
    #所以如果传过来的只是一个空格或者句号，则会因为没有中文字而通不过check
    if  contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:
            return True
    return False

def insert(a, b, cur):
    cur.execute("""
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_if(question, answer, cur, input_len=500, output_len=500):
    if valid(question, input_len) and valid(answer, output_len):
        insert(question, answer, cur)
        return 1
    return 0

def main(file_path):
    lines = file_lines(file_path)
    #得到文本中所有的句子，放在list里，每个元素是一个字符串，代表一句话
    print('一共读取 %d 行数据' % len(lines))

    db = 'db/conversation.db'
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)  #连接上本地的sqlit3数据库
    #sqlite3是一个小型轻量级本地数据库，需要自己安装一下
    #即下载两个包，解压后全放到一个文件夹下，然后将该文件夹所在路径添加到path底下即可。
    cur = conn.cursor() #获取一个类似指针的东西
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation
        (ask text, answer text);
        """)
    conn.commit()
    #创建一个表，关联提问和回答

    words = Counter()
    a = ''
    b = ''
    inserted = 0
    
    for index, line in tqdm(enumerate(lines), total=len(lines)):
        words.update(Counter(line))
        #即将每句话都作为一次问答，如ABC三句话，则ask：answer总共有三个：‘’：A，A:B,B:C
        a = b
        b = line
        ask = a
        answer = b
        inserted += insert_if(ask, answer, cur)
        # 批量提交
        if inserted != 0 and inserted % 1000 == 0:
            conn.commit()
    conn.commit()

if __name__ == '__main__':
    file_path = 'dgk_shooter_min.conv'
    #该文件里都是一些语句对话，但是没有太格式对齐
    if len(sys.argv) == 2: #默认只有一个参数，即当前文件，如果传进来一个参数，那么sys.argv就会有两个参数
        file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print('文件 {} 不存在'.format(file_path))
    else:
        main(file_path)
