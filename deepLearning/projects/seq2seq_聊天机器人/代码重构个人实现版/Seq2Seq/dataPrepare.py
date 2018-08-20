import os
root_path = os.getcwd()+os.sep
import shutil
import tensorflow as tf
from tqdm import tqdm  #注意这里不能直接import tqdm，因为这是一个大库，要使用的是其底下重名的tqdm
import re

flags = tf.app.flags
#flags.DEFINE_string("filePath", root_path+"data"+os.sep+"dgk_shooter_min.conv" , "Path for data")
flags.DEFINE_string("filePath", root_path+"data"+os.sep+"test_1wLines" , "Path for data")
flags.DEFINE_string("directory_Path", root_path+"bucket_datas" , "directory for bucket_datas")
FLAGS = tf.app.flags.FLAGS

def prepareFiles(directory_Path):
    if os.path.isdir(directory_Path):
        shutil.rmtree(directory_Path)
    os.mkdir(directory_Path)

def get_question_and_answer(filePath):
    with open(filePath,"r",encoding="utf-8",errors="ignore") as file:
        lines = [ line.rstrip().lower() for line in file.readlines() ]
    ask,answer = "",""
    for line in tqdm(lines,total=len(lines)):
        if line.startswith("e") or len(line) <= 2 or not contain_chinese(line):
            continue
        temp = line[2:].replace("/","")
        ask = answer
        answer = temp[:-1] if temp[-1]=="." or temp[-1]=="。" else temp
        #print("ask:{}  answer:{}".format(ask,answer))
        dump_ask_answer(ask,answer)

def dump_ask_answer(ask,answer,buckets = [(5, 15), (10, 20), (15, 25), (20, 30)]):
    if len(ask.strip())==0 or len(answer.strip())==0:
        return None
    ask = re.sub("\s+",",",ask.strip())
    answer = re.sub("\s+", ",", answer.strip())
    for i in range(len(buckets)):
        encoder_size,decoder_size = buckets[i]
        if len(ask) <= encoder_size and len(answer) <= decoder_size:
            fileName = FLAGS.directory_Path+os.sep+"bucket_{}_{}".format(encoder_size,decoder_size)
            if os.path.isfile(fileName):
                with open(fileName,"a+",encoding="utf-8") as file: #有文件，直接追加,以追加的方式添加，减少内存开销
                    file.write("\n"+ask+"   "+answer.strip())
            else:
                with open(fileName, "w+", encoding="utf-8") as file:  # 没有文件，创建
                    file.write(ask + "   " + answer.strip())
            return None

def contain_chinese(line):
    pattern = re.compile('[\u4e00-\u9fa5]+')
    if pattern.findall(line):
        return True
    return False

def main(_):
    assert os.path.isfile(FLAGS.filePath)
    prepareFiles(FLAGS.directory_Path)
    get_question_and_answer(FLAGS.filePath)

if __name__=="__main__":
    tf.app.run(main)


