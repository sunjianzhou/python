import tensorflow as tf

flags = tf.app.flags
#定义变量主要即三项：变量名，变量值，变量说明
flags.DEFINE_boolean("var1", True, "this is for boolean")
flags.DEFINE_float("var2", 6.6, "this is for float")
flags.DEFINE_string("var3", "shusheng", "this is for string")
flags.DEFINE_integer("var4", 666, "this is for integer")

Flags = tf.app.flags.FLAGS

def main(_):  ##必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   main的参数名随意定,无要求
    print(Flags.var1)  #True
    print(Flags.var2)  #6.6
    print(Flags.var3)  #shusheng
    print(Flags.var4)  #666

if __name__ == "__main__":
    tf.app.run(main) #目的即在执行main()函数之前，先对flags进行解析，即获取到flags里定义的变量