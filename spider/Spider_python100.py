import re,requests,time
from bs4 import BeautifulSoup
'''
入口地址:http://www.runoob.com/python/python-100-examples.html
需求:
    获取100个python程序实例的题目,程序分析,源代码
需求拆分:
    1.获取到100个子页面的url,存到一个列表里
    2.遍历子页面,依次获取其题目,程序分析及源代码部分的内容,顺便写到文件中
    3.调试过程找出特殊子页面,定制特定规则
'''

# #1.获取html内容
# url = "http://www.runoob.com/python/python-100-examples.html"
# headers = {
#     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
# }
# content = requests.get(url,headers = headers).content.decode("utf-8") # 加个header伪装一下
# #print(content)
#
# #2.解析html,拿到所有想要的链接,存到列表里
# soup = BeautifulSoup(content,"html.parser")
# all_href = []
# all_a = soup.find(id="content").ul.find_all('a')
# #print(all_a)
#
# #从标签中过滤出想要的href地址
# # 方式一
# # for each_a in all_a:
# #     all_href.append(each_a.attrs['href'])
# #print(all_href)
#
# #方式二
# pattern = re.compile("href=\".*?\"")#因为匹配以"结尾,所以需要用非贪婪式
# for each in all_a:
#     href = pattern.findall(str(each))#获取到所有匹配内容的列表
#     all_href.append(href[0][6:-1])#因为只要链接地址,所以稍加切片一下,去除无用的头尾
# #print(all_href)
#
# #3.逐个页面去爬取想要的内容
# pre_url = 'http://www.runoob.com'
# num = 1
# for each_href in all_href:
#     print("正在爬取第%s个页面"%num)
#     sub_html = requests.get(pre_url+each_href).content.decode("utf-8") #由于给的都是相对路径地址,所以需要自己添加上地址头
#     soup2 = BeautifulSoup(sub_html,"html.parser")
#     #print(soup2)
#
#     content = {}
#     content['标题'] = soup2.find(id='content').h1.text
#     content['题目'] = soup2.find(id='content').find_all('p')[1].text
#     content['程序分析'] = soup2.find(id='content').find_all('p')[2].text
#     try:
#         #content['程序源代码'] = soup2.find(id='content').find(class_='hl-main').text #由于class是关键字,故添加上下划线
#         content['程序源代码'] = soup2.find(id='content').find(name='div',attrs={'class':'hl-main'}).text
#     except: #调试过程中捕捉到第42个页面对于当前规则不适应,那么特殊页面,特殊规则
#         content['程序源代码'] = soup2.find(id='content').pre.text
#     finally:
#         pass #加个finally只是为了完整那么一点点
#
#     with open("python_100_examples.txt","a+",encoding='utf-8') as file: #因为内容爬的少,追加到一个文本文件里即可
#         file.write(content['标题']+'\n')
#         file.write(content['题目'] + '\n')
#         file.write(content['程序分析'] + '\n')
#         file.write(content['程序源代码'] + '\n')
#         file.write( '\n' + '='*30 + '\n' ) #只是为了看起来略微条理些
#
#     time.sleep(0.2)    #由于每次要写内容量少,写得频繁,所以让写文件的节奏稍微缓一点,以免IO过载
#     num += 1
#
# print("mission complete")


#使用css模式过滤想要的元素
#1.获取html内容
url = "http://www.runoob.com/python/python-100-examples.html"
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}
content = requests.get(url,headers = headers).content.decode("utf-8") # 加个header伪装一下
#print(content)

#2.解析html,拿到所有想要的链接,存到列表里
soup = BeautifulSoup(content,"html.parser")
all_href = []
# all_a = soup.find(id="content").ul.find_all('a')
all_a = soup.select("#content > ul a")
# print(all_a)

#从标签中过滤出想要的href地址
# 方式一
for each_a in all_a:
    all_href.append(each_a.attrs['href'])
# print(all_href)

#方式二
# pattern = re.compile("href=\".*?\"")#因为匹配以"结尾,所以需要用非贪婪式
# for each in all_a:
#     href = pattern.findall(str(each))#获取到所有匹配内容的列表
#     all_href.append(href[0][6:-1])#因为只要链接地址,所以稍加切片一下,去除无用的头尾
# #print(all_href)
#
#3.逐个页面去爬取想要的内容
pre_url = 'http://www.runoob.com'
num = 1
for each_href in all_href:
    print("正在爬取第%s个页面"%num)
    sub_html = requests.get(pre_url+each_href).content.decode("utf-8") #由于给的都是相对路径地址,所以需要自己添加上地址头
    soup2 = BeautifulSoup(sub_html,"html.parser")
    #print(soup2)

    content = {}
    # content['标题'] = soup2.find(id='content').h1.text
    # content['题目'] = soup2.find(id='content').find_all('p')[1].text
    # content['程序分析'] = soup2.find(id='content').find_all('p')[2].text
    content['标题'] = soup2.select("#content h1")[0].text
    #content['题目'] = soup2.select("#content p:nth-of-type(2)")[0].text
    content['题目'] = soup2.select("#content p")[1].text
    content['程序分析'] = soup2.select("#content p:nth-of-type(3)")[0].text
    try:
        # content['程序源代码'] = soup2.find(id='content').find(class_='hl-main').text #由于class是关键字,故添加上下划线
        # content['程序源代码'] = soup2.find(id='content').find(name='div',attrs={'class':'hl-main'}).text
        content['程序源代码'] = soup2.select("#content .hl-main")[0].text
    except: #调试过程中捕捉到第42个页面对于当前规则不适应,那么特殊页面,特殊规则
        # content['程序源代码'] = soup2.find(id='content').pre.text
        content['程序源代码'] = soup2.select("#content pre")[0].text
    finally:
        pass #加个finally只是为了完整那么一点点

    #print(content)
    with open("python_100_examples.txt","a+",encoding='utf-8') as file: #因为内容爬的少,追加到一个文本文件里即可
        file.write(content['标题']+'\n')
        file.write(content['题目'] + '\n')
        file.write(content['程序分析'] + '\n')
        file.write(content['程序源代码'] + '\n')
        file.write( '\n' + '='*30 + '\n' ) #只是为了看起来略微条理些

    time.sleep(0.2)    #由于每次要写内容量少,写得频繁,所以让写文件的节奏稍微缓一点,以免IO过载
    num += 1

print("mission complete")