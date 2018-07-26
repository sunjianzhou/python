#链家所有二手房

import requests,re
from bs4 import BeautifulSoup
from lxml import etree
#1、获取html内容
pre_url = "https://bj.lianjia.com/ershoufang/"
url = "https://bj.lianjia.com/ershoufang/"
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}
content = requests.get(url,headers=headers).content.decode('utf-8')
html = etree.HTML(content)
#print(content)

#2、解析首页html内容，拿到所有想要链接
soup = BeautifulSoup(content,'html.parser')
all_href = []
link_num = soup.select('div[class*="house-lst-page-box"]')[0].attrs['page-data']

pattern = re.compile('\d+')
numbers = pattern.findall(link_num)
if len(numbers) == 2:
    page_number = int(numbers[0])
#print(page_number)

#看起来就是拿不到div下的a，就找了一下link的规律，换一种简单的替代方式自己替代了
# print( soup.select('div[class*="house-lst-page-box"]'))
# sublinks = html.xpath("//div[contains(@class,'house-lst-page-box')]/a")
# print(sublinks)

all_href.append(url)
for i in range(2,page_number+1):
    each_link = pre_url + 'pg' + str(i)
    #print(each_link)
    all_href.append(each_link)

#对每页内容获取想要的信息
target_num = 0
for num in range(len(all_href)):
    print("正在查询第{}页".format(num))
    sub_url = all_href[num]
    sub_content = requests.get(sub_url,headers=headers).content.decode('utf-8')
    soup = BeautifulSoup(sub_content, 'html.parser')
    sub_html = etree.HTML(sub_content)
    all_infos = sub_html.xpath("//div[@class='info clear']")
    for i in range(len(all_infos)):
        houseInfo = soup.select("div[class*='info clear'] div[class*='houseInfo']")[i].text
        totalPrice = soup.select("div[class*='info clear'] div[class*='totalPrice']")[i].text
        locationInfo = soup.select("div[class*='info clear'] div[class*='tag']")[i].text
        unitPrice = soup.select("div[class*='info clear'] div[class*='unitPrice']")[i].text
        unitPrice = unitPrice.split('单价')[1]
        layer = soup.select("div[class*='info clear'] div[class*='positionInfo']")[i].text

        price = totalPrice.split('万')[0]
        if float(price) > 300.0:
            continue

        area = houseInfo.split('平米')[0].split('/')[-1]
        if float(area) < 75.0:
            continue

        target_num += 1
        print(houseInfo + "  "+ totalPrice + "  " + locationInfo)
        with open("lianjia2shoufang.txt", 'a+', encoding='utf-8') as file:
            file.write(houseInfo + "  "+ totalPrice + "  " + locationInfo + "  " + unitPrice + "  " + layer)
            file.write("\n")
print("目标内总房数：",target_num,end=' ')
print("筛选条件：面积大于75，总价小于300w")