'''
需求:
    获取博客园网站上所有精华帖的标题和内容
入口地址:
    https://www.cnblogs.com/pick/
流程分析:
    1.获取所有子链接
    2.对每个子链接进行获取标题和内容
'''
import requests,time
from lxml import etree
from bs4 import BeautifulSoup

#1.get the contents from the base page
pre_url = "https://www.cnblogs.com/pick/"
url = pre_url
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}
page = 1
while True:
    re = requests.get(url, headers=headers).content.decode('utf-8')

    #2.get each sub_links from html contents
    html = etree.HTML(re)
    soup = BeautifulSoup(re,'html.parser')

    # all_a = soup.select("a[class*='titlelnk']")
    # all_href = []
    # for each_a in all_a:
    #     all_href.append(each_a.attrs['href'])
    # print(all_href)

    all_href = html.xpath("//a[@class='titlelnk']/@href") #为了方便快捷,直接使用lxml一句话搞定href定位
    #print(all_href)
    next_page = html.xpath("//div[@class='pager']/a[last()]")

    next_page_link = next_page[0].xpath("@href")[0]
    next_page_content = next_page[0].xpath("text()")[0] #用来判断最后一页的终止,因为最后一页没有下一页了,其它页最后一个元素都是下一页
    # print(next_page_link)
    # print(next_page_content)

    #3.parse all htmls for each link
    num = 1
    for each_href in all_href:
        print("getting the %sth boke from page %sth"%(num,page) )
        re = requests.get(each_href,headers = headers).text
        #re = requests.get(each_href,headers = headers).content.decode('utf-8')

        html = etree.HTML(re)
        soup2 = BeautifulSoup(re,'html.parser')

        #get the title
        title = html.xpath("string(//a[@id='cb_post_title_url'])")
        #title = soup2.select("#cb_post_title_url")[0].text
        # print(title)

        #get the body
        body = html.xpath("string(//div[@id='cnblogs_post_body'])").replace("\n\n","")
        #body = soup2.select("#cnblogs_post_body")[0].text.replace("\n\n","")
        #print(body)

        #write to local file
        with open("cn-blogs.txt","a+",encoding='utf-8') as file:
            file.write(title+"\n")
            file.write(body+"\n")
            file.write("\n" + "="*80 +"\n")
        time.sleep(0.2)
        num += 1

    if next_page_content == "Next >":
        url = pre_url[:-6] + next_page_link #按需求切片获取每个下一页的链接地址
        #print(url)
        page += 1
    else:
        break

print("mission complete")