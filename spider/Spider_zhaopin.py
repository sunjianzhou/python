'''
需求:
    获取智联招聘上和人工智能相关的职业情况,包括
        招聘单位,福利待遇,薪资待遇及基本要求信息
入口地址:
    http://sou.zhaopin.com/
关键定位:
    第一层(所有职位链接):
        hrefs = html.xpath("//div[@id='search_right_demo']/div/div/a/@href")

        all_a = soup.select("#search_right_demo div div a")
        hrefs = []
        for each_a in all_a:
            hrefs.append("each_a.attrs['href']")
    第二层(某个具体职位链接):
        hrefs = html.xpath("//td[@class='zwmc']/div/a[1]/@href")

        all_a = soup.select(".zwmc div a")[0]
        hrefs = []
        for each_a in all_a:
            hrefs.append("each_a.attrs['href']")

        最后一页判断条件:
            对应的那个a标签里虽然还有下一页文字,但是不再有href属性
            html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href") != None
            soup.select(".pagesDown ul li a[class*='next-page']")[0].attrs['href'] != None
    第三层(目标层,某个职位的具体详细信息):
        职位名称:
            html.xpath("string(//div[@class='fl']/h1)")
            soup.select(".fl > h1").text
        公司名称:
            html.xpath("string(//div[@class='fl']/h2/a)")
            soup.select(".fl h2 a").text
        公司福利:
            html.xpath("string(//div[@class='welfare-tab-box']/span)")
            soup.select(".welfare-tab-box span").text
        基本要求信息:
            html.xpath("string(//div[@class='terminalpage-left']/ul")
            soup.select(".terminalpage-left ul").text
'''
import requests
from bs4 import BeautifulSoup
from lxml import etree

#1.获取第一层所有链接
def get_entrance(url,headers):
    re = requests.get(url,headers = headers).content.decode('utf-8')
    html = etree.HTML(re)
    soup = BeautifulSoup(re,"html.parser")

    # all_hrefs = html.xpath("//div[@id='search_right_demo']/div/div/a/@href")

    all_a = soup.select("#search_right_demo div div a")
    all_hrefs = []
    for each_a in all_a:
        all_hrefs.append(each_a.attrs['href'])

    all_hrefs = [ url[:-1] + x for x in all_hrefs]
    #all_hrefs = map(lambda x: url[:-1] + x, all_hrefs)
    return all_hrefs

#2.获取第二层所有链接
def get_job_entrance(url,headers):
    all_hrefs = []
    page = 1
    while True:
        print("getting the hrefs from page %s" %page )
        re = requests.get(url, headers=headers).content.decode('utf-8')
        html = etree.HTML(re)
        soup = BeautifulSoup(re, 'html.parser')

        hrefs = html.xpath("//td[@class='zwmc']/div/a[1]/@href")

        # all_a = soup.select(".zwmc div a")[0]
        # hrefs = []
        # for each_a in all_a:
        #     hrefs.append("each_a.attrs['href']")

        all_hrefs.extend(hrefs)

        if len(html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href")) != 0:
            url = html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href")[0]
            # url = soup.select(".pagesDown ul li a[class*='next-page']")[0].attrs['href']
            page += 1
        else:
            break
    return all_hrefs

#3.获取详细信息
def get_job_infos(url,headers):
    re = requests.get(url,headers = headers).content.decode("utf-8")
    html = etree.HTML(re)
    soup = BeautifulSoup(re,'html.parser')

    contents = {}
    contents['职位名称'] = html.xpath("string(//div[contains(@class,'fl')]/h1)")
    #contents['职位名称'] = soup.select(".fl h1")[0].text

    contents['公司名称'] = html.xpath("string(//div[contains(@class,'fl')]/h2/a)")
    #contents['公司名称'] = soup.select(".fl h2 a")[0].text

    contents['公司福利'] = html.xpath("//div[@class='welfare-tab-box']/span/text()")
    #contents['公司福利'] = soup.select(".welfare-tab-box span")[0].text

    contents['基本要求信息'] = html.xpath("string(//div[@class='terminalpage-left']/ul)")
    #contents['基本要求信息'] = soup.select(".terminalpage-left ul")[0].text

    print(contents)

#4.写入文件
def sort_to_txt(data,fileName):
    data = ','.join([str(i) for i in data.values()])
    with open(fileName,'a+',encoding='utf-8') as file:
        file.write(data+'\n')


if __name__ == '__main__':
    url = "http://sou.zhaopin.com/"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
    }
    with open('zl.txt','w+',encoding='utf-8') as file:
        file.write('zwmc,gsmc,gsfl,gzdd,fbrq,gzxz,gzjy,zdxl,zprs,zwlb,min_zwyx,max_zwyx'+'\n')

    basic_urls = get_entrance(url,headers)
    for basic_url in basic_urls:
        # all_jobs_links = get_job_entrance(basic_url, headers)
        # print(basic_url)
        pass
    print(basic_urls[0])

    #print(get_job_entrance(basic_urls[0],headers))
    one_job_link = "http://jobs.zhaopin.com/390497334250004.htm"
    get_job_infos(one_job_link,headers)
