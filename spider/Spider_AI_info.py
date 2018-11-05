'''
需求:
    获取智联招聘上和人工智能相关的职业情况,包括
        招聘单位,福利待遇,薪资待遇及基本要求信息
入口地址:
    http://sou.zhaopin.com/jobs/searchresult.ashx?jl=%E5%8C%97%E4%BA%AC&kw=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&sm=0&p=1
关键定位:
    第一层(某个具体职位链接,假定:不做过滤直接获取):
        hrefs = html.xpath("//td[@class='zwmc']/div/a[1]/@href")

        all_a = soup.select(".zwmc div a")[0]
        hrefs = []
        for each_a in all_a:
            hrefs.append("each_a.attrs['href']")

        最后一页判断条件:
            对应的那个a标签里虽然还有下一页文字,但是不再有href属性
            len(html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href")) != 0
            soup.select(".pagesDown ul li a[class*='next-page']")[0].attrs['href'] != ""
    第二层(目标层,某个职位的具体详细信息):
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
import re,requests,time
from bs4 import BeautifulSoup
from lxml import etree

#1.获取第一层所有链接
def get_job_entrance(url,headers):
    all_hrefs = []
    page = 1
    while True:

        res = requests.get(url, headers=headers).content.decode('utf-8')
        html = etree.HTML(res)
        soup = BeautifulSoup(res, 'html.parser')

        # hrefs = html.xpath("//td[@class='zwmc']/div/a[1]/@href") #获取所有href
        all_a = html.xpath("//td[@class='zwmc']/div/a[1]")
        hrefs = []

        expect_position = [""]
        num = 1
        for each_a in all_a:
            print("getting the %sth hrefs from page %s" %(num,page))
            position = each_a.xpath("string()")
            # print(position)
            #数据简单过滤一下,没必要的数据咱就不收集了
            if position.find("人工智能") != -1 \
                    or position.lower().find("ai") != -1 \
                    or position.find("机器学习") != -1 \
                    or position.find("深度学习") != -1 \
                    or position.find("自然语言") != -1  \
                    or position.find("图像识别") != -1 \
                    or position.find("数据挖掘") != -1:
                hrefs.append(each_a.xpath("@href")[0])
            num += 1

        all_hrefs.extend(hrefs)

        if len(html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href")) != 0:
            url = html.xpath("//div[@class='pagesDown']/ul/li/a[contains(@class,'next-page')]/@href")[0]
            # url = soup.select(".pagesDown ul li a[class*='next-page']")[0].attrs['href']
            page += 1
        else:
            break
    return all_hrefs

#2.获取详细信息
def get_job_infos(url,headers):
    res = requests.get(url,headers = headers).content.decode("utf-8")
    html = etree.HTML(res)
    soup = BeautifulSoup(res,'html.parser')

    contents = {}
    contents['职位名称'] = html.xpath("string(//div[contains(@class,'fl')]/h1)")
    #contents['职位名称'] = soup.select(".fl h1")[0].text

    contents['公司名称'] = html.xpath("string(//div[contains(@class,'fl')]/h2/a)")
    #contents['公司名称'] = soup.select(".fl h2 a")[0].text

    contents['公司福利'] = html.xpath("//div[@class='welfare-tab-box']/span/text()")
    #contents['公司福利'] = soup.select(".welfare-tab-box span")[0].text

    #contents['基本要求信息'] = html.xpath("string(//div[@class='terminalpage-left']/ul)").replace("\r\n","\t")
    #contents['基本要求信息'] = soup.select(".terminalpage-left ul")[0].text.replace("\r\n","\t")

    contents['职位月薪'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[1]/strong)")
    contents['工作地点'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[2]/strong)")
    contents['发布日期'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[3]/strong)")
    contents['工作经验'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[5]/strong)")
    contents['最低学历'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[6]/strong)")
    contents['招聘人数'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[7]/strong)")
    contents['职位类别'] = html.xpath("string(//div[@class='terminalpage-left']/ul/li[8]/strong)")

    print(contents)
    #简单清洗一下数据:
    #职位要求就到大专级别的不看, 月薪面议的先不看了, 最低薪资低于一万五的咱也没热情, 咱是实在人说实在话.
    if contents['最低学历'].find("大专") != -1:
        contents['职位月薪'] = ""

    if contents['职位月薪'].find("面议") != -1:
        contents['职位月薪'] = ""

    #月薪数据类似长这样: 15001-20000元/月
    pattern = re.compile("\d+")
    salary = pattern.findall(contents['职位月薪'])
    if len(salary) == 2:
        contents['最低月薪'] = salary[0]
        if int(contents['最低月薪']) < 15000:
            contents['职位月薪'] = ""
        contents['最高月薪'] = salary[1]
    else:
        contents['职位月薪'] = ""
    return contents

#3.写入文件
def sort_to_txt(contents):
    if contents['职位月薪'] == "":
        return
    datas = ','.join([str(i) for i in contents.values()])
    with open("aiSalary.txt",'a+',encoding='utf-8') as file:
        file.write("最低月薪:" + contents['最低月薪'] + "\t" + "最高月薪:" + contents['最高月薪'] + "\t" + datas + "\n")
        file.write("\n")

if __name__ == '__main__':
    url = "http://sou.zhaopin.com/jobs/searchresult.ashx?jl=%E5%8C%97%E4%BA%AC&kw=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD&sm=0&p=1"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
    }
    all_jobs_links = get_job_entrance(url, headers)
    for each_job_link in all_jobs_links:
        contents = get_job_infos(each_job_link,headers)
        sort_to_txt(contents)
        #time.sleep(0.02)
    print("mission complete")