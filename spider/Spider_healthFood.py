import requests,time
from bs4 import BeautifulSoup
'''
入口地址:https://www.xinshipu.com/chishenme/114208/
需求:
    筛选食谱名字中带有养颜两字的食谱,获取其简介,材料及做法
'''
#1.get the entry adress
url = "https://www.xinshipu.com/chishenme/114208/"
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}
html = requests.get(url,headers = headers).content.decode('utf-8')
#print(html)

#2.parse html to get the whole target links
soup = BeautifulSoup(html,'html.parser')
soup.prettify() # beautify the format
#print(soup)

#all_a = soup.find(name='div',attrs={'class':'new-menu-list'}).find_all('a')
all_a = soup.select("div[class*=new-menu-list] a")
#print(all_a)

all_href = []
for each_a in all_a:
    if each_a.text.strip().find("养颜") != -1:
        all_href.append(each_a.attrs['href'])
#print(all_href)

#3.get the target contents from each link
pre_url = "https://www.xinshipu.com/"
num = 0
for each_url in all_href:
    print("正在获取第%s份食谱" % num)
    sub_html = requests.get(pre_url+each_url,headers = headers).content.decode('utf-8')
    soup2 = BeautifulSoup(sub_html,'html.parser')
    #all_divs = soup2.find('div',attrs={'class':'re-steps'}).find_all('div',attrs={'class':'clearfix'})
    all_divs = soup2.select("div[class*='re-steps'] div[class*=clearfix]")

    contents={}
    contents['简介'] = soup2.select("div[class*='re-steps'] div[class*=clearfix] div[class*=cg2]")[0].text + ": " + \
                     soup2.select("div[class*='re-steps'] div[class*=clearfix] div[class=dd]")[0].text.strip().replace("\n\n","")
    contents['材料'] = soup2.select("div[class*='re-steps'] div[class*=clearfix] div[class*=cg2]")[1].text + ": "
    for each in soup2.select("div[class*='re-steps'] em a"):
        contents['材料'] += " " + each.text.strip().replace("\n\n","")
    contents['做法'] = soup2.select("div[class*='re-steps'] div[class*=clearfix] div[class*=cg2]")[2].text + ": "
    for each in soup2.select("div[class*='re-steps'] ol li"):
        contents['做法'] += " " + each.text.strip().replace("\n\n","")
    #print(contents)

    with open("foods_2.txt", "a+", encoding="utf-8") as file:
        file.write("第%s份食谱" % num + "\n")
        file.write(contents['简介'].strip()+"\n")
        file.write(contents['材料'].strip() + "\n")
        file.write(contents['做法'].strip() + "\n")
        file.write("=" * 80 + "\n" )
    time.sleep(0.2)
    num += 1

print("mission complete")
