import requests
from bs4 import BeautifulSoup as bs
import re


def days():
    url = 'https://www.hltv.org/matches'

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 YaBrowser/20.12.3.140 Yowser/2.5 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    soup = bs(response.text, 'lxml')
    data = []
    block = soup.find_all('div', class_='upcomingMatchesSection')
    for i in range(3):
        data_of_the_matches = re.findall(r'^\w+', block[i].find('span', class_="matchDayHeadline").text)[0]
        data.append(data_of_the_matches)
    return data
