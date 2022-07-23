import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
from datetime import datetime


def matches(number_of_day):
    url = 'https://www.hltv.org/matches'

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 YaBrowser/20.12.3.140 Yowser/2.5 Safari/537.36'
    }
    data_frame = pd.DataFrame()
    response = requests.get(url, headers=headers)
    time.sleep(1)
    soup = bs(response.text, 'lxml')
    block = soup.find_all('div', class_='upcomingMatchesSection')[number_of_day]
    data_of_the_matches = block.find('div', class_='matchDayHeadline')
    matches = block.find_all('a', class_='match a-reset')
    for match in matches:
        time.sleep(0.5)
        time_of_the_match_unix = match.find('div', class_='matchTime').get('data-unix')
        time_of_the_match = datetime.fromtimestamp(int(time_of_the_match_unix)/1000).strftime('%H:%M')
        match_meta = match.find('div', class_='matchMeta').text
        Names_of_the_teams = match.find_all('div', class_='matchTeamName text-ellipsis')
        if Names_of_the_teams == [] or Names_of_the_teams[0] == Names_of_the_teams[-1]:
            continue
        team_1 = Names_of_the_teams[0].text
        team_2 = Names_of_the_teams[-1].text
        match_event = match.find('div', class_='matchEventName gtSmartphone-only').text
        link = 'https://www.hltv.org' + match.get('href')
        data_frame = data_frame.append({
            'time_of_the_match': time_of_the_match,
            'match_meta': match_meta,
            'match_event': match_event,
            'team_1': team_1, 'team_2': team_2,
            'URL': link
            },
            ignore_index=True)

    return data_frame

