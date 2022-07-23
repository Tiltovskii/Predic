import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import time
from datetime import datetime
import config
import schedule


def parse(data_frame):
    url = 'https://www.hltv.org/results'

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 YaBrowser/20.12.3.140 Yowser/2.5 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    time.sleep(1)
    soup = bs(response.text, 'lxml')
    blocks = soup.find_all('div', class_='result-con')
    for i in range(len(blocks) - 1, -1, -1):
        time.sleep(0.5)
        link = 'https://www.hltv.org' + blocks[i].find('a').get('href')
        response = requests.get(link, headers=headers)
        soup1 = bs(response.text, 'lxml')
        players_of_team1 = []
        players_of_team2 = []
        players = soup1.find_all('td', class_='player')
        for i in range(5):
            player1_html = players[i + 5]
            player2_html = players[-i - 1]
            player1 = player1_html.find('div', class_="text-ellipsis").text
            player2 = player2_html.find('div', class_="text-ellipsis").text
            players_of_team1.append(player1)
            players_of_team2.append(player2)
        maps = soup1.find_all('div', class_='mapholder')  # смотрит все сыграные карты одного матча
        ranks = soup1.find_all('div', class_='teamRanking')  # смотрит рейтинги команд, участвующих в матче
        data_unix = soup1.find('div', class_='date').get('data-unix')
        date_yesterday = datetime.fromtimestamp(time.time() - 86400).strftime('%Y-%m-%d')
        date_match = datetime.fromtimestamp(int(data_unix) / 1000).strftime('%Y-%m-%d')
        if date_match == date_yesterday:
            if ranks[1].text != '\nUnranked\n' and ranks[
                0].text != '\nUnranked\n':  # выбрасываем матчи,где команды не имеют ранга
                for j in range(len(maps)):  # идем по всем картам
                    if maps[j].find('div',
                                    class_='results-left won pick') is not None:  # во всех условиях выкидываем карты,
                        teams = maps[j].find_all('div', class_='results-teamname text-ellipsis')
                        scores = maps[j].find_all('div', class_='results-team-score')
                        map = maps[j].find('div', class_='mapname').text
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[0].text, 'team_2': teams[1].text, 'player1_team1': players_of_team1[0],
                            'player2_team1': players_of_team1[1], 'player3_team1': players_of_team1[2],
                            'player4_team1': players_of_team1[3], 'player5_team1': players_of_team1[4],
                            'player1_team2': players_of_team2[0], 'player2_team2': players_of_team2[1],
                            'player3_team2': players_of_team2[2], 'player4_team2': players_of_team2[3],
                            'player5_team2': players_of_team2[4], 'score_1': scores[0].text,
                            'score_2': scores[1].text,
                            'Map': map, 'team_rank_1': ranks[0].text.split('#')[1],
                            'team_rank_2': ranks[1].text.split('#')[1]},
                            ignore_index=True)
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[1].text, 'team_2': teams[0].text, 'player1_team1': players_of_team2[0],
                            'player2_team1': players_of_team2[1], 'player3_team1': players_of_team2[2],
                            'player4_team1': players_of_team2[3], 'player5_team1': players_of_team2[4],
                            'player1_team2': players_of_team1[0], 'player2_team2': players_of_team1[1],
                            'player3_team2': players_of_team1[2], 'player4_team2': players_of_team1[3],
                            'player5_team2': players_of_team1[4],  'score_1': scores[1].text,
                            'score_2': scores[0].text,
                            'Map': map, 'team_rank_1': ranks[1].text.split('#')[1],
                            'team_rank_2': ranks[0].text.split('#')[1]},
                            ignore_index=True)

                    elif maps[j].find('div', class_='results-left lost pick') is not None:
                        teams = maps[j].find_all('div', class_='results-teamname text-ellipsis')
                        scores = maps[j].find_all('div', class_='results-team-score')
                        map = maps[j].find('div', class_='mapname').text
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[0].text, 'team_2': teams[1].text, 'player1_team1': players_of_team1[0],
                            'player2_team1': players_of_team1[1], 'player3_team1': players_of_team1[2],
                            'player4_team1': players_of_team1[3], 'player5_team1': players_of_team1[4],
                            'player1_team2': players_of_team2[0], 'player2_team2': players_of_team2[1],
                            'player3_team2': players_of_team2[2], 'player4_team2': players_of_team2[3],
                            'player5_team2': players_of_team2[4], 'score_1': scores[0].text,
                            'score_2': scores[1].text,
                            'Map': map, 'team_rank_1': ranks[0].text.split('#')[1],
                            'team_rank_2': ranks[1].text.split('#')[1]},
                            ignore_index=True)
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[1].text, 'team_2': teams[0].text, 'player1_team1': players_of_team2[0],
                            'player2_team1': players_of_team2[1], 'player3_team1': players_of_team2[2],
                            'player4_team1': players_of_team2[3], 'player5_team1': players_of_team2[4],
                            'player1_team2': players_of_team1[0], 'player2_team2': players_of_team1[1],
                            'player3_team2': players_of_team1[2], 'player4_team2': players_of_team1[3],
                            'player5_team2': players_of_team1[4], 'score_1': scores[1].text,
                            'score_2': scores[0].text,
                            'Map': map, 'team_rank_1': ranks[1].text.split('#')[1],
                            'team_rank_2': ranks[0].text.split('#')[1]},
                            ignore_index=True)

                    elif maps[j].find('div', class_='results-left won') is not None:
                        teams = maps[j].find_all('div', class_='results-teamname text-ellipsis')
                        scores = maps[j].find_all('div', class_='results-team-score')
                        map = maps[j].find('div', class_='mapname').text
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[0].text, 'team_2': teams[1].text, 'player1_team1': players_of_team1[0],
                            'player2_team1': players_of_team1[1], 'player3_team1': players_of_team1[2],
                            'player4_team1': players_of_team1[3], 'player5_team1': players_of_team1[4],
                            'player1_team2': players_of_team2[0], 'player2_team2': players_of_team2[1],
                            'player3_team2': players_of_team2[2], 'player4_team2': players_of_team2[3],
                            'player5_team2': players_of_team2[4], 'score_1': scores[0].text,
                            'score_2': scores[1].text,
                            'Map': map, 'team_rank_1': ranks[0].text.split('#')[1],
                            'team_rank_2': ranks[1].text.split('#')[1]},
                            ignore_index=True)
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[1].text, 'team_2': teams[0].text, 'player1_team1': players_of_team2[0],
                            'player2_team1': players_of_team2[1], 'player3_team1': players_of_team2[2],
                            'player4_team1': players_of_team2[3], 'player5_team1': players_of_team2[4],
                            'player1_team2': players_of_team1[0], 'player2_team2': players_of_team1[1],
                            'player3_team2': players_of_team1[2], 'player4_team2': players_of_team1[3],
                            'player5_team2': players_of_team1[4], 'score_1': scores[1].text,
                            'score_2': scores[0].text,
                            'Map': map, 'team_rank_1': ranks[1].text.split('#')[1],
                            'team_rank_2': ranks[0].text.split('#')[1]},
                            ignore_index=True)

                    elif maps[j].find('div', class_='results-left lost') is not None:
                        teams = maps[j].find_all('div', class_='results-teamname text-ellipsis')
                        scores = maps[j].find_all('div', class_='results-team-score')
                        map = maps[j].find('div', class_='mapname').text
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[0].text, 'team_2': teams[1].text, 'player1_team1': players_of_team1[0],
                            'player2_team1': players_of_team1[1], 'player3_team1': players_of_team1[2],
                            'player4_team1': players_of_team1[3], 'player5_team1': players_of_team1[4],
                            'player1_team2': players_of_team2[0], 'player2_team2': players_of_team2[1],
                            'player3_team2': players_of_team2[2], 'player4_team2': players_of_team2[3],
                            'player5_team2': players_of_team2[4], 'score_1': scores[0].text,
                            'score_2': scores[1].text,
                            'Map': map, 'team_rank_1': ranks[0].text.split('#')[1],
                            'team_rank_2': ranks[1].text.split('#')[1]},
                            ignore_index=True)
                        data_frame = data_frame.append({
                            'date': date_match,
                            'team_1': teams[1].text, 'team_2': teams[0].text, 'player1_team1': players_of_team2[0],
                            'player2_team1': players_of_team2[1], 'player3_team1': players_of_team2[2],
                            'player4_team1': players_of_team2[3], 'player5_team1': players_of_team2[4],
                            'player1_team2': players_of_team1[0], 'player2_team2': players_of_team1[1],
                            'player3_team2': players_of_team1[2], 'player4_team2': players_of_team1[3],
                            'player5_team2': players_of_team1[4], 'score_1': scores[1].text,
                            'score_2': scores[0].text,
                            'Map': map, 'team_rank_1': ranks[1].text.split('#')[1],
                            'team_rank_2': ranks[0].text.split('#')[1]},
                            ignore_index=True)
    return data_frame


def score(data_frame):
    data_frame['date'] = pd.to_datetime(data_frame['date'])
    data_frame['DeltaTime'] = (data_frame['date'] - data_frame['date'][0]).dt.days
    data_frame['score_1'] = data_frame['score_1'].astype('int')
    data_frame['score_2'] = data_frame['score_2'].astype('int')
    data_frame['score'] = 1 / (1 + np.exp(-(data_frame['score_1'] / data_frame['score_2']) + 1))
    data_frame.loc[data_frame['score'].copy() < 0.5, 'score'] = 1 - 1 / \
                                                (1 + np.exp(-(data_frame.loc[data_frame['score'].copy() < 0.5, 'score_2'] /
                                                              data_frame.loc[data_frame['score'].copy() < 0.5, 'score_1']) + 1))
    return data_frame


def parse_and_score():
    data = pd.read_csv(config.Root + r'/rating_of_teams.csv', delimiter=',')
    data = parse(data)
    data = score(data)
    data.to_csv(config.Root + r'/rating_of_teams.csv', index=False)  # тут путь поменять


schedule.every().day.at("01:00").do(parse_and_score)

while True:
    schedule.run_pending()
    time.sleep(1)

if __name__ == '__main__':
    df = pd.read_csv(config.Root + r'/rating_of_teams_experimental.csv', delimiter=',')
    df = parse(df)
    df = score(df)
    df.to_csv(config.Root + r'/rating_of_teams_experimental.csv', index=False)  # тут путь поменять
