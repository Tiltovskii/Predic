import requests
from bs4 import BeautifulSoup as bs
from catboost import CatBoostRegressor
import pandas as pd
import time
from datetime import datetime
import config


def search_site(url):
    data = {'date': [], 'team_1': [], 'team_2': [], 'Map': [], 'team_rank_1': [], 'team_rank_2': []}
    data_frame = pd.DataFrame(data)

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
         Chrome/87.0.4280.141 YaBrowser/20.12.3.140 Yowser/2.5 Safari/537.36'
    }
    time.sleep(0.5)
    response = requests.get(url, headers=headers)
    soup = bs(response.text, 'lxml')
    players_of_team1 = []
    players_of_team2 = []
    players = soup.find_all('td', class_='player')
    maps = soup.find_all('div', class_='mapholder')
    ranks = soup.find_all('div', class_='teamRanking')  # смотрит рейтинги команд, участвующих в матче
    if ranks[1].text != '\nUnranked\n' and ranks[0].text != '\nUnranked\n':  # выбрасываем матчи, где команды не имеют ранга
        for i in range(5):
            player1_html = players[i + 5]
            player2_html = players[-i - 1]
            player1 = player1_html.find('div', class_="text-ellipsis").text
            player2 = player2_html.find('div', class_="text-ellipsis").text
            players_of_team1.append(player1)
            players_of_team2.append(player2)
        if maps[0].find('div', class_='mapname').text == 'TBA':
            teams = soup.find_all('div', class_='teamName')
            for i in ['Dust2', 'Mirage', 'Inferno', 'Nuke', 'Overpass', 'Vertigo', 'Ancient']:
                data_unix = soup.find('div', class_='date').get('data-unix')
                data_frame = data_frame.append({
                    'date': datetime.utcfromtimestamp(int(data_unix) / 1000).strftime('%Y-%m-%d'),
                    'team_1': teams[0].text, 'team_2': teams[1].text, 'players1': players_of_team1, 'players2': players_of_team2,
                    'Map': i, 'team_rank_1': ranks[0].text.split('#')[1],
                    'team_rank_2': ranks[1].text.split('#')[1]},
                    ignore_index=True)
        else:
            for j in range(len(maps)):  # идем по всем картам
                data_unix = soup.find('div', class_='date').get('data-unix')
                teams = soup.find_all('div', class_='teamName')
                map = maps[j].find('div', class_='mapname').text
                data_frame = data_frame.append({
                        'date': datetime.utcfromtimestamp(int(data_unix) / 1000).strftime('%Y-%m-%d'),
                        'team_1': teams[0].text, 'team_2': teams[1].text, 'players1': players_of_team1, 'players2': players_of_team2,
                        'Map': map, 'team_rank_1': ranks[0].text.split('#')[1],
                        'team_rank_2': ranks[1].text.split('#')[1]},
                        ignore_index=True)
    return data_frame


def normalize(data_frame):
    Data_frame = pd.read_csv(config.Root + r'/rating_of_teams.csv', delimiter=',')  # тут путь поменять
    data_frame['date'], Data_frame['date'] = pd.to_datetime(data_frame['date']), pd.to_datetime(Data_frame['date'])
    data_frame['DeltaTime'] = (data_frame['date'] - Data_frame['date'][0]).dt.days
    data_frame = data_frame.drop(['date'], axis=1)
    try:
        for i in range(5):
            string = f'player{i + 1}_team1'
            data_frame[string] = data_frame['players1'].str[i]
        for i in range(5):
            string = f'player{i + 1}_team2'
            data_frame[string] = data_frame['players2'].str[i]
        data_frame = data_frame.drop(['players1', 'players2'], axis=1)
    except:
        for i in range(5):
            string = f'player{i + 1}_team1'
            data_frame[string] = None
        for i in range(5):
            string = f'player{i + 1}_team2'
            data_frame[string] = None
    columns_titles = ['DeltaTime', 'team_1', 'team_2', 'player1_team1', 'player2_team1', 'player3_team1',
                      'player4_team1', 'player5_team1', 'player1_team2', 'player2_team2', 'player3_team2',
                      'player4_team2', 'player5_team2', 'team_rank_1', 'team_rank_2',
                      'Map']
    data_frame = data_frame.reindex(columns=columns_titles)
    return data_frame


if __name__ == '__main__':
    url = input('Ссылка на матч:')
    df = search_site(url)
    dog = CatBoostRegressor()
    dog.load_model('predictor', format='cbm')  # файл с моделью
    df = normalize(df)  # выбрасывает ненужное
    print(df)
    df['scores'] = dog.predict(df)

    #  дальше все идет вывод данных для удобства копирования было сделано, поэтому полностью изменить

    print(df['team_1'][0]+' - '+df['team_2'][0])

    for i in range(len(df)):
        if i != 0 and df['team_1'][i] != df['team_1'][i-1]:
            print()
            print(df['team_1'][i] + ' - ' + df['team_2'][i])
        print(df['Map'][i],  round(df['scores'][i], 3))
