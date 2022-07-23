from catboost import CatBoostRegressor
import pandas as pd
import time
from datetime import datetime
import config
import schedule


def fit():
    DF = pd.read_csv(config.Root + r'/rating_of_teams.csv', delimiter=',')  # тут путь поменять
    DF['date'] = pd.to_datetime(DF['date'])
    date = datetime.fromtimestamp(time.time() - 92 * 86400).strftime('%Y-%m-%d')
    X, y = DF.drop(['score_1', 'score_2', 'score', 'date'], axis=1), DF['score']
    cat_features = ['team_1', 'team_2', 'player1_team1', 'player2_team1', 'player3_team1',
                    'player4_team1', 'player5_team1', 'player1_team2', 'player2_team2', 'player3_team2',
                    'player4_team2', 'player5_team2', 'Map', 'team_rank_1', 'team_rank_2']
    columns_titles = ['DeltaTime', 'team_1', 'team_2', 'player1_team1', 'player2_team1', 'player3_team1',
                      'player4_team1', 'player5_team1', 'player1_team2', 'player2_team2', 'player3_team2',
                      'player4_team2', 'player5_team2', 'team_rank_1', 'team_rank_2', 'Map']
    X = X.reindex(columns=columns_titles)
    X = X[DF['date'] > date]
    y = y[DF['date'] > date]
    cat = CatBoostRegressor()
    return cat.fit(X, y, cat_features=cat_features)


def fit_and_save():
    dog = fit()
    dog.save_model('predictor',
                   format="cbm")


schedule.every().day.at("01:05").do(fit_and_save)

while True:
    schedule.run_pending()
    time.sleep(1)

