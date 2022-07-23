import pandas as pd
import telebot
import config
import Parse_matches_of_the_day
from Predict import search_site, normalize
from catboost import CatBoostRegressor
from datetime import datetime
import time
import re
import Parse_days
import numpy as np

bot = telebot.TeleBot(config.Token)

eng_to_rus_day = {'Monday': 'Понедельник (-_-)',
                  'Sunday': 'Воскресенье (>^_^)>',
                  'Tuesday': 'Вторник (>_<)',
                  'Wednesday': 'Среду (@_@)',
                  'Thursday': 'Четверг (=_=)',
                  'Friday': 'Пятницу \(^O^)/',
                  'Saturday': 'Субботу (^.^)/'}

imen_eng_to_rus_day = {'Monday': 'Понедельник (-_-)',
                       'Sunday': 'Воскресенье (>^_^)>',
                       'Tuesday': 'Вторник (>_<)',
                       'Wednesday': 'Среда (@_@)',
                       'Thursday': 'Четверг (=_=)',
                       'Friday': 'Пятница \(^O^)/',
                       'Saturday': 'Суббота (^.^)/'}


@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(
        message.chat.id,
        'Привет, я могу предсказывать результаты матчей в CS:GO!\n' +
        'Чтобы вызвать список ближайших матчей нажмите /todaymatches.\n' +
        'Чтобы выбрать день матча нажмите /dayofthematches'
    )


@bot.message_handler(commands=['help'])
def start_command(message):
    bot.send_message(
        message.chat.id,
        'Привет, я могу предсказывать результаты матчей в CS:GO!\n' +
        'Чтобы вызвать список ближайших матчей нажмите /todaymatches.\n' +
        'Чтобы выбрать день матча нажмите /dayofthematches'
    )


@bot.message_handler(commands=['dayofthematches'])
def days(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    date = Parse_days.days()
    print(date)
    for i, day in enumerate(date):
        # day = datetime.fromtimestamp(time.time() + i * 86400).strftime('%A\n%d %B')
        # day_to_callback = datetime.fromtimestamp(time.time() + i * 86400).strftime('%d')
        keyboard.row(
            telebot.types.InlineKeyboardButton(f'{imen_eng_to_rus_day[day]}', callback_data='d' + str(i))
        )
    bot.send_message(
        message.chat.id,
        'Привет, выбери день, в который будет проходить матч.',
        reply_markup=keyboard
    )


def most_valuable_predict_and_unknown_teams(df):
    scores = []
    unknown_teams = set()
    for number_of_the_match in range(df.shape[0]):
        url = df['URL'][number_of_the_match]
        df_of_the_match = search_site(url)
        df_of_the_match = normalize(df_of_the_match)
        dog = CatBoostRegressor()
        dog.load_model('predictor', format='cbm')
        df_of_the_match['scores'] = dog.predict(df_of_the_match)
        mean = df_of_the_match['scores'].mean(axis=0)
        if pd.isnull(mean):
            mean = 0.5
            unknown_teams.add(number_of_the_match)
        scores.append(mean)
    scores = abs(np.array(scores) - 0.5)
    return np.argmax(scores), unknown_teams


def update_keyboard(df, number_of_the_day=0):
    keyboard = telebot.types.InlineKeyboardMarkup()
    best_predict, unknown_teams = most_valuable_predict_and_unknown_teams(df)
    for i in range(df.shape[0]):
        team1 = df['team_1'][i]
        team2 = df['team_2'][i]
        date = df['time_of_the_match'][i]
        if i in unknown_teams:
            keyboard.row(
                telebot.types.InlineKeyboardButton(f'{team1} vs {team2} ❌\n{date}', callback_data='m' + str(i) + '%' + str(number_of_the_day))
            )
        else:
            keyboard.row(
                telebot.types.InlineKeyboardButton(f'{team1} vs {team2}\n{date}',
                                                   callback_data='m' + str(i) + '%' + str(number_of_the_day))
            )
    keyboard.row(
        telebot.types.InlineKeyboardButton('Лучший предикт', callback_data='m' + str(best_predict) + '%' + str(number_of_the_day))
    )
    keyboard.row(
        telebot.types.InlineKeyboardButton('Обновить'@bot.callback_query_handler(func=lambda call: True) + str(number_of_the_day))
    )
    return keyboard


@bot.message_handler(commands=['todaymatches'])
def list_of_matches(message, number_of_the_day=0):
    day_of_the_week_eng = Parse_days.days()[number_of_the_day]
    day_of_the_week_rus = eng_to_rus_day[day_of_the_week_eng]
    bot.send_message(
        message.from_user.id,
        f'Смотрим матчи на {day_of_the_week_rus}...'
    )
    bot.send_chat_action(message.from_user.id, 'typing')
    df = Parse_matches_of_the_day.matches(number_of_the_day)
    df.to_csv(config.Root + r'/list_of_matches' + str(number_of_the_day) + r'.csv', index=False)
    if df.empty:
        bot.send_message(
            message.from_user.id,
            f'Пока матчей нет :D',
        )
    else:
        keyboard = update_keyboard(df, number_of_the_day)
        # day = datetime.fromtimestamp(time.time() + number_of_the_day * 86400).strftime('%A\n%d %B')
        bot.send_message(
            message.from_user.id,
            f'Список матчей на {day_of_the_week_rus}:',
            reply_markup=keyboard
        )


def probability(number_of_the_match, number_of_the_day=0):
    df = pd.read_csv(config.Root + r'/list_of_matches' + str(number_of_the_day) + r'.csv', delimiter=',')
    url = df['URL'][number_of_the_match]
    team1 = df['team_1'][number_of_the_match]
    team2 = df['team_2'][number_of_the_match]
    date = df['time_of_the_match'][number_of_the_match]
    df_of_the_match = search_site(url)
    dog = CatBoostRegressor()
    dog.load_model('predictor', format='cbm')
    df_of_the_match = normalize(df_of_the_match)
    df_of_the_match['scores'] = dog.predict(df_of_the_match)
    result = f'Игра {team1} vs {team2}\nВероятность победы {team1}\n'
    for i in range(df_of_the_match.shape[0]):
        map = df_of_the_match['Map'][i]
        score = df_of_the_match['scores'][i]
        result += f'{map}: {round(score * 100, 2)}%\n'
    mean = df_of_the_match['scores'].mean(axis=0)
    result += f'Среднее по картам: {round(mean*100, 2)}%'
    return result


@bot.callback_query_handler(func=lambda call: True)
def iq_callback(query):
    data = query.data
    bot.send_chat_action(query.message.chat.id, 'typing')
    bot.answer_callback_query(query.id)
    if data.startswith('m'):
        try:
            number_of_the_match = int(re.findall(r'^(\w+)', data)[0][1:])
            number_of_the_day = int(re.findall(r'%(\w+)', data)[0])
            result = probability(number_of_the_match, number_of_the_day)
            bot.send_message(
                query.from_user.id,
                result
            )
        except ValueError:
            pass
    elif data.startswith('update'):
        try:
            number_of_the_day = int(data[6:])
            df = Parse_matches_of_the_day.matches(number_of_the_day)
            df.to_csv(config.Root + r'/list_of_matches' + str(number_of_the_day) + r'.csv', index=False)
            day_of_the_week_eng = datetime.fromtimestamp(time.time() + number_of_the_day * 86400).strftime('%A')
            day_of_the_week_rus = eng_to_rus_day[day_of_the_week_eng]
            bot.edit_message_text(
                f'Список матчей на {day_of_the_week_rus}:',
                query.message.chat.id,
                query.message.message_id,
                reply_markup=update_keyboard(df, number_of_the_day)
            )
        except ValueError:
            pass
    else:
        try:
            data = int(data[1:])
            list_of_matches(query, data)
        except ValueError:
            pass


bot.polling(none_stop=True)
