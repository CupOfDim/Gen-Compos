import os
import json
import re
import config
from kinopoisk_dev import KinopoiskDev, MovieField, MovieParams
from kinopoisk_dev.model import MovieDocsResponseDto
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


def get_movies(TOKEN) -> MovieDocsResponseDto:
    kp = KinopoiskDev(token=TOKEN)
    data = []
    for i in range(1, 54):
        data += kp.find_many_movie(
            params=[
                MovieParams(keys=MovieField.PAGE, value=i),
                MovieParams(keys=MovieField.LIMIT, value=250),
            ]
        ).docs
    dict_list = []
    for i in range(len(data)):
        name = data[i].name
        cat = data[i].type
        describe = data[i].description
        genres = ', '.join([data[i].genres[j].name for j in range(len(data[i].genres))])
        j = {
            'name': name,
            'types': cat,
            'description': describe,
            'genres': genres
        }
        dict_list.append(j)

    return dict_list


def get_data(typeCont='manga'):

    def RequestContent(url, session, settings, headers):
        reaspons = session.post(url, headers=headers, json=settings)
        return reaspons
    if typeCont == 'manga':
        urlSess = "https://mangalib.me/manga-list?types[]=1"
        url = "https://mangalib.me/api/list"
        site_id = 1
    elif typeCont=='anime':
        urlSess = "https://animelib.me/anime-list?types[]=5"
        url = "https://animelib.me/api/list"
        site_id = 5
    sess = requests.Session()
    reaspons = sess.get(urlSess).text
    #XSRF, CSRF токены чтобы сайт не блокировал запросы
    XSRF = sess.cookies['XSRF-TOKEN']
    CSRF = reaspons.split('<meta name="_token" content="')[1].split('">')[0]

    headers = {
        'X-CSRF-TOKEN': CSRF,
        'X-XSRF-TOKEN': XSRF
        }
    data_content = []
    for i in range(1, 31):
        settings = {
            "sort": "rate",
            "dir": "desc",
            "page": i,
            "site_id": site_id,
            "caution_list": [
                "Отсутствует",
                "16+",
                "18+"
            ]
        }
        reaspons = RequestContent(url, sess, settings, headers)
        if typeCont == 'manga':
            data_content += eval('None'.join(reaspons.text.split('null')))['items']['data']
        elif typeCont=='anime':
            data_content += eval('None'.join(reaspons.text.replace('false','False').replace('true','True').split('null')))['items']['data']
    return data_content


def get_dict_data(data, cookies, headers, params, typeCont='manga'):
    data_dict = []
    for i in range(len(data)):
        if typeCont == 'anime':
            slug = data[i]['slug']
            ids = data[i]['id']
            summary = data[i]['summary']
            name = data[i]['rus_name']
            types = data[i]['modelType']
            response = requests.get(f'https://animelib.me/anime/{ids}-{slug}', params=params, cookies=cookies,
                                    headers=headers)
            src = response.text
            soup = BeautifulSoup(src, 'lxml')
        elif typeCont == 'manga':
            slug = data[i]['slug']
            types = data[i]['modelType']
            name = data[i]['rus_name']
            response = requests.get(f'https://mangalib.me/{slug}', params=params, cookies=cookies, headers=headers)
            src = response.text
            soup = BeautifulSoup(src, 'lxml')
            descrip = soup.find('div', class_='media-description__text')
            if descrip:
                summary = descrip.get_text(separator='\n')
                summary.replace('\n', '').replace('\r', '')
            else:
                summary = ''

        media_tags = soup.find('div', class_="media-tags")
        tag_text = []
        if media_tags:
            tags = media_tags.find_all('a')
            for tag in tags:
                tag_text.append(tag.get_text())
        tags = ", ".join(tag_text)
        d = {
            'name': name,
            'types': types,
            'description': summary,
            'genres': tags,
        }
        data_dict.append(d)
    return data_dict


if __name__ == "__main__":
    TOKEN = config.kinopoisk_token
    cookies = config.cookies
    headers = config.headers
    params = config.params

    movies_data = get_movies(TOKEN)
    manga_data = get_data()
    anime_data = get_data('anime')

    anime_dict = get_dict_data(anime_data, cookies, headers, params, 'anime')
    manga_dict = get_dict_data(manga_data, cookies, headers, params)

    movie_df = pd.DataFrame.from_dict(movies_data)
    anime_df = pd.DataFrame.from_dict(anime_dict)
    manga_df = pd.DataFrame.from_dict(manga_dict)

    df = pd.concat([movie_df, anime_df, manga_df], axis=0)

    df.to_csv('content_dataframe.csv', index=False)