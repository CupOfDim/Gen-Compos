import json
import numpy as np
import pandas as pd
import re

df_path = 'your_path/content_dataframe.csv'
df = pd.read_csv(df_path)
df = df.dropna(subset=['genres', 'description'])
types_genres = {}
for t in df['types'].unique():
    type_df = df[df['types']==t]
    genres_split = type_df['genres'].str.split(', ')
    unique_genres = set(genre.lower() for subsplit in genres_split for genre in subsplit)
    types_genres[t] = list(unique_genres)
anim_manga_gen = list(set(types_genres['manga']) & set(types_genres['anime']))
west_cartoon_gen = list(set(types_genres['cartoon']) & set(types_genres['animated-series']))
film_series_gen = list(set(types_genres['movie']) & set(types_genres['tv-series']))

genres = {
    'movie':film_series_gen,
    'tv-series':film_series_gen,
    'cartoon':west_cartoon_gen,
    'animated-series':west_cartoon_gen,
    'anime':anim_manga_gen,
    'manga':anim_manga_gen,
}
ruTypes = {
    'movie':'фильм',
    'tv-series':'сериал',
    'cartoon':'мультфильм',
    'animated-series':'мульт-сериал',
    'anime':'аниме',
    'manga':'манга',
}

df_comb = []
for i in range(len(df)):
    row = df.iloc[i]
    row_gen = []
    for gen in row['genres'].split(', '):
        if gen in genres[row['types']]:
            row_gen.append(gen)
    outputs = re.sub(r'\s+', ' ', row['description']).strip()
    inputs = f'Продолжи: {ruTypes[row["types"]]}, с названием "{row["name"]}", c  жанрами "{", ".join(row_gen)}".' + '\n Ты:'
    df_comb.append({'input':inputs, 'output':outputs})

json_path = 'your_path/question_answer.json'
with open(json_path, 'w') as f:
    json.dump(df_comb, f)
