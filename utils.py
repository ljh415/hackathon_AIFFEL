#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import random


# In[2]:


dict_categories = ['Cinema - DVD', 'PC Games - Standard Editions',
                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',
                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',
                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',
                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',
                    'Programs - Home and Office', 'Gifts - Development',
                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',
                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',
                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',
                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',
                    'Accessories - PS4', 'Games - PSVita',
                    'Books - Methodical materials 1C', 'Payment cards - PSN',
                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',
                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',
                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',
                    'Gifts - Attributes', 'Service Tools',
                    'Music - branded production CD', 'Payment cards - Live!',
                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',
                    'Music - Music Video', 'Game Consoles - PS3',
                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',
                    'Books - Audiobooks 1C', 'Books - Digit',
                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',
                    'Accessories - XBOX ONE', 'Pure media (piece)',
                    'Programs - Home and Office (Digital)', 'Programs - Educational',
                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',
                    'Programs - Educational (Digit)', 'Accessories - PSP',
                    'Gaming consoles - XBOX ONE', 'Delivery of goods',
                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',
                    'Music - Gift Edition', 'Service Tools - Tickets',
                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',
                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',
                    'Gifts - Certificates, Services', 'Android Games - Digit',
                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',
                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',
                    'Books - Computer Literature', 'Books - Travel Guides',
                    'PC - Headsets / Headphones', 'Books - Fiction',
                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',
                    'Books - Cognitive literature']

dict_shops = ['Moscow Shopping Center "Semenovskiy"', 
              'Moscow TRK "Atrium"', 
              "Khimki Shopping Center",
              'Moscow TC "MEGA Teply Stan" II', 
              'Yakutsk Ordzhonikidze, 56',
              'St. Petersburg TC "Nevsky Center"', 
              'Moscow TC "MEGA Belaya Dacha II"',
              'Voronezh (Plekhanovskaya, 13)', 
              'Yakutsk Shopping Center "Central"',
              'Chekhov SEC "Carnival"', 
              'Sergiev Posad TC "7Ya"',
              'Tyumen TC "Goodwin"',
              'Kursk TC "Pushkinsky"', 
              'Kaluga SEC "XXI Century"',
              'N.Novgorod Science and entertainment complex "Fantastic"',
              'Moscow MTRC "Afi Mall"',
              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',
              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',
              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',
              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',
              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',
              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',
              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',
              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',
              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',
              "Volzhsky Shopping Center 'Volga Mall' ",
              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',
              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',
              'Online store of emergencies', 'Adygea Shopping Center "Mega"',
              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',
              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',
              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',
              'RostovNaDonu TRK "Megacenter Horizon"',
              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',
              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",
              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',
              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',
              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",
              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']


# In[3]:


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y") # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday #extracting week day
    df["_day"] = df['date'].dt.day # extracting day
    df["_month"] = df['date'].dt.month # extracting month
    
    return df #returning the df after the transformations

def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):
    temp = cols
    cm = sns.light_palette("green", as_cmap=True)
    return pd.crosstab(df[temp[0]], df[temp[1]], 
                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)

def quantiles(df, columns):
    for name in columns:
        print(name + " quantiles")
        print(df[name].quantile([.01,.25,.5,.75,.99]))
        print("")

def chi2_test(col ):
    stat, p, dof, expected = stats.chi2_contingency((pd.crosstab(df_train[col], df_train.item_cnt_day)))
    # interpret test-statistic
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print(f"Testing the {col} by Total Items Sold")
    print('dof=%d' % dof)
    print(p)
    print("Critical Result: ")
    if abs(stat) >= critical:
        print(f"Critical {round(critical,4)}")
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
    print("")
    alpha = 1.0 - prob
    print("P Value: ")
    if p <= alpha:
        print(f"P-Value: {round(p,8)}")
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
def log_transforms(df, cols):
    for col in cols:
        df[col+'_log'] = np.log(df[col] + 1)
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df

def knowningData(df, limit=5): #seting the function with df, 
    print(f"Dataset Shape: {df.shape}")
    print('Unique values per column: ')
    print(df.nunique())
    print("################")
    print("")    
    for column in df.columns: #initializing the loop
        print("Column Name: ", column )
        entropy = round(stats.entropy(df[column].value_counts(), base=2),2)
        print("entropy ", entropy, 
              " | Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | Total unique values: ", df.nunique()[column], #print the data and % of nulls
              " | Missing: ", df[column].isna().sum())
        print("Top 5 most frequent values: ")
        print(round(df[column].value_counts()[:limit] / df[column].value_counts().sum() * 100,2))
        print("")
        print("####################################")


# In[ ]:




