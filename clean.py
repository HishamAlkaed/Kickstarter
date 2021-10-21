import pandas

import re
# import spacy
# from spacy.tokens import Doc, Token
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import category_encoders as ce

from os.path import splitext

# nlp = spacy.load("en_core_web_sm")
# pandas.set_option("display.max_colwidth", None)


def clean_text(text: str) -> str:
    # To lower-case
    text = text.lower()

    # NLP with Spacy
    tokens: list[Token] = nlp(text)

    filtered_str: list[str] = []
    for token in tokens:
        # Check if token is not punct or space or non-unicode
        if (
            not token.is_space
            and token.is_alpha
            and not token.is_stop
            and not token.is_punct
        ):
            filtered_str.append(token.lemma_)

    text = " ".join(filtered_str)

    return text

def scale_data(df, features=['blurb', 'name', 'hbl', 'goal', 'funded']):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.drop(features, axis=1))
    return df_scaled


def onehot_encode(df):
    encoder = ce.one_hot.OneHotEncoder(cols=['category', 'country', 'currency', 'subcategory', 'staff_pick'])

    df = encoder.fit_transform(df)
    return df

def prepare(df):
    ''' This bad boy:
        - adds log hbl
        - adds hbd
            - fills country values that are missing witht the mode
            - removes null values and empty strings from blurb AND NAME
            - removes numeric values from blurb AND NAME
        - one hot encodes the categorical features /////Not Any More/////
    '''
    # hours before launch => log to normalize
    df['log_hbl'] = df.apply(lambda x: abs(np.log((x.launched_at - x.created_at)/3600)) , axis=1)
    # hours before deadline => not logged because of the 
    df['hbd'] = df.apply(lambda x: (x.deadline - x.launched_at)/3600 , axis=1)
    # convert the goal to usd currency.
    df['goal_usd'] = df.apply(lambda x: x.goal * x.fx_rate, axis=1)
    # 
    df['len_blurb'] = df.apply(lambda x: len(x.blurb.split()), axis=1)
    
    df = onehot_encode(df)

    return df

def remove_unneeded(df, mode='training'):
    ''' This bad boy:
        - fills country values that are missing witht the mode
        - removes null values and empty strings from blurb AND NAME
        - removes numeric values from blurb AND NAME
    '''
    df.country = df.country.fillna(df.country.mode().iloc[0])
    
    # remove null values from blurb AND NAME
    df = df.dropna(subset=['blurb'])
    df = df.dropna(subset=['name'])
    
    # remove numeric values from blurb AND NAME
    df = df.drop(df[df.blurb.str.isnumeric()].index)
    df = df.drop(df[df.name.str.isnumeric()].index)
    
    # remove goals higher than 1 mil (OUTLIERS)
#     df = df.drop(df[df.goal > 1000000].index.tolist())
    if mode == 'training':
        df = df.drop(['pledged', 'usd_pledged', 'converted_pledged_amount', 'backers_count', 'created_at', 'launched_at', 'deadline', 'project_url', 'reward_url', 'location'], axis=1)
    else:
        df = df.drop(['project_url', 'reward_url', 'location'], axis=1)
    
    return df 

def clean(data, limited: bool):
    """Transforms an entire dataset into clean data."""

    # Select the first 50 items during testing
    if limited:
        data = data[:50]
        
#     data = prepare(data)
    
    # Re-map the text column with cleaned text
    data['blurb'] = data['blurb'].map(clean_text)
    data['name'] = data['name'].map(clean_text)

    data = data.reset_index(drop=True)

    # Export data to CSV
#     file_name, _ = splitext('d)
    file_name = f"clean_dataset.csv"
    data.to_csv(file_name, index=False)
    return data
