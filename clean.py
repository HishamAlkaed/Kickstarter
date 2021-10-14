import pandas
import re
import spacy
from spacy.tokens import Doc, Token
from sklearn import preprocessing
import numpy as np

from os.path import splitext

nlp = spacy.load("en_core_web_sm")
pandas.set_option("display.max_colwidth", None)


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

def encode(feature, df):
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.asarray(df[feature]).reshape(-1, 1))
    encoded_feature = enc.transform(np.asarray(df[feature]).reshape(-1, 1))
    return encoded_feature, enc

def encode_features(df):
        # one hot encode the categorical features
    df.category, _ = encode('category', df)
    df.category = df.category.toarray()
    df.subcategory, _ = encode('subcategory', df)
    df.subcategory = df.subcategory.toarray()
    df.country, _ = encode('country', df)
    df.country = df.country.toarray()
    df.currency, _ = encode('currency', df)
    df.currency = df.currency.toarray()
    df.location, _ = encode('location', df)
    df.location = df.location.toarray()
    
    return df

def prepare(df):
    ''' This bad boy:
        - adds log hbl
        - adds hbd
            - fills country values that are missing witht the mode
            - removes null values and empty strings from blurb AND NAME
            - removes numeric values from blurb AND NAME
        - one hot encodes the categorical features /////Not Any More/////
        + vectorizes blurb  # TODO : No need if we use two models
        + vectorizes name  # TODO : No need if we use two models
    '''
    # hours before launch => log to normalize
    df['log_hbl'] = df.apply(lambda x: abs(np.log((x.launched_at - x.created_at)/3600)) , axis=1)
    # hours before deadline => not logged because of the 
    df['hbd'] = df.apply(lambda x: (x.deadline - x.launched_at)/3600 , axis=1)
    
    df = remove_unneeded(df)
    
#     df = encode_features(df)

    
    # vectorizes blurb

    return df

def remove_unneeded(df):
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
    
    df = df.drop(['pledged', 'usd_pledged', 'converted_pledged_amount', 'backers_count', 'project_id', 'created_at', 'launched_at', 'deadline', 'project_url', 'reward_url', 'fx_rate', 'location'], axis=1)
    
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
