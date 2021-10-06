# CLI
# import click

# Data
import pandas
import re
import spacy
from spacy.tokens import Doc, Token

# Other
from os.path import splitext

# import en_core_web_sm
# nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm")
pandas.set_option("display.max_colwidth", None)


def clean_text(text: str) -> str:
    # To lower-case
    text = text.lower()

    # Remove full URLs
#     text = re.sub("http\S+", "", text)

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


# @click.group()
def cli():
    pass


# @cli.command()
# @click.argument("text")
def test(text: str):
    """Allows you to test the text cleaning without a dataset."""

    new_text = clean_text(text)
    print(f"Text without processing:\n{text}\n\nText with processing:\n{new_text}")


def remove_unneeded(df):
    df.country = df.country.fillna(df.country.mode().iloc[0])
    df = df.drop([5423, 27780]) # remove null blurbs
    # remove float numbers from blurb
    df = df.drop([31717, 50117]) # remove float blurbs
    # remove goals higher than 1 mil (OUTLIERS)
    df = df.drop(df[df.goal > 1000000].index.tolist())
    
    df = df.drop(['pledged', 'usd_pledged', 'converted_pledged_amount', 'backers_count', 'project_id', 'created_at', 'launched_at', 'deadline', 'project_url', 'reward_url', 'fx_rate'], axis=1)
    
    return df 

# @cli.command()
# @click.argument("file_name", type=click.Path(exists=True))
# @click.option("--limited/--full", default=True, help="Limited output")
def clean(data, limited: bool, feature: str):
    """Transforms an entire dataset into clean data."""

    # Select the first 50 items during testing
    if limited:
        data = data[:50]

    # Re-map the text column with cleaned text
    data[feature] = data[feature].map(clean_text)

    # Drop the `id` column
    data = data.drop("project_id", axis=1)
    data = data.reset_index(drop=True)

    # Drop duplicates
#     data = data.drop_duplicates(subset=[feature])

#     if limited:
#         print(f"Cleaned items:\n{data}\n")

#     print(data.info())

    # Export data to CSV
#     file_name, _ = splitext('d)
    file_name = f"clean_dataset.csv"
    data.to_csv(file_name, index=False)
    return data


# if __name__ == "__main__":
#     cli()