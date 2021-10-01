# CLI
import click

# Data
import pandas
import re
import spacy
from spacy.tokens import Doc, Token

# Other
from os.path import splitext

nlp = spacy.load("en_core_web_sm")
pandas.set_option("display.max_colwidth", None)


def clean_text(text: str) -> str:
    # To lower-case
    text = text.lower()

    # Remove full URLs
    text = re.sub("http\S+", "", text)

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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("text")
def test(text: str):
    """Allows you to test the text cleaning without a dataset."""

    new_text = clean_text(text)
    print(f"Text without processing:\n{text}\n\nText with processing:\n{new_text}")


@cli.command()
@click.argument("file_name", type=click.Path(exists=True))
@click.option("--limited/--full", default=True, help="Limited output")
def clean(file_name: str, limited: bool):
    """Transforms an entire dataset into clean data."""

    # Open the dataset
    data = pandas.read_csv(file_name)

    # Drop the location column as we currently have no use for it.
    # TODO: Figure out how to clean/use location?
    data = data.drop("location", axis=1)

    # Drop all rows with missing values
    data = data.dropna()

    # Select the first 50 items during testing
    if limited:
        data = data[:50]

    # Re-map the text column with cleaned text
    data["text"] = data["text"].map(clean_text)

    # Drop the `id` column
    data = data.drop("id", axis=1)
    data = data.reset_index(drop=True)

    # Drop duplicates
    data = data.drop_duplicates(subset=["text"])

    if limited:
        print(f"Cleaned items:\n{data}\n")

    print(data.info())

    # Export data to CSV
    file_name, _ = splitext(file_name)
    file_name = f"{file_name}_clean.csv"
    data.to_csv(file_name, index=False)


if __name__ == "__main__":
    cli()