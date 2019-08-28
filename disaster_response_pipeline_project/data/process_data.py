import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the two given csv files to a pandas DataFrame object by joining them on their id column.

    :param messages_filepath: string - path to the csv file containing the messages
    :param categories_filepath: string - path to the csv file containing the categories
    :return: pandas DataFrame created from the two files
    '''

    # Load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the Datasets and return them
    return pd.merge(messages, categories, how='inner', on='id')


def clean_data(df):
    '''
    Cleans the given DataFrame by splitting the categorie column and transforming it into numeric values and removes duplicates.

    :param df: DataFrame - a DataFrame containing the message and the categorie data
    :return: DataFrame - the cleaned DataFrame
    '''

    # split the categories
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True).values)

    # change column names in the category file
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # converting strings with values to only values
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])

        # some values are > 1 which makes no sense therefore the are reduced to 1
        categories[column] = categories[column].apply(lambda x: x if x < 2 else 1)

    # drop the original categories column
    df = df.drop('categories', 1)

    # concat the new columns with the DataFrame
    df = pd.concat((df, categories), 1)

    # drop the duplicates
    return df.drop_duplicates()


def save_data(df, database_filename):
    '''
    Saves the given DataFrame to a database file named the given name as a table named "BASE"

    :param df: DataFrame - Data to be saved
    :param database_filename: string - Name for the file
    :return: none
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('BASE', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()