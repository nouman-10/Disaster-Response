import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    :param messages_filepath:
        path for messages.csv file
    :param categories_filepath:
        path for categories.csv file
    :return:
        dataframe - merged messages and categories file
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages_df, categories_df, on='id')
    return merged_df


def clean_data(df):
    '''
    :param df:
        dataframe containing messages.csv and categories.csv data
    :return:
        dataframe after cleaning - spliting the categories into different columns
    '''

    categories = df['categories'].str.split(';', expand=True)
    first_row = categories.iloc[0,:]
    category_colnames = list(map(lambda x: x[:-2], first_row))
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].astype(float) > 0

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)

    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.dropna()

    return df

def save_data(df, database_filename):
    '''
    :param df:
        dataframe to be saved
    :param database_filename:
        name of the destination database file
    :return:
        nothing - saves the database file to system
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('categories', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()