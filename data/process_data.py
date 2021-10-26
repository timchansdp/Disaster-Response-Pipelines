import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to :
        - Loads data from csv files into dataframes and merge them afterwards
    
    Args:
        messages_filepath (str): File path of messages
        categories_filepath (str): File pathe of categories
    
    Returns:
        pandas dataframe: Merged dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
        
    return pd.merge(messages, categories, how = 'inner', on = ['id'])


def clean_data(df):
    """
    Function to :
        - Clean the data loaded as merged pandas dataframe
    
    Args:
        df (pandas dataframe): Dataframe containing messages and categories
    
    Returns:
        pandas dataframe: Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split( ";", expand = True )
    
    # extract a list of new column names for categories.
    category_colnames = categories.iloc[0].apply( lambda x: x[:-2] ).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop( columns = ['categories'], axis = 1 )
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat( [df, categories], axis = 1 )
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Function to :
        - Save cleaned data as sqlite database
    
    Args:
        df (pandas dataframe): Dataframe containing messages and categories
        database_filename (str): Database name
    
    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    
    database_name = database_filename.replace(".db","")
    df.to_sql(database_name, engine, index = False)


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