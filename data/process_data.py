import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Returns the pandas DataFrame after merging 2 data sources (csv)

            Parameters:
                    messages_filepath (str): file path of messages data
                    categories_filepath (str):  file path of categoriess data

            Returns:
                    pandas.core.frame.DataFrame after merging 2 files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """
    Returns a clean DataFrame after applying all the necessary cleaning 
    actions

            Parameters:
                    df (pandas.core.frame.DataFrame): dataframe that needs to 
                    be cleaned

            Returns:
                    pandas.core.frame.DataFrame (cleaned)
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand = True)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = list(categories.iloc[0])
    category_colnames = [col.split("-")[0] for col in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    # remove existing categories column
    df.drop(["categories"],inplace=True,axis=1)
    df.drop(["original"],inplace=True,axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(keep = 'first',inplace=True)
    # # drop unwanted columns 
    # df.drop(["original","id","genre"],inplace=True, axis=1)
    # drop rows with nan values
    df.dropna(inplace=True)
    # # drop child_lone column no message for that
    # df.drop(["child_alone"],inplace=True,axis=1)
    # we have '2' vale in related columns this might be mistake while entering data making it'1'
    print
    return df


def save_data(df, database_filename):
    """
    saves data to sqlite datbase

            Parameters:
                    df (pandas.core.frame.DataFrame): dataframe that needs to 
                    be stored
                    database_filename: 

            Returns:
                    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(f"    Rows:{df.shape[0]}, columns:{df.shape[1]}")
        
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
