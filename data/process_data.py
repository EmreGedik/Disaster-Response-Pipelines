import sys
import pandas as pd
from sqlalchemy import create_engine
from models.train_classifier import tokenize
from collections import Counter

def load_data(messages_filepath, categories_filepath):
    
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge two datasets
    df = messages.merge(categories, how='outer', on=['id'])

    return df



def clean_data(df):
        
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = pd.Series(row).apply(lambda x: x.split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # In 'related' column, there are rows with "2" values. Here, I remove them.
    df = df[(df['related'].isin([0, 1]))]
    
    # reset index of the dataframe
    df.reset_index(inplace = True, drop = True)
        
    return df



def vocabulary_saver(df, database_filename):
    # take the messages from dataframe
    X = df['message']
    
    # create the vocabulary using tokenize
    vocabulary = []
    for i in range(len(X)):
        vocabulary += tokenize(X[i])
    
    # count the number of occurances for each word in the vocabulary
    # and sort the from the most common to least common
    vcb = Counter(vocabulary)
    words_and_occurances = vcb.most_common()
    
    # create a dataframe called vocabulary_df to store the vocabulary and
    # number of occurances
    word = []
    for i in range(len(words_and_occurances)):
        word.append(words_and_occurances[i][0])
    
    occurance = []
    for i in range(len(words_and_occurances)):
        occurance.append(words_and_occurances[i][1])
    
    data = {'word':  word, 'occurance': occurance}
    vocabulary_df = pd.DataFrame (data, columns = ['word','occurance'])
    
    # create a database and store the dataframe in a database table
    engine = create_engine('sqlite:///'+database_filename)
    vocabulary_df.to_sql('Vocabulary', engine, index=False)

    pass  



def save_data(df, database_filename):
    
    # create a database and store the dataframe in a database table
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessagesAndLabels', engine, index=False)

    pass  



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

        print('Saving vocabulary...\n    DATABASE: {}'.format(database_filepath))
        vocabulary_saver(df, database_filepath)        
        
        print('Cleaned data and vocabulary saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()