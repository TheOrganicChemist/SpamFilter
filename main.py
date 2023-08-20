import pandas as pd
from sklearn.model_selection import train_test_split

import spam

"""
Example of implementation for a sms spam filter
Author: Christian Valencia
Year: 2023
.\python.exe -m spacy download 'en_core_web_sm'
"""

if __name__ == '__main__':
    # Random state for data partition
    random_state = 42

    # Reading and preparation of dataset from file with encoding iso-8859-1
    dataset = pd.read_csv(r'C:\Users\Chris\Downloads\spam.csv', encoding='iso-8859-1')
    dataset.rename(columns={'v1': 'Target', 'v2': 'SMS'}, inplace=True)
    dataset['SMS'] = dataset['SMS'].apply(spam.prepare)

    # Partitioning of dataset
    dataset = dataset.sample(frac=1, random_state=random_state)
    train_set, test_set, y_train, y_test = train_test_split(dataset['SMS'], dataset['Target'],
                                                            train_size=0.8,
                                                            shuffle=False)

    # Training model
    train_bow = spam.bag_of_words(train_set)
    embedded_train_df = pd.concat([y_train, train_bow], axis=1)
    probs_df, p_spam, p_ham = spam.naive_bayes(embedded_train_df, train_bow)

    # Setting the size for printing
    pd.options.display.max_columns = probs_df.shape[1]
    pd.options.display.max_rows = probs_df.shape[0]

    # Testing predictions with the test set and printing metrics
    predictions = test_set.apply(lambda sms: spam.predict(probs_df, sms, p_spam, p_ham)).rename('Predicted')
    y_test.rename('Actual', inplace=True)
    print(spam.metrics(pd.concat([predictions, y_test], axis=1)))
