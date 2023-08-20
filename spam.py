import re
import string

import pandas as pd
import spacy as spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

# Loading Spacy english model
en_sm_model = spacy.load('en_core_web_sm')


def prepare(text: str) -> str:
    """
    Prepare text to be tokenized later

    :param text: A string to be modified
    :return: The input string in lower case with numeric data changed to
    'aanumbers', without punctuation or stopwords or one letter words
    """

    # Tokenization
    doc = en_sm_model(text.lower())
    prepared_text = ''

    # Preparing the string to be returned
    for word in doc:
        str_word = ''.join([c for c in word.lemma_ if c not in string.punctuation])
        if re.match(r'.*?\d+.*', str_word):
            prepared_text += ' aanumbers'
        elif str_word in STOP_WORDS or len(str_word) == 1:
            continue
        else:
            prepared_text += ' ' + str_word

    return prepared_text


def bag_of_words(train_set: pd.Series) -> pd.DataFrame:
    """
    Convert a series of text documents to a matrix of token counts.

    :param train_set: An iterable which generates either str, unicode or file objects.
    :return: A pandas Dataframe with the document-term matrix.
    """

    # Instantiating sklearn.feature_extraction.text.CountVectorizer
    vectorizer = CountVectorizer()

    # Learn the vocabulary dictionary and return document-term matrix.
    X = vectorizer.fit_transform(train_set)

    return pd.DataFrame(X.toarray(), index=train_set.index, columns=vectorizer.get_feature_names_out())


def naive_bayes(train_set: pd.DataFrame, bow: pd.DataFrame, alpha=1):
    """
    Trains the model using naive Bayes

    :param train_set: The dataframe of the train set
    :param bow: The dataframe of the bag of words
    :param alpha: The smoothing value between 0 and 1
    :return: A tuple consisting of the dataframe with the words probabilities and the probabilities of the two labels
    """

    bow_columns = bow.columns
    n_vocab = len(bow_columns)
    spam = train_set.loc[train_set['Target'] == 'spam', bow_columns].sum()
    ham = train_set.loc[train_set['Target'] == 'ham', bow_columns].sum()
    spam_sum = spam.sum()
    ham_sum = ham.sum()
    p_word_spam = (spam + alpha) / (spam_sum + alpha * n_vocab)
    p_word_ham = (ham + alpha) / (ham_sum + alpha * n_vocab)
    df = pd.DataFrame({'Spam Probability': p_word_spam, 'Ham Probability': p_word_ham}, index=bow_columns)
    spam_ham_sum = spam_sum + ham_sum
    p_spam_ = spam_sum / spam_ham_sum
    p_ham_ = ham_sum / spam_ham_sum
    return df, p_spam_, p_ham_


def predict(words_probs: pd.DataFrame, sms: str, p_spam_: float, p_ham_: float) -> str:
    """
    Predicts if a text is ham or spam

    :param words_probs: The dataframe with the words probabilities
    :param sms: The text to do the prediction
    :param p_spam_: Probability of spam
    :param p_ham_: Probability of ham
    :return: A string of the prediction (spam or ham)
    """
    p_sms_spam = p_spam_
    p_sms_ham = p_ham_
    for word in prepare(sms).split():
        if word not in words_probs.index:
            continue
        p_sms_spam *= words_probs.loc[word, 'Spam Probability']
        p_sms_ham *= words_probs.loc[word, 'Ham Probability']
    return 'spam' if p_sms_spam > p_sms_ham else 'ham'


def metrics(dataframe: pd.DataFrame) -> dict:
    """
    Generates the metrics of the predictions (accuracy, precision, recall and f1 score)

    :param dataframe: Pandas dataframe with both predicted an actual values for spam and ham
    :return: A dictionary with the metrics
    """

    tn = dataframe.loc[(dataframe['Predicted'] == 'ham') & (dataframe['Actual'] == 'ham')].count()[0]
    fp = dataframe.loc[(dataframe['Predicted'] == 'spam') & (dataframe['Actual'] == 'ham')].count()[0]
    fn = dataframe.loc[(dataframe['Predicted'] == 'ham') & (dataframe['Actual'] == 'spam')].count()[0]
    tp = dataframe.loc[(dataframe['Predicted'] == 'spam') & (dataframe['Actual'] == 'spam')].count()[0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}
