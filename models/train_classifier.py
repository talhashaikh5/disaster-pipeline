import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import fbeta_score, make_scorer
import pickle
from scipy.stats import gmean

def load_data(database_filepath):
    """
    Returns X, Y and categories names

            Parameters:
                    database_filepath (str): path of database file 

            Returns:
                    X (df), Y (df), Columns Names (list)
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    # engine.table_names()
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message']
    y = df.iloc[:,3:]
    columns_names = list(df.columns)[3:]
    return X, y, columns_names


def tokenize(text):
    """
    Returns list of tokens after performing tokenizing actions
            Parameters:
                    text (str): string that needs to be tokenised 
            Returns:
                    list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf_transformer', TfidfTransformer())
                ])),

                ('starting_verb_transformer', StartingVerbExtractor())
            ])),

            ('clf', MultiOutputClassifier(
                RandomForestClassifier(n_estimators=200,min_samples_split=3)
                ))
        ])

    #  From Grid search it is concluded that n_estimators=200,min_samples_split=3
    # parameters = {
    #     'clf__estimator__n_estimators': [200,100,50],
    #     'clf__estimator__min_samples_split': [3,2,1],
    # }

    # cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1,verbose=3) 
    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is a performance metric of my own creation.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
        
                    Parameters:
                        y_true -> List of labels
                        y_prod -> List of predictions
                        beta -> Beta value to be used to calculate fscore metric
                    
                    Return:
                        f1score -> Calculation geometric mean of fscore
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],average='weighted',beta=beta)
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_model(pipeline, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
                    Parameters:
                        pipeline = A valid scikit ML Pipeline
                        X_test = Test features
                        Y_test = Test labels
                        category_names = label names (multi-output)
    """
    Y_pred = pipeline.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta=1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))

# def evaluate_model(model, X_test, Y_test, category_names):
#     y_pred = model.predict(X_test)
#     print(classification_report(Y_test,y_pred,target_names=category_names))   


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()