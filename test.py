# from kaggle.competitions import nflrush
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def preprocess(df):

    # originally has data for every player during every play, in the intrest of simplicity use only info about the rusher
    df = df.loc[df['NflId'] == df['NflIdRusher']]

    #normalize

    #flip so all going in same direction


    ######### feature selection V1, top 10 based on F-value between label/feature for regression tasks #########
    # columns = df.columns
    # for col in columns:
    #     if df[col].dtype == 'object':
    #         df[col] = pd.factorize(df[col])[0]
    # df = df.dropna()
    #
    # x = list(df.columns)
    # x.remove('Yards')
    # x.remove('GameId')
    # x.remove('PlayId')
    # X = df[x]
    # y = df['Yards']
    # bestfeatures = SelectKBest(score_func=f_regression, k=10).fit(X,y)
    # scores = pd.DataFrame(bestfeatures.scores_)
    # columns = pd.DataFrame(X.columns)
    # featureScores = pd.concat([columns,scores],axis=1)
    # featureScores.columns = ['Specs','Score']
    # print(featureScores.nlargest(10,'Score'))
    # top_features = list(featureScores.nlargest(10,'Score')['Specs'])
    # top_features.append('Yards')


    ##################################### feature selection V2, intuition #####################################
    top_features = ['S', 'A', 'Orientation', 'Dir', 'NflId', 'YardLine', 'Down',
        'Distance', 'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox',
         'DefensePersonnel', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'Yards']

    ######################################## feature selection V3, all ########################################
    # top_features = list(df.columns)
    # top_features.remove('GameId')
    # top_features.remove('PlayId')

    return df[top_features]


def train_my_model(train_df):

    # cross validation split

    ## AVIs MODEL --> ADABOOST ##

    ## BILLYs MODEL -->  NEURAL BOI ##

    ## KYLEs MODEL --> RANDY FOREST ##

    ## SURAJs MODEL --> LINEAR ##

    #each one of us will return a model
    return

def make_my_predictions(test_df, sample_prediction_df):
    #
    return



# env = nflrush.make_env()

# Training data is in the competition dataset as usual
train_df = pd.read_csv('../nfl-big-data-bowl-2020/train.csv', low_memory=False)

preprocess(train_df)

train_my_model(train_df)
#
# for (test_df, sample_prediction_df) in env.iter_test():
#   predictions_df = make_my_predictions(test_df, sample_prediction_df)
#   env.predict(predictions_df)
#
# env.write_submission_file()
