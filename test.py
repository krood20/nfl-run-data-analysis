from kaggle.competitions import nflrush
import pandas as pd

def preprocess():
    #normalize

    #flip so all going in same direction

    #get top features / we pick top features / use all featrues (using all three)

    return

def train_my_model(train_df):
    ## AVIs MODEL --> ADABOOST ##

    ## BILLYs MODEL -->  NEURAL BOI ##

    ## KYLEs MODEL --> RANDY FOREST ##

    ## SURAJs MODEL --> LINEAR ##

    #each one of us will return a model
    return

def make_my_predictions(test_df, sample_prediction_df):
    #
    return

env = nflrush.make_env()

# Training data is in the competition dataset as usual
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_my_model(train_df)

for (test_df, sample_prediction_df) in env.iter_test():
  predictions_df = make_my_predictions(test_df, sample_prediction_df)
  env.predict(predictions_df)

env.write_submission_file()
