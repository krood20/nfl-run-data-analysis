import time

# from kaggle.competitions import nflrush
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# pip install -q git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# Make sure to download the data from Kaggle before running:
# https://www.kaggle.com/c/nfl-big-data-bowl-2020/data


def build_neural_net_model(dataset):
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mae', 'mse'])
  return model


def preprocess(df):

	# originally has data for every player during every play, in the intrest of simplicity use only info about the rusher
	df = df.loc[df['NflId'] == df['NflIdRusher']]
	df = df.dropna()

	columns = df.columns
	for col in columns:
		if df[col].dtype == 'object':
			df[col] = pd.factorize(df[col])[0]

	# normalize

	# flip so all going in same direction


	######### feature selection V1, top 10 based on F-value between label/feature for regression tasks #########
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

## BILLY ##
def neural_net(X_train, y_train, X_test, y_test):
	model = build_neural_net_model(X_train)
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # prevents overfitting
	history = model.fit(X_train, y_train, epochs=100, validation_split = 0.2, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
	print()

	# hist = pd.DataFrame(history.history)
	# hist['epoch'] = history.epoch
	# plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
	# plotter.plot({'Basic': history}, metric = "mae")
	# plt.ylim([0, 10])
	# plt.ylabel('MAE')
	# plt.show()

	# print(model.evaluate(X_test, y_test, verbose=2))
	#
	# test_predictions = model.predict(X_test).flatten()
	#
	# a = plt.axes(aspect='equal')
	# plt.scatter(y_test, test_predictions)
	# plt.xlabel('True Values')
	# plt.ylabel('Predictions')
	# lims = [0, 50]
	# plt.xlim(lims)
	# plt.ylim(lims)
	# plt.plot(lims, lims)
	# plt.show()
	#
	# error = test_predictions - y_test
	# plt.hist(error, bins = 25)
	# plt.xlabel("Prediction Error")
	# plt.ylabel("Count")
	# plt.show()
	return model

## KYLE ##
def random_forest(X_train, y_train):
    #set up our Forest
    print("Fitting...")
    rf = RandomForestRegressor(n_estimators = 100)
    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()
    print("Time to fit: " + str(end-start))
    return rf

## AVI ##
def adaboost(X_train, y_train):
    #Set up the adaboost model
    print("Fitting...")
    abc = AdaBoostRegressor(n_estimators = 100, learning_rate = 1.0)
    start = time.time()
    abc.fit(X_train, y_train)
    end = time.time()
    print("Time to fit: " + str(end-start))
    return abc

def train_my_model(train_df):

	# cross validation split
	y = train_df.pop('Yards')
	X = train_df
	X_train, X_test, y_train, y_test = train_test_split(X, y)


	## AVIs MODEL --> ADABOOST ##
	adaboost_model = adaboost(X_train, y_train)

	## BILLYs MODEL -->  NEURAL BOI ##
	neural_net_model = neural_net(X_train, y_train, X_test, y_test)

	## KYLEs MODEL --> RANDY FOREST ##
	forest_model = random_forest(X_train, y_train)

	## SURAJs MODEL --> LINEAR ##
	X_train = StandardScaler().fit_transform(X_train)
	linreg = LinearRegression().fit(X_train, y_train)

	###### ----------------------------------- ##########

	#each one of us will return a model
	return (linreg, forest_model, adaboost_model, neural_net_model)

def make_my_predictions(test_df, sample_prediction_df):
	#
	return



# env = nflrush.make_env()

# Training data is in the competition dataset as usual
df = pd.read_csv('../nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df = preprocess(df)

linreg, forest_model, adaboost_model, neural_net_model = train_my_model(train_df)


# for (test_df, sample_prediction_df) in env.iter_test():
#   predictions_df = make_my_predictions(test_df, sample_prediction_df)
#   env.predict(predictions_df)
#
# env.write_submission_file()
