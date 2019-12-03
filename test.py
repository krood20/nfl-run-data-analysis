import time
import re
import math

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

#Preprocess our
def preprocess(df):
	# originally has data for every player during every play, in the intrest of simplicity use only info about the rusher
	df = df.loc[df['NflId'] == df['NflIdRusher']]
	df = df.loc[df['Yards'] < 5]
	df = df.loc[df['Yards'] >= 0]

	df = df.dropna()
	df = df.reset_index(drop=True)

	# convert player height to inches
	df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x : int(re.search("[^-]+", x).group()) * 12 + int(re.search("[^-]*$", x).group()))

	# fix inconsistent team abreviations between possesion team and home/away team
	df.loc[df['PossessionTeam'] == 'BLT', 'PossessionTeam'] = 'BAL'
	df.loc[df['PossessionTeam'] == 'CLV', 'PossessionTeam'] = 'CLE'
	df.loc[df['PossessionTeam'] == 'ARZ', 'PossessionTeam'] = 'ARI'
	df.loc[df['PossessionTeam'] == 'HST', 'PossessionTeam'] = 'HOU'

	# change yardline to be distance to the goal
	df.loc[df['PossessionTeam'] == df['FieldPosition'], 'YardLine'] = 100 - df['YardLine']

	# convert categorical variable to numbers
	columns = df.columns
	for col in columns:
		if df[col].dtype == 'object':
			df[col] = pd.factorize(df[col])[0]


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

    #split our processed data into train and test
	processed_df = df[top_features]
	X = processed_df
	y = processed_df.pop('Yards')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

	#put em back together
	train = pd.concat([X_train, y_train.reindex(X_train.index)], axis=1)
	test = pd.concat([X_test, y_test.reindex(X_test.index)], axis=1)

	return (train, test)


## BILLY ##
def neural_net(X_train, y_train, X_test, y_test):
    model = build_neural_net_model(X_train)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # prevents overfitting
    history = model.fit(X_train, y_train, epochs=100, validation_split = 0.2, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    print()

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    # plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    # plotter.plot({'Basic': history}, metric = "mae")
    # plt.ylim([0, 10])
    # plt.ylabel('MAE')
    # plt.show()

    print(model.evaluate(X_test, y_test, verbose=2))

    test_predictions = model.predict(X_test).flatten()

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, test_predictions)
    # plt.title("NN")
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # lims = [0, 50]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.plot(lims, lims)
    # # plt.show()
	#
    # error = test_predictions - y_test
    # plt.hist(error, bins = 25)
    # plt.title("NN")
    # plt.xlabel("Prediction Error")
    # plt.ylabel("Count")
    # # plt.show()
    return model


## KYLE ##
def random_forest(X_train, y_train, X_test, y_test):
	#set up our Forest
    print("Fitting...")
    rf = RandomForestRegressor(n_estimators = 100)
    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()
    print("Time to fit: " + str(end-start))

    #get features importances
    # importance_list = rf.feature_importances_
    #
    # most_important = rf.feature_importances_
    # most_important.sort()
    # most_important = most_important[-2:]
    # print("Most important features")
    # print(most_important)
    # print(importance_list.index(most_important[0]), importance_list.index(most_important[1]))

    #predictions
    test_predictions = rf.predict(X_test)

    #testing a bit
    # a = plt.axes(aspect='equal')
    # plt.title("Random Forest")
    # plt.scatter(y_test, test_predictions)
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # lims = [0, 50]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.plot(lims, lims)
    # # plt.show()
	#
    # error = test_predictions - y_test
    # plt.hist(error, bins = 25)
    # plt.title("Random Forest")
    # plt.xlabel("Prediction Error")
    # plt.ylabel("Count")
    # plt.show()

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


## SURAJ ##
def linear_regression(X_train, y_train, X_test, y_test):
    print("Fitting...")
    linreg = LinearRegression().fit(X_train, y_train)
    test_predictions = linreg.predict(X_test).flatten()
    print(test_predictions)

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, test_predictions)
    # plt.title("LinReg")
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
    # plt.title("LinReg")
    # plt.xlabel("Prediction Error")
    # plt.ylabel("Count")
    # plt.show()

    return linreg


def train_my_model(train_df):

	## Train test split
	X = train_df
	y = train_df.pop('Yards')
	X_columns = X.columns
	X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X_columns) # normalize X


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)

	## AVIs MODEL --> ADABOOST ##
	adaboost_model = adaboost(X_train, y_train)

	## BILLYs MODEL -->  NEURAL BOI ##
	neural_net_model = neural_net(X_train, y_train, X_test, y_test)

	## KYLEs MODEL --> RANDY FOREST ##
	forest_model = random_forest(X_train, y_train, X_test, y_test)

	## SURAJs MODEL --> LINEAR ##
	linreg_model = linear_regression(X_train, y_train, X_test, y_test)

	#each one of us will return a model
	return (linreg_model, forest_model, adaboost_model, neural_net_model)


def test_model(df, linreg_model, forest_model, adaboost_model, neural_net_model):
	correct_labels = np.array(df.pop('Yards'))

	columns = df.columns
	df = pd.DataFrame(StandardScaler().fit_transform(df), columns=columns) # normalize X

	# Linear regression
	linreg_preds = linreg_model.predict(df).flatten()

	# Random forest
	randy_preds = forest_model.predict(df).flatten()

	# Adaboost
	ada_preds = adaboost_model.predict(df).flatten()

	# Neural Net Predictions
	neural_net_preds = neural_net_model.predict(df).flatten()


	# Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth. —Abraham Lincoln

	# Stacc the data
	staccs = np.stack((linreg_preds, randy_preds, ada_preds, neural_net_preds), axis = 1)
	avg = np.mean(staccs, axis=1)

    # get the error of each prediction
	print('resutls')
	print(avg)
	print('correct')
	print(correct_labels)
	print('error')
	errors = abs(avg - correct_labels)
	print(errors)
	print('average error')
	print(np.mean(errors))

	plt.title("Accuracy vs. Test")
	plt.xlabel("test")
	plt.ylabel("accuracy")
	plt.plot(range(len(errors)), errors)
	plt.show()

	return




# Training data is in the competition dataset as usual
df = pd.read_csv('../nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df, test_df = preprocess(df)

print(train_df)

linreg_model, forest_model, adaboost_model, neural_net_model = train_my_model(train_df)

test_model(test_df, linreg_model, forest_model, adaboost_model, neural_net_model)
