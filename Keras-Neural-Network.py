import os
import glob
import csv
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


data_path = '/Users/johakim/Desktop/EPIC-Lab-Project/Feature-Extracted'
os.chdir(data_path)
data_files = glob.glob('*.csv')

data_df_collection = []
for file in data_files:
    data_df = pd.read_csv(file)
    data_df_collection.append(data_df)

# Define per-fold score containers
num_folds = len(data_df_collection)
loss_per_fold = []

for fold in range(num_folds):
    # choose one file for testing and rest for training
    test_set = data_df_collection[fold]
    train_set = pd.concat([y for i, y in enumerate(data_df_collection) if i != fold], ignore_index=True)

    test_X = test_set.iloc[:, :-2]
    test_y = test_set['Gait Percent']

    train_X = train_set.iloc[:, :-2]
    train_y = train_set['Gait Percent']

    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=(train_X.shape[1],)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model
    print('-'*80)
    print(f'Training for fold {fold}...')
    epochs = 10
    history = model.fit(train_X, train_y, validation_split=0.2, epochs=epochs, verbose=0)

    # generate generalization metrics
    loss = model.evaluate(test_X, test_y, verbose=0)
    print('Test loss:', loss)
    loss_per_fold.append(loss)

# generate average scores
print('-'*80)
print('Average loss for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')


#
# # read in data using pandas
# data = pd.read_csv('Feature-Extracted/OA08_03.csv')
# X = data.iloc[:, :-2]
# y = data['Gait Percent']
#
# # check correct data has been read in
# print("Input headers: {}\n".format(list(X.columns.values)))
#
# # split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # create the Sequential model that allows buildup layer by layer
# num_nodes = 30
# num_layers = 3
# model = Sequential()
# model.add(Dense(num_nodes, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(num_nodes, activation='relu'))
# model.add(Dense(num_nodes, activation='relu'))
# model.add(Dense(1))
#
# # compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # fit the model
# # set early stopping monitor so the model stops training when it won't improve anymore
# # validation split determines how much of training data is set aside to test model performance
# # epochs is number of times model cycles through data
# epochs = 100
# early_stopping_monitor = EarlyStopping(patience=5)
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=0, callbacks=[early_stopping_monitor])
#
# # summarize history for training loss and validation split loss per epoch
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# plt.plot(loss_values,'bo',label='training loss')
# plt.plot(val_loss_values,'r',label='val training loss')
# plt.ylabel('Loss (MSE)')
# plt.xlabel('Epoch')
# plt.title('Learning Curve')
# plt.legend(['Training Loss', 'Validation Training Loss'], loc='upper left')
# plt.show()
#
# # predict gait phase
# y_train_predict = model.predict(X_train)
# y_test_predict = model.predict(X_test)
# train_score = r2_score(y_train, y_train_predict)
# test_score = r2_score(y_test, y_test_predict)
#
# # print scores
# print("Num layers: {}, Num nodes: {}, Epochs: {}".format(num_layers, num_nodes, epochs))
# print("R2 score on training set: {:0.3f}".format(train_score))
# print("R2 score on test set: {:0.3f}".format(test_score))
#
# # save results of neural net models
# fname = 'Keras-Neural-Network-Results.csv'
# new_row = [num_layers, num_nodes, epochs, str(round(train_score, 3)), str(round(test_score, 3))]
# if os.path.isfile(fname):
#     with open(fname, 'a') as fd:
#         writer = csv.writer(fd)
#         writer.writerow(new_row)
# else:
#     results = [[num_layers, num_nodes, epochs, train_score, test_score]]
#     results_df = pd.DataFrame(results, columns=['Num Layers', 'Num Nodes', 'Epochs', 'Train Score', 'Test Score'])
#     results_df.to_csv(fname, index=False)
