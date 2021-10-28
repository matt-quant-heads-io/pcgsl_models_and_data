from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from tensorflow import keras

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import pickle

"""
Next Steps:
===========
1) Create keras callback for saving best model during training

3) Retrain model w/ one hot obs
4) Run inference.py on 40 randomly generated maps
5) Create numpy archive dataset for stable_baselines .pretrain()
6) Compare inference results from .pretrain() trained model with the pure keras MLP model.

"""

# df = pd.read_csv('narrow_greedy_obs_size_5_ep_len_30.csv')

# file_to_read = open("narrow_td_full.pickle", "rb")

# td_dict = pickle.load(file_to_read)
# df = pd.DataFrame(td_dict)

# df_files = ["narrow_td_onehot_obs_5_goals_lg_part0.csv", "narrow_td_onehot_obs_5_goals_lg_part1.csv", "narrow_td_onehot_obs_5_goals_lg_part2.csv",
#  "narrow_td_onehot_obs_5_goals_lg_part3.csv","narrow_td_onehot_obs_5_goals_lg_part4.csv"]


obs_size = 22
ep_len = 77
goal_map_size = 50
print(f"obs_size: {obs_size}, ep_len: {ep_len}, goal_map_size: {goal_map_size}")
dfs = []




# for file in df_files:
for file in os.list(f"exp_traj_obs_{obs_size}_ep_len_{ep_len}_goal_size_{goal_map_size}"):
    dfs.append(pd.read_csv(file))
# #
# # print(f"here2")
df = pd.concat(dfs) #pd.read_csv('narrow_td_onehot_obs_1_goal_lg.csv')
# df = pd.read_csv('narrow_td_onehot_obs_50_goals_25_starts.csv')
print(f"df shape: rows: {df.shape[0]} cols: {df.shape[1]}")
print(f"{df.head()}")

# commend this out to not balance data
# df = df[df['target'] > 1].append(df[df['target'] <= 1].iloc[:9000, :])

df = df.sample(frac=1).reset_index(drop=True)
# print(f"{df.head()}")
# print(f"df length {len(df)}")
y_true = df[['target']]
y_true = np_utils.to_categorical(y_true)
df.drop('target', axis=1, inplace=True)
# TODO: uncomment this if you are reading in df from .csv (if using pickle then make sure this is commented b/c column 1 is index
# X = df.iloc[:, 1:]
X = df.iloc[:, :]

train_split = 1.0 #0.9
train_idx = int(len(X) * train_split)
# y_train = y_true.iloc[:train_idx].values.astype('int32')
# y_test = y_true.iloc[train_idx:].values.astype('int32')
y_train = y_true[:train_idx].astype('int32')
y_test = y_true[train_idx:].astype('int32')

X_train = X.iloc[:train_idx, :].values.astype('int32')
X_test = X.iloc[train_idx:, :].values.astype('int32')

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's an MLP (DDMLP)
model = Sequential()
model.add(Dense(4096, input_dim=input_dim))
model.add(Activation('relu'))
# model.add(Dropout(0.15))
model.add(Dense(4096))
model.add(Activation('relu'))
# model.add(Dropout(0.15))
model.add(Dense(8))
model.add(Activation('softmax'))

"""
Experiments (obs_size_ep_len_iteration)
===========
exp 1 trajectory length: longer are better (talk about increasing traj over time during training); rerun these with equal training data size;
exp 2 we have obs size (make all training lengths same size); have 
exp 3 number goal maps (same training data size): 1 goal map; 5 goal map; 25 goal map; 50 goal map; show the effect of number of playable numbers, unique playable levels, etc.  
      training data size based on number of steps
      Methods: baselines for comparisons --> compare to pcgrl framework; can use;  k-nearest neighbors as model (run on big data set)
exp 4 take best config and run on other games to show it works on other games
      with binary and sokoban (should we add mario?) can start with goal map set of size 5
      
Me:      
a) run the experiments and get the results; 
b) write the methods section


Future
======
- apply this to wider range 
- and submit to HCAI

      


bseline models
==============
1) 3 conv layers and then 1 fully connected (DENSE) LAYER (64 by 64 by 128 check from the atari paper)

2) k-nearest neighbors (pick k that get decent results) - use hamming distance for results

3) maybe add random??


Provw stability of training
===========================
train each obs model for 3 times to prove stability (record the novelty stats)

"""

# model = keras.models.load_model('/Users/matt/gym_pcgrl/gym-pcgil/gym_pcgil/narrow_best_max_acc2.h5')
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(), metrics=[tf.keras.metrics.CategoricalAccuracy()])


# min. loss
# earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='min', restore_best_weights=True)
# mcp_save = ModelCheckpoint('narrow_best_min_loss.h5', save_best_only=True, monitor='val_loss', mode='min')

# max. acc
# earlyStopping = EarlyStopping(monitor='accuracy', patience=500, verbose=0, mode='max', restore_best_weights=True)
# mcp_save = ModelCheckpoint('narrow_best_max_acc2.h5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')

# TODO: narrow_best_max_acc_5_goals_500_starts_reg_fit_incomplete.h5 was stopped at 91% acc
mcp_save = ModelCheckpoint(f'narrow_greedy_obs_size_{obs_size}_ep_len_{ep_len}_goal_map_size_{goal_map_size}.h5', save_best_only=True, monitor='categorical_accuracy', mode='max')
# mcp_save = ModelCheckpoint('narrow_best_max_acc.h5', save_best_only=True, monitor='val_loss', mode='min')


# model.fit(X_train, y_train, epochs=500, batch_size=1, validation_split=0.25, callbacks=[earlyStopping, mcp_save],
#           verbose=2)
model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.0, callbacks=[mcp_save], verbose=2)

# print("Training...")
# for i in range(3):
#     model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.01, verbose=2)
#
# # print("Generating test predictions...")
# # preds = model.predict_classes(X_test, verbose=0)
#
# df = pd.read_csv('narrow_td_int_obs.csv')
# df = df.sample(frac=1).reset_index(drop=True)
# df = df[df['target'] <= 1]
# y_true = df[['target']]
# y_true = np_utils.to_categorical(y_true)
# df.drop('target', axis=1, inplace=True)
# X = df.iloc[:, 1:]
#
# train_split = 0.9
# train_idx = int(len(X) * train_split)
# # y_train = y_true.iloc[:train_idx].values.astype('int32')
# # y_test = y_true.iloc[train_idx:].values.astype('int32')
# y_train = y_true[:train_idx].astype('int32')
# y_test = y_true[train_idx:].astype('int32')
#
# X_train = X.iloc[:train_idx, :].values.astype('int32')
# X_test = X.iloc[train_idx:, :].values.astype('int32')
#
#
# model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.01, verbose=2)
#
#
# #
# for idx in range(len(X_test)):
#     data = X_test[idx]
#     print(f"data is {data}")
#     # print(f"prediction is {model.predict_classes(np.array([data]), verbose=0)}")
#     print(f"prediction is {np.argmax(model.predict(np.array([data]), verbose=0)[0])}")


# model.save('narrow1_balanced2.h5')

