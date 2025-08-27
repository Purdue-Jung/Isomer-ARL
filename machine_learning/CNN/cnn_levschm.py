import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers
import matplotlib.pyplot as plt
import random
import networkx as nx

kern = 3
mat_size = 512
neg_pad = -10
train_split = 0.5
val_split = 0.25
test_split = 0.25
assert train_split + val_split + test_split == 1
epochs = 32
batch_size = 25
learn = 0.001
filename = "simulation_results3r512x_custombg.npz"
drop = 0.2
sz = int(mat_size)

train_samples = 3
val_samples = 1
test_samples = 1

# create_matrix(lev_scheme)
# input:  lev_scheme - level scheme dictionary
# output: negative-padded adjacency matrix (int[][])
# desc: takes the level scheme dictionary and creates an adjacency matrix, where
#       nodes are sorted by ascending energy

def create_matrix(lev_scheme):
  mat = np.full((mat_size, mat_size), neg_pad)
  state_energies = [(0,0)]
  names = {'grnd': 0}
  edges = []
  next_id = 1
  for band in lev_scheme:
    band_states = list(band['nodes'].keys())
    for i in band_states:
      if i == 'grnd':
        continue
      names[i] = next_id
      state_energies.append((next_id, band['nodes'][i]))
      next_id += 1
    for i in band['edges']:
      edges.append((names[i[0]], names[i[1]]))

  state_energies.sort(key = lambda st: st[1])
  id_to_idx = list(range(next_id))
  for i in range(next_id):
    id_to_idx[state_energies[i][0]] = i

  mat[0:next_id, 0:next_id] = 0
  for edge in edges:
    mat[id_to_idx[edge[0]], id_to_idx[edge[1]]] = state_energies[id_to_idx[edge[0]]][1] - state_energies[id_to_idx[edge[1]]][1]

  return mat

# count_nodes(mat)
# input:  mat - adjacency matrix
# output: number of nodes
# desc: takes matrix and counts the number of rows which correspond to nodes

def count_nodes(mat):
  for i in range(mat_size):
    if mat[i][0] == neg_pad:
      return i
  return mat_size

# visualize(mat, cnt)
# input: mat - adjacency matrix
#       cnt - node count
# output: none
# desc: prints visualization with state energies

def visualize(mat, cnt):
  G = nx.DiGraph()
  for i in range(cnt):
    G.add_node(i,energy=0)

  for i in range(cnt):
    if i == 0:
      continue
    max_idx = 0
    for j in range(cnt):
      if mat[i][j] > mat[i][max_idx]:
        max_idx = j
    if mat[i][max_idx] > 0:
      G.add_edge(i,max_idx,weight=mat[i][max_idx])
      G.nodes[i]['energy'] = int(G.nodes[max_idx]['energy'] + mat[i][max_idx])

  plt.subplot(111)
  plt.figure(figsize=(10,10))
  nx.draw_kamada_kawai(G, with_labels=False)
  nx.draw_networkx_labels(G, pos=nx.kamada_kawai_layout(G), labels=nx.get_node_attributes(G, 'energy'))
  # nx.draw_networkx_edge_labels(G, pos=nx.kamada_kawai_layout(G), edge_labels=nx.get_edge_attributes(G, 'weight'))
  plt.show()

# create_splits(ggm, lschm)
# input:  ggm - gamma-gamma matrices in list form
#        lschm - level scheme dicts in list form
# output: training, validation, test splits
# desc: converts level schemes to matrices, splits data as per weights

def create_splits(ggm, lschm):
  tot = len(ggm)
  train_end = int(tot * train_split)
  val_end = int(tot * (train_split + val_split))
  bundled = list(zip(ggm, lschm))
  random.seed(85)
  random.shuffle(bundled)
  train_data = []
  train_label = []
  val_data = []
  val_label = []
  test_data = []
  test_label = []
  for i in range(tot):
    gg = bundled[i][0]
    mat = create_matrix(bundled[i][1])
    if i < train_end:
      train_data.append(gg)
      train_label.append(mat)
    elif i < val_end:
      val_data.append(gg)
      val_label.append(mat)
    else:
      test_data.append(gg)
      test_label.append(mat)
  return np.array(train_data), np.array(val_data), np.array(test_data), np.array(train_label), np.array(val_label), np.array(test_label)

# autoencoder pieces
def build_encoder():
  model = Sequential()
  model.add(layers.Input(shape=(mat_size,mat_size)))
  model.add(layers.Reshape((mat_size,mat_size,1)))
  model.add(layers.Conv2D(16, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(32, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Conv2D(32, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Conv2D(16, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Conv2D(1, (kern,kern), activation='relu', padding='same'))
  model.add(layers.Flatten())
  return model

# model 1: one layer
def build_model1():
  model = build_encoder()
  model.add(layers.Dense(sz,activation='relu'))
  model.add(layers.Dropout(drop))
  model.add(layers.Dense(mat_size*mat_size,activation='linear'))
  model.add(layers.Reshape((mat_size,mat_size)))
  return model

# model 2: more encoder
def build_model2():
  model = build_encoder()
  model.add(layers.Dense(sz,activation='relu'))
  model.add(layers.Dense(mat_size*mat_size,activation='linear'))
  model.add(layers.Reshape((mat_size,mat_size,1)))
  model.add(layers.Conv2D(16, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(32, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (kern,kern), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Conv2D(32, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Conv2D(16, (kern,kern), activation='relu', padding='same'))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.Dropout(drop))
  model.add(layers.Conv2D(1, (kern,kern), activation='linear', padding='same'))
  model.add(layers.Flatten())
  model.add(layers.Reshape((mat_size,mat_size)))
  return model

# generate splits
data = np.load(filename, allow_pickle=True)

matrix_0 = data['matrix_0']

matrix_keys = [k for k in data.files if k.startswith('matrix_')]
matrix_list = [data[k] for k in matrix_keys]

level_scheme_keys = [k for k in data.files if k.startswith('level_scheme_')]
level_scheme_list = [data[k] for k in level_scheme_keys]

print(np.shape(matrix_list))

print(level_scheme_list[0])

train_data, val_data, test_data, train_label, val_label, test_label = create_splits(matrix_list, level_scheme_list)

visualize(train_label[0], count_nodes(train_label[0]))
visualize(val_label[0], count_nodes(val_label[0]))
visualize(test_label[0], count_nodes(test_label[0]))

def train(mod, lossfn=keras.losses.MeanSquaredError(), name="model"):
  mod.compile(optimizer=keras.optimizers.Adam(learning_rate=learn), loss=lossfn)
  mod.summary()
  mod.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(val_data, val_label), shuffle=True)
  mod.evaluate(test_data, test_label, verbose=2)

model1 = build_model1()
train(model1, name="model1")

train_preds = model1.predict(train_data)
val_preds = model1.predict(val_data)
test_preds = model1.predict(test_data)
print("Training samples")
for i in range(train_samples):
  r = random.randint(0,train_data.shape[0]-1)
  cnt = count_nodes(train_label[r])
  visualize(train_label[r], cnt)
  visualize(train_preds[r], cnt)

print("Validation samples")
for i in range(val_samples):
  r = random.randint(0,val_data.shape[0]-1)
  cnt = count_nodes(val_label[r])
  visualize(val_label[r], cnt)
  visualize(val_preds[r], cnt)

print("Test samples")
for i in range(test_samples):
  r = random.randint(0,test_data.shape[0]-1)
  cnt = count_nodes(test_label[r])
  visualize(test_label[r], cnt)
  visualize(test_preds[r], cnt)

x = []
y = []

for i in range(train_data.shape[0]):
  cnt = count_nodes(train_label[i])
  for j in range(cnt):
    for k in range(cnt):
      if train_label[i][j][k] > 0:
        x.append(train_label[i][j][k])
        y.append(train_preds[i][j][k])

x = np.array(x)
y = np.array(y)

m, b = np.polyfit(x,y,1)
y_bf = m*x + b
plt.scatter(x,y)
plt.plot(x,y_bf,color='red')
plt.show()

corr = np.corrcoef(x,y)[0,1]
print(f"Line of best fit: y = {m:.3f}x + {b:.3f}")
print(f"Correlation coefficient: {corr:.3f}")

model2 = build_model2()
train(model2, name="model2")

train_preds = model2.predict(train_data)
val_preds = model2.predict(val_data)
test_preds = model2.predict(test_data)
print("Training samples")
for i in range(train_samples):
  r = random.randint(0,train_data.shape[0]-1)
  cnt = count_nodes(train_label[r])
  visualize(train_label[r], cnt)
  visualize(train_preds[r], cnt)

print("Validation samples")
for i in range(val_samples):
  r = random.randint(0,val_data.shape[0]-1)
  cnt = count_nodes(val_label[r])
  visualize(val_label[r], cnt)
  visualize(val_preds[r], cnt)

print("Test samples")
for i in range(test_samples):
  r = random.randint(0,test_data.shape[0]-1)
  cnt = count_nodes(test_label[r])
  visualize(test_label[r], cnt)
  visualize(test_preds[r], cnt)

x = []
y = []

for i in range(train_data.shape[0]):
  cnt = count_nodes(train_label[i])
  for j in range(cnt):
    for k in range(cnt):
      if train_label[i][j][k] > 0:
        x.append(train_label[i][j][k])
        y.append(train_preds[i][j][k])

x = np.array(x)
y = np.array(y)

m, b = np.polyfit(x,y,1)
y_bf = m*x + b
plt.scatter(x,y)
plt.plot(x,y_bf,color='red')
plt.show()

corr = np.corrcoef(x,y)[0,1]
print(f"Line of best fit: y = {m:.3f}x + {b:.3f}")
print(f"Correlation coefficient: {corr:.3f}")