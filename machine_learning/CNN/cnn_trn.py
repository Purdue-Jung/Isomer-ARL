import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from PIL import Image

import time
import pickle
import random
import math

import matplotlib.pyplot as plt

batch_size = 16
mat_len = 64
tot_epochs = 100
max_node = 64
twelve = 12
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_show = 37
item_show = 12
drop = 0.2

class ToulouseRoadNetworkDataset(Dataset):
  r"""
  Generates a subclass of the PyTorch torch.utils.data.Dataset class
  """

  def __init__(self, root_path="dataset/", split="valid",
            max_prev_node=4, step=0.001, use_raw_images=False, return_coordinates=False):
    r"""

    :param root_path: root dataset path
    :param split: dataset split in {"train", "valid", "test", "augment"}
    :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
        default is 4, which corresponds to the 95th percentile in the dataset
    :param step: step size used in the dataset generation, default is 0.001° (around 110 metres per datapoint)
    :param use_raw_images: loads raw images if yes, otherwise faster and more compact numpy array representations
    :param return_coordinates: returns coordinates on the real map for each datapoint, used for qualitative studies
    """
    root_path += str(step)
    assert split in {"train", "valid", "test", "augment"}
    print(f"Started loading the dataset ({split})...")
    start_time = time.time()

    dataset_path = f"{root_path}/{split}.pickle"
    images_path = f"{root_path}/{split}_images.pickle"
    images_raw_path = f"{root_path}/{split}/images/"

    ids, (x_adj, x_coord, y_adj, y_coord, seq_len, map_coordinates) = load_dataset(dataset_path, max_prev_node,
                                                                                return_coordinates)

    self.return_coordinates = return_coordinates
    self.ids = ["{:0>7d}".format(int(i)) for i in ids]
    self.x_adj = x_adj
    self.x_coord = x_coord
    self.y_adj = y_adj
    self.y_coord = y_coord
    self.seq_len = seq_len
    self.map_coordinates = map_coordinates

    print(f"Started loading the images...")

    if use_raw_images:
        self.images = load_raw_images(ids, images_raw_path)
    else:
        self.images = load_images(ids, images_path)

    print(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds!")
    print(f"Dataset size: {len(self)}\n")

  def __len__(self):
    r"""
    :return: dataset length
    """
    return len(self.seq_len)

  def __getitem__(self, idx):
    r"""
    :param idx: index in the dataset
    :return: chosen data point
    """
    if self.return_coordinates:
      return self.x_adj[idx], self.x_coord[idx], self.y_adj[idx], self.y_coord[idx], self.images[idx], \
            self.seq_len[idx], self.ids[idx], self.map_coordinates[idx]
    return self.x_adj[idx], self.x_coord[idx], self.y_adj[idx], self.y_coord[idx], self.images[idx], \
        self.seq_len[idx], self.ids[idx]

def load_dataset(dataset_path, max_prev_node, return_coordinates):
  r"""
  Loads the chosen split of the dataset

  :param dataset_path: path of the dataset split pickle
  :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
  :param return_coordinates: returns coordinates on the real map for each datapoint
  :return:
  """
  with open(dataset_path, "rb") as pickled_file:
    dataset = pickle.load(pickled_file)

  list_x_adj = []
  list_x_coord = []
  list_y_adj = []
  list_y_coord = []
  list_seq_len = []
  list_original_xy = []
  ids = list(dataset.keys())
  random.Random(42).shuffle(ids)  # permute to remove any correlation between consecutive datapoints

  for id in ids:
    datapoint = dataset[id]

    x_adj = torch.FloatTensor(datapoint["bfs_nodes_adj"])
    x_coord = torch.FloatTensor(datapoint["bfs_nodes_points"])

    # add termination token (zero-vector) to model the termination of a connected component AND the whole graph
    x_adj = torch.cat([x_adj, torch.zeros_like(x_adj[0]).unsqueeze(0)])
    x_coord = torch.cat([x_coord, torch.zeros_like(x_coord[0]).unsqueeze(0)])
    # add 2nd termination token (zero-vector) to model the termination of a connected component AND the whole graph
    y_adj = torch.cat([x_adj[1:, :].clone(), torch.zeros_like(x_adj[0]).unsqueeze(0)])
    y_coord = torch.cat([x_coord[1:, :].clone(), torch.zeros_like(x_coord[0]).unsqueeze(0)])
    # slice up to max_prev_node length
    x_adj = x_adj[:, :max_prev_node]
    y_adj = y_adj[:, :max_prev_node]

    list_x_adj.append(x_adj)
    list_x_coord.append(x_coord)
    list_y_adj.append(y_adj)
    list_y_coord.append(y_coord)
    list_seq_len.append(len(x_adj))  # Seq len is computed here, after creating the actual sequence
    list_original_xy.append(datapoint["coordinates"])
  list_seq_len = torch.LongTensor(list_seq_len)

  if return_coordinates:
    return ids, (list_x_adj, list_x_coord, list_y_adj, list_y_coord, list_seq_len, list_original_xy)
  return ids, (list_x_adj, list_x_coord, list_y_adj, list_y_coord, list_seq_len, None)


def load_images(ids, images_path):
  r"""
  Load images from arrays in pickle files

  :param ids: ids of the images in the dataset order
  :param images_path: path of the pickle file
  :return: the images, as pytorch tensors
  """
  images = []
  with open(images_path, "rb") as pickled_file:
    images_features = pickle.load(pickled_file)
  for id in ids:
    img = torch.FloatTensor(images_features["{:0>7d}".format(int(id))])
    assert img.shape[1] == img.shape[2]
    assert img.shape[1] in {64}
    images.append(img)

  return images


def load_raw_images(ids, images_path):
  r"""
  Load images from raw files

  :param ids: ids of the images in the dataset order
  :param images_path: path of the raw images
  :return: the images, as pytorch tensors
  """
  images = []
  for count, id in enumerate(ids):
    # if count % 10000 == 0:
    #     print(count)
    image_path = images_path + "{:0>7d}".format(int(id)) + ".png"
    img = Image.open(image_path).convert('L')
    img = tvf.to_tensor(img)
    assert img.shape[1] == img.shape[2]
    assert img.shape[1] in {64, 128}
    images.append(img)
  return images


def custom_collate_fn(batch):
  r"""
  Custom collate function ordering the element in a batch by descending length

  :param batch: batch from pytorch dataloader
  :return: the ordered batch
  """
  x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = zip(*batch)

  x_adj = pad_sequence(x_adj, batch_first=True, padding_value=0)
  x_coord = pad_sequence(x_coord, batch_first=True, padding_value=0)
  y_adj = pad_sequence(y_adj, batch_first=True, padding_value=0)
  y_coord = pad_sequence(y_coord, batch_first=True, padding_value=0)
  img, seq_len = torch.stack(img), torch.stack(seq_len)

  seq_len, perm_index = seq_len.sort(0, descending=True)
  x_adj = x_adj[perm_index]
  x_coord = x_coord[perm_index]
  y_adj = y_adj[perm_index]
  y_coord = y_coord[perm_index]
  img = img[perm_index]
  ids = [ids[perm_index[i]] for i in range(perm_index.shape[0])]

  return x_adj, x_coord, y_adj, y_coord, img, seq_len, ids


def custom_collate_fn_with_coordinates(batch):
  r"""
  Custom collate function ordering the element in a batch by descending length

  :param batch: batch from pytorch dataloader
  :return: the ordered batch
  """
  x_adj, x_coord, y_adj, y_coord, img, seq_len, ids, original_xy = zip(*batch)

  x_adj = pad_sequence(x_adj, batch_first=True, padding_value=0)
  x_coord = pad_sequence(x_coord, batch_first=True, padding_value=0)
  y_adj = pad_sequence(y_adj, batch_first=True, padding_value=0)
  y_coord = pad_sequence(y_coord, batch_first=True, padding_value=0)
  img, seq_len = torch.stack(img), torch.stack(seq_len)

  seq_len, perm_index = seq_len.sort(0, descending=True)
  x_adj = x_adj[perm_index]
  x_coord = x_coord[perm_index]
  y_adj = y_adj[perm_index]
  y_coord = y_coord[perm_index]
  img = img[perm_index]
  original_xy = original_xy[perm_index]
  ids = [ids[perm_index[i]] for i in range(perm_index.shape[0])]

  return x_adj, x_coord, y_adj, y_coord, img, seq_len, ids, original_xy


def normalize(x, normalization=True):
  r"""
  Image normalization in [-1,+1]

  :param x: input tensor
  :param normalization: if False, return the input
  :return:
  """
  if normalization:
    x = (x * 2) - 1
  return x


def denormalize(x, normalization=True):
  r"""
  Image denormalization, converting back to [0,+1]

  :param x: input tensor
  :param normalization: if False, return the input
  :return:
  """
  if normalization:
    x = (x + 1) / 2
  return x

class CNNAutoEncoder(nn.Module):
  def __init__(self, kernel_size):
    super().__init__()
    self.conv1 = nn.Conv2d(1,16,kernel_size,padding='same')
    self.act1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(16,32,kernel_size,padding='same')
    self.act2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(32,32,kernel_size,padding='same')
    self.act3 = nn.ReLU()
    self.up1 = nn.Upsample(scale_factor=2)
    self.conv4 = nn.Conv2d(32,16,kernel_size,padding='same')
    self.act4 = nn.ReLU()
    self.up2 = nn.Upsample(scale_factor=2)
    self.conv5 = nn.Conv2d(16,1,kernel_size,padding='same')
    self.flatten = nn.Flatten()
    self.act5 = nn.ReLU()
    self.dropout = nn.Dropout(drop)
    self.lin = nn.Linear(mat_len*mat_len, twelve*max_node)
    self.sig = nn.Sigmoid()
    self.ret = nn.Unflatten(1, (max_node, twelve))

  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.act2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.act3(x)
    x = self.up1(x)
    x = self.conv4(x)
    x = self.act4(x)
    x = self.up2(x)
    x = self.conv5(x)
    x = self.flatten(x)
    x = self.act5(x)
    x = self.dropout(x)
    x = self.lin(x)
    x = self.sig(x)
    x = self.ret(x)
    return x

d = ToulouseRoadNetworkDataset(split="train", step=0.001, max_prev_node=4, use_raw_images=False)
dataloader = DataLoader(d, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

pr = True
start_time = time.time()
for datapoint in dataloader:
  this_x_adj, this_x_coord, this_y_adj, this_y_coord, this_img, this_seq_len, this_id = datapoint
  if pr:
    print(this_x_adj[0])
    print(this_x_coord[0])
    print(this_y_adj[0])
    print(this_y_coord[0])
    print(this_img[0])
    print(this_seq_len[0])
    print(this_id[0])
    pr = False
print(f"Iteration over the dataset completed, took {round(time.time() - start_time, 2)}s!")

def train_one_epoch(model, lossfn, optim, loader, epoch, val_loader):
  start = time.time()
  cur_loss = 0
  i = 0
  model.train()
  for datapoint in loader:
    x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = datapoint
    cur_sz = x_adj.shape[0]
    label = torch.zeros(cur_sz, max_node, twelve)
    label[:, :x_adj.shape[1], :4] = x_adj
    label[:, :x_adj.shape[1], 4:6] = (x_coord + 1)/2
    label[:, :x_adj.shape[1], 6:10] = y_adj
    label[:, :x_adj.shape[1], 10:12] = (y_coord + 1)/2
    label = label.to(device)
    img = img.to(device) # Move the image tensor to the device
    optim.zero_grad()
    pred = model(img)
    loss = lossfn(pred, label)
    loss.backward()
    optim.step()
    cur_loss += loss.item()
    i += 1
  val_loss = 0
  model.eval()
  for datapoint in val_loader:
    x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = datapoint
    label = torch.zeros(batch_size, max_node, twelve)
    cur_sz = x_adj.shape[0]
    label = torch.zeros(cur_sz, max_node, twelve)
    label[:, :x_adj.shape[1], :4] = x_adj
    label[:, :x_adj.shape[1], 4:6] = (x_coord + 1)/2
    label[:, :x_adj.shape[1], 6:10] = y_adj
    label[:, :x_adj.shape[1], 10:12] = (y_coord + 1)/2
    label = label.to(device)
    img = img.to(device) # Move the image tensor to the device
    pred = model(img)
    loss = lossfn(pred, label)
    val_loss += loss.item()
  print(f"Epoch {epoch} train loss: {cur_loss / len(loader)} val loss: {val_loss / len(val_loader)} elapsed time: {round(time.time() - start,2)}")
  return cur_loss / len(loader), val_loss / len(val_loader)

cnn = CNNAutoEncoder(3).to(device)
lossfn = nn.MSELoss()
optim = torch.optim.Adam(cnn.parameters(), lr=0.001)
val = ToulouseRoadNetworkDataset(split="valid", step=0.001, max_prev_node=4, use_raw_images=False)
valload = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
train_loss = []
val_loss = []
for epoch in range(tot_epochs):
  tl, vl = train_one_epoch(cnn, lossfn, optim, dataloader, epoch+1, valload)
  train_loss.append(tl)
  val_loss.append(vl)
  if epoch % 20 == 19:
    torch.save(cnn.state_dict(), f"cnn_epoch_kern3_{epoch+1}.pt")

x = np.linspace(1, tot_epochs, tot_epochs)

plt.figure(figsize=(10, 5),layout='constrained')
plt.plot(x,train_loss,label='train_loss',color='blue')
plt.plot(x,val_loss,label='val_loss',color='green')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.title('Training and Validation MSE Losses of Basic CNN Encoder Model')
plt.legend()