import pandas as pd
import numpy as np
from scipy.io import arff

class NeuralNet:
  def __init__(self, dataset: str, hidden_layers: int, neuronsPerLayer: int):
    self.dataset = dataset
    self.data = None
    self.hidden_layers = hidden_layers
    self.neuronsLen = neuronsPerLayer
    self.train = None
    self.test = None
    self.train_X = None
    self.train_Y = None
    self.test_X = None
    self.test_Y = None
    self.input_weights = None
    self.hidden_weights = None
    self.hidden_input_weights = None
    self.output_weights = None
    self.output_weights = None
    self.actual_value = 'localization_site'
    self.input_neurons = 2
    self.output_neurons = 1

  def loadDataset(self):
    # data = arff.loadarff(self.dataset)
    self.data = pd.read_csv(self.dataset)
  
  def splitData(self):
    self.train = self.data.sample(frac=0.8)
    self.test = self.data.drop(self.train.index)
  
  def splitTrainData(self):
    self.train_X = self.train.drop(columns=[self.actual_value], axis=1)
    self.train_X.insert(0, 'constant', 1)
    self.train_Y = np.array(self.train[self.actual_value]).reshape(1, self.train_X.shape[0])

  def initializeWeights(self):
    if self.hidden_layers:
      self.input_weights = np.random.rand(self.input_neurons, self.train_X.shape[1])
      self.hidden_input_weights = np.random.rand(self.neuronsLen, self.input_neurons)
      if self.hidden_layers > 1:
        self.hidden_weights = np.random.random((self.hidden_layers-1, self.neuronsLen, self.neuronsLen))
      self.output_weights = np.random.rand(self.output_neurons, self.neuronsLen)
      # print("input weights shape is ", self.input_weights.shape)
      # print("hidden input weights shape is ", self.hidden_input_weights.shape)
      # if self.hidden_weights.shape:
      #   print("hidden weights shape is ", self.hidden_weights.shape)
      # print("output weights shape is ", self.output_weights.shape)

  
  def sigmoid(self, net):
    return 1/(1 + np.exp(-net))

  """
  method to determine mean square error
  """
  def MSE(self, actual, pred):
    return ((actual-pred)**2).sum()/2

  def train_data(self, epochs: int, lr: float):
    train_trans = self.train_X.T
    inputLayerNets = np.zeros((self.input_neurons, self.train_X.shape[0]))
    hiddenNets = np.zeros((self.hidden_layers, self.neuronsLen, self.train_X.shape[0]))
    # print("input layer nets shape is ", inputLayerNets.shape)
    # print("hidden Nets shape is ", hiddenNets.shape)
    pred = None
    for epoch in range(epochs):
      # forward pass
      # input layer
      inputLayerNets = self.sigmoid(self.input_weights.dot(train_trans))
      # hidden input weights
      hiddenNets[0] = self.sigmoid(self.hidden_input_weights.dot(inputLayerNets))
      # all the hidden layers
      for layer in range(1, self.hidden_layers):
        hiddenNets[layer] = self.sigmoid(self.hidden_weights[layer-1].dot(hiddenNets[layer-1]))
      # output layer   
      pred = self.sigmoid(self.output_weights.dot(hiddenNets[-1]))

      # backward pass
      # output layer
      output_delta = pred*(1-pred)*(self.train_Y-pred)
      self.output_weights += lr*(output_delta.dot(hiddenNets[-1].T))
      # hidden layers
      # last hidden layer
      hidden_last_delta = (hiddenNets[-1]*(1 - hiddenNets[-1])*(self.output_weights.T*output_delta))
      self.hidden_weights[-1] += lr*(hidden_last_delta.dot(hiddenNets[-1].T))
      # remaining hidden layers
      for layer in range(self.hidden_layers-3, -1, -1):
        hidden_last_delta = (hiddenNets[layer+1]*(1 - hiddenNets[layer+1])*((self.hidden_weights[layer+1].T).dot(hidden_last_delta)))
        self.hidden_weights[layer] += lr*(hidden_last_delta.dot(hiddenNets[layer].T))
      # hidden input layer
      hidden_last_delta = (hiddenNets[0]*(1 - hiddenNets[0])*((self.hidden_weights[0].T).dot(hidden_last_delta)))
      self.hidden_input_weights += lr*(hidden_last_delta.dot(inputLayerNets.T))

      delta = (inputLayerNets*(1 - inputLayerNets)*((self.hidden_input_weights.T).dot(hidden_last_delta)))
      self.input_weights += lr*(delta.dot(self.train_X))
      print("epoch ", epoch)
      loss = self.MSE(self.train_Y, pred)
      print("loss is ", loss)
    print(pred[0], self.train_Y[0])
      

model = NeuralNet('./yeast/yeast.csv', 5, 3)
# model.loadDataset()
# model.splitData()
# model.splitTrainData()
# model.initializeWeights()
# model.train_data(1000, 0.05)