import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import csv

class Activation(Enum):
  SIGMOID = 'sigmoid'
  TANH = 'tanh'
  RELU = 'relu'

class NeuralNet:
  def __init__(self, dataset: str, hidden_layers: int, neuronsPerLayer: int, activation: Activation='sigmoid'):
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
    self.train_Y_act = None
    self.input_weights = None
    self.hidden_weights = None
    self.hidden_input_weights = None
    self.output_weights = None
    self.output_weights = None
    self.actual_value = 'class'
    self.input_neurons = 4
    self.output_neurons = 1
    self.displayPlots = False
    self.scaled_data = None
    self.feature_stats = {}
    self.activation: Activation = activation
    self.log = []

  def loadDataset(self):
    # data = arff.loadarff(self.dataset)
    self.data = pd.read_csv(self.dataset)
    self.preprocess_data(0.8)
  
  """
  method to remove the duplicate values in the dataset
  """
  def filter_duplicates(self):
    duplicates = self.data.duplicated()
    if len(duplicates):
      print("found duplicates\n")
      print(self.data[duplicates], '\n')
      print('removing duplicate rows \n')
      self.data = self.data.drop_duplicates()
      print('duplicate rows removed\n')
    else:
      print("no duplicates found")

  """
  method to check the correlation between the independent features to check and remove columns which are highly correlated.
  """
  def feature_selection(self, threshold):
    correlation_matrix = (self.data.drop(columns=[self.actual_value])).corr()
    print("correlation matrix within the features:\n")
    print(correlation_matrix)
    print("\n")
    high_corr_pairs = [(self.data.columns[i], self.data.columns[j]) for i in range(len(correlation_matrix.columns)) 
                   for j in range(i+1, len(correlation_matrix.columns)) 
                   if abs(correlation_matrix.iloc[i, j]) > threshold]
    print("column pairs with high correlation:\n")
    print(high_corr_pairs)
    print("\n")
    columns_to_drop = set()
    columns_remain = set()
    for i in high_corr_pairs:
      corr0 = self.data[i[0]].corr(self.data[self.actual_value])
      corr1 = self.data[i[1]].corr(self.data[self.actual_value])
      print(f'correlation of {i[0]} with {self.actual_value} is {corr0}\n')
      print(f'correlation of {i[1]} with {self.actual_value} is {corr1}\n')
      if abs(corr0) > abs(corr1):
        if i[0] not in columns_remain:
          columns_remain.add(i[0])
        if i[1] not in columns_to_drop:
          columns_to_drop.add(i[1])
      else:
        if i[1] not in columns_remain:
          columns_remain.add(i[1])
        if i[0] not in columns_to_drop:
          columns_to_drop.add(i[0])
    print('columns to remove \n')
    print(columns_to_drop, '\n')
    print('columns to remain\n')
    print(columns_remain, '\n\n')
    print(f'removing features: {",".join(list(columns_to_drop))}\n')
    self.data = self.data.drop(columns=list(columns_to_drop))
    self.features_len = len(self.data.columns)
    if self.displayPlots:
      plt.figure(figsize=(10, 5))
      sns.heatmap(correlation_matrix, vmin=-1, cmap='coolwarm', annot=True)
      plt.show()

  """
  method to normalize the features i.e, transform them into same scale
  """
  def normalize_features(self):
    # using min max scaling
    self.scaled_data = self.data
    for column in self.data.columns:
      if column != self.actual_value:
        min = self.data[column].min()
        max = self.data[column].max()
        self.scaled_data[column] = (((self.data[column] - min)/(max-min))*(6))-3
        self.feature_stats[column]={
          'min': min,
          'max': max
        }
    print('normalized features')
    return

  """
  method to preprocess the data
  """
  def preprocess_data(self, correlation_threshold):
    # checking for null values
    if self.data.isnull().sum().sum():
      print('Null values are present')
      self.data.dropna()
    else:
      print('No null values are present')

    # filtering features based on the correlation between them
    self.feature_selection(correlation_threshold)

    # normalizing the filtered features
    self.normalize_features()
    

  """
  method to determine the r square score
  """
  def R_squared_stat(self, actual, pred):
    mean_y = actual.mean()
    RSS = ((actual - pred)**2).sum() # residual sum of squares
    TSS = ((actual - mean_y)**2).sum() # total sum of squares
    return 1 - (RSS/TSS)
  
  """
  method to determine root of mean square error
  """
  def RMSE(self, mse):
    return np.sqrt(mse)

  """
  method to write the log data in a csv file
  """
  def logData(self):
    fields = ['epoch', 'trainingLoss', 'testLoss', 'epochs', 'lr', 'training_r2', 'testing_r2', 'training_rmse', 'testing_rmse']
    # fields = fields + ['W'+str(i) for i in range(self.features_len)]
    logFile = open('logfile.csv', 'a')
    # logFile.truncate(0)
    writer = csv.DictWriter(logFile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(self.log)
    logFile.close()

  def splitData(self, frac):
    self.train = self.scaled_data.sample(frac=frac)
    self.test = self.scaled_data.drop(self.train.index)
    # self.train = np.array(self.data[:2])
    # print(self.train.shape) 

  def splitTrainData(self):
    self.train_X = self.train.drop(columns=[self.actual_value], axis=1)
    self.train_X.insert(0, 'constant', 1)
    self.train_Y = np.array(self.train[self.actual_value]).reshape(1, self.train_X.shape[0])
    # self.train_Y = self.train_Y[:,::-1]
    # print(self.train_Y, self.train_Y_act)

  """
  method to split the testing data frame into features and actual truth
  """
  def test_split(self):
    self.test_x = self.test.drop(columns=[self.actual_value], axis=1)
    self.test_x.insert(0, 'constant', 1)
    self.test_Y = np.array(self.test[self.actual_value]).reshape(self.test_x.shape[0], 1)

  def initializeWeights(self):
    if self.hidden_layers:
      # self.input_weights = np.random.rand(self.input_neurons, self.train_X.shape[1])
      self.hidden_input_weights = np.random.rand(self.neuronsLen, self.train_X.shape[1])
      if self.hidden_layers > 1:
        self.hidden_weights = np.random.random((self.hidden_layers-1, self.neuronsLen+1, self.neuronsLen))
      self.output_weights = np.random.rand(self.neuronsLen+1, self.output_neurons)
      # print("input weights shape is ", self.input_weights.shape)
      # print("hidden input weights shape is ", self.hidden_input_weights.shape)
      # if self.hidden_layers > 1:
        # print("hidden weights shape is ", self.hidden_weights.shape)
      # print("output weights shape is ", self.output_weights.shape)

  
  def sigmoid(self, net):
    return 1/(1 + np.exp(-net))

  def tanh(self, net):
    return (np.exp(net) - np.exp(-net))/(np.exp(net) + np.exp(-net))
  
  def applyRelu(self, x):
    return 0 if x < 0 else x

  def relu(self, net):
    reluFn = np.vectorize(self.applyRelu)
    return reluFn(net)

  def derivateRelu(self, x):
    return 0 if x < 0 else 1

  def backpropRelu(self, pred):
    dRelu = np.vectorize(self.derivateRelu)
    return dRelu(pred)

  def activate(self, net):
    if self.activation == Activation.SIGMOID.value:
      return self.sigmoid(net)
    elif self.activation == Activation.TANH.value:
      return self.tanh(net)
    else:
      return self.relu(net)
  
  def backprop(self, pred):
    if self.activation == Activation.SIGMOID.value:
      return (pred*(1 - pred))
    elif self.activation == Activation.TANH.value:
      return (1 - pred**2)
    else:
      return self.backpropRelu(pred)

  """
  method to determine mean square error
  """
  def MSE(self, actual, pred):
    return np.sum((actual-pred)**2)/(2*len(pred))
    
  def train_data(self, epochs: int, lr: float):
    inputLayerNets = self.train_X.T
    hiddenNets = np.zeros((self.hidden_layers, self.neuronsLen+1, self.train_X.shape[0]))
    # print("input layer nets shape is ", inputLayerNets.shape)
    # print("hidden Nets shape is ", hiddenNets.shape)
    pred = None
    for _ in range(epochs):
      # forward pass
      # hidden input weights
      hiddenNets[0] = np.concatenate((self.activate(self.hidden_input_weights.dot(inputLayerNets)), np.ones((1, self.train_X.shape[0]))))
      # all the hidden layers
      for layer in range(1, self.hidden_layers):
        hiddenNets[layer] = np.concatenate((self.activate(self.hidden_weights[layer-1].T.dot(hiddenNets[layer-1])), np.ones((1, self.train_X.shape[0]))))
      # output layer   
      pred = self.activate(self.output_weights.T.dot(hiddenNets[-1]))

      # backward pass
      # output layer
      output_delta = self.backprop(pred)*(self.train_Y[0].T-pred)
      self.output_weights += lr*((hiddenNets[-1].dot(output_delta.T))/self.train_X.shape[0])
      # hidden layers
      # last hidden layer
      if self.hidden_layers > 1:
        hidden_last_delta = (self.backprop(hiddenNets[-1])*(self.output_weights.dot(output_delta)))
        self.hidden_weights[-1] += lr*(hidden_last_delta[:-1, :].dot(hiddenNets[-1].T)).T/self.train_X.shape[0]
        # remaining hidden layers
        for layer in range(self.hidden_layers-3, -1, -1):
          hidden_last_delta = (self.backprop(hiddenNets[layer+1])*((self.hidden_weights[layer+1]).dot(hidden_last_delta[:-1, :])))
          self.hidden_weights[layer] += lr*(hidden_last_delta[:-1, :].dot(hiddenNets[layer].T).T)/self.train_X.shape[0]
      # hidden input layer
        hidden_last_delta = (self.backprop(hiddenNets[0])*((self.hidden_weights[0]).dot(hidden_last_delta[:-1, :])))
        self.hidden_input_weights += lr*(hidden_last_delta[:-1, :].dot(inputLayerNets.T))/self.train_X.shape[0]
      else:
        hidden_last_delta = (self.backprop(hiddenNets[-1])*(self.output_weights.dot(output_delta)))
        self.hidden_input_weights += lr*(hidden_last_delta[:-1, :].dot(inputLayerNets.T))/self.train_X.shape[0]
      # delta = (inputLayerNets*(1 - inputLayerNets)*((self.hidden_input_weights.T).dot(hidden_last_delta)))
      # self.input_weights += lr*(delta.dot(self.train_X))
      # print("epoch ", epoch)
      trainingLoss = self.MSE(self.train_Y, pred)
      testLoss = self.MSE(self.test_Y)
      training_r2 = self.R_squared_stat(self.train_Y, pred)
      testing_r2 = self.R_squared_stat(self.test_Y, self.test_x.dot(self.weights))
      training_rmse = self.RMSE(trainingLoss)
      testing_rmse = self.RMSE(testLoss)
      logData = {
          'epoch': epoch,
          'trainingLoss': trainingLoss.iloc[0],
          'testLoss': testLoss.iloc[0],
          'lr': lr,
          'epochs': epochs,
          'training_r2': training_r2.iloc[0],
          'testing_r2': testing_r2.iloc[0],
          'training_rmse': training_rmse.iloc[0],
          'testing_rmse': testing_rmse.iloc[0]
        }
      self.log.append(logData)
    loss = self.MSE(self.train_Y, pred)
    print("loss is ", loss)
    print(self.train_Y.T[:4], pred.T[:4])
    print(self.R_squared_stat(self.train_Y, pred))

  def fit(self, split, epochs: int, lr: float, logging: bool = False, displayPlots: bool = False, plotFile: str = './part1-logfile.csv'):
    self.loadDataset()
    self.splitData(split)
    self.splitTrainData()
    self.test_split()
    self.initializeWeights()
    self.train_data(epochs, lr)
    if logging:
      self.logData()
    if displayPlots and logging:
      self.plotMetrics(plotFile)

  """
  method to plot the metrics data stored in the log file
  """
  def plotMetrics(self, file: str):
    logDf = pd.read_csv(file)
    # 'MSE plot Training'
    plt.plot(logDf['epoch'], logDf['trainingLoss'])
    plt.title('MSE training data vs epoch')
    plt.show()
    # 'MSE plot Testing'
    plt.plot(logDf['epoch'], logDf['testLoss'])
    plt.title('MSE testing data vs epoch')
    plt.show()
    # 'R2 plot Training'
    plt.plot(logDf['epoch'], logDf['training_r2'])
    plt.title('Training R2 score vs epoch')
    plt.show()
    # 'R2 plot Testing'
    plt.plot(logDf['epoch'], logDf['testing_r2'])
    plt.title('Testing R2 score vs epoch')
    plt.show()
    # 'RMSE plot Training'
    plt.plot(logDf['epoch'], logDf['training_rmse'])
    plt.title('Training RMSE vs epoch')
    plt.show()
    # 'RMSE plot Testing'
    plt.plot(logDf['epoch'], logDf['testing_rmse'])
    plt.title('Testing RMSE vs epoch')
    plt.show()
    # Plots for weights
    for i in range(self.features_len):
      plt.plot(logDf['epoch'], logDf['W'+str(i)])
      plt.title(f'W{i} vs epoch')
      plt.show()   

model = NeuralNet('./dataset/bank_note.csv', 2,3, Activation.RELU.value)
model.fit(0.8, 1, 0.1)