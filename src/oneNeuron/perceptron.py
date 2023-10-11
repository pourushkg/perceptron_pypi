import numpy as np
import logging   
from tqdm import tqdm  


class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3)*1e-4  # small weight unit
    logging.info(f"Initial weight before training :\n{self.weights}")
    self.eta=eta, # Learning rate
    self.epochs=epochs

  def activationFunction(self,inputs, weights):
    z=np.dot(inputs,weights) # z=w1*x1 + w2*x2 + w3*x3
    return np.where( z > 0, 1, 0)

  def fit(self,x,y):
    self.x=x
    self.y=y
    x_with_bias= np.c_[self.x, -np.ones((len(self.x),1))] ## for concatination
    logging.info(f"x with bias: \n{x_with_bias}")

    for epoch in tqdm(range(self.epochs), total=self.epochs,desc="training the model"):
      logging.info("--"*10)
      logging.info(f"for epoch: {epoch}")
      logging.info("--"*10)

      y_hat = self.activationFunction(x_with_bias,self.weights) ## forward propagation
      logging.info(f"predicted value after forward pass:\n{y_hat}")
      self.error = self.y -y_hat
      logging.info(f"error = \n{self.error}")
      self.weights = self.weights + self.eta*np.dot(x_with_bias.T,self.error)
      logging.info(f"updated weights after epoch: {epoch}/{self.epochs} = {self.weights}") ## backward propagation
      logging.info("##"*10)

  def predict(self,x):
    x_with_bias=np.c_[x,-np.ones((len(x),1))]
    return self.activationFunction(x_with_bias,self.weights)


  def total_loss(self):
    total_loss = np.sum(self.error)
    logging.info(f"total loss: {total_loss}")
    return total_loss
