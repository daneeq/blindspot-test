import matplotlib.pyplot as plt
import numpy as np
import csv


# Load a CSV file
def load_csv(filename):
  dataset = list()
  with open(filename, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
      if not row:
        continue
      dataset.append(row)
  return dataset


def main():


  # load data
  filename = 'data/linsep-traindata.csv'
  dataset = load_csv(filename)
  X = np.array(dataset,dtype=np.float64)


  filename = 'data/linsep-trainclass.csv'
  dataset = load_csv(filename)
  Y = np.array([int(x[0]) for x in dataset])
  # print Y.squeeze()

  plt.figure()
  plt.scatter(X[:, 0], X[:, 1], marker="o", s=50, linewidths=0, c=Y.squeeze(), cmap=plt.cm.coolwarm)
  

  weights = np.array([0.0 for i in range(len(X[0])+1)]) # improve
  learning_rate = 0.01
  epochs = 2


  for epoch in range(epochs):
    for j in range(len(X)):
      row = np.insert(X[j],0,1) # add 1 to match weights
      output = Y[j]
      
      activation = weights.dot(row)
      if activation >= 0.0:
        prediction = 1
      else:
        prediction = -1

      error = output - prediction
      print row
      print output, prediction, error

      weights = weights + learning_rate * error * row 

  print weights

  Y = (-weights[0] - (weights[1] * X)) / weights[2]
  plt.plot(X, Y, "r-")
  plt.show()


if __name__ == '__main__':
  main()