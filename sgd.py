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
  X = np.array(dataset)


  filename = 'data/linsep-trainclass.csv'
  dataset = load_csv(filename)
  Y = np.array([int(x[0]) for x in dataset])
  print Y.squeeze()


  plt.figure()
  plt.scatter(X[:, 0], X[:, 1], marker="o", s=50, linewidths=0, c=Y.squeeze(), cmap=plt.cm.coolwarm)
  # plt.plot(X, Y, "r-")
  plt.show()

if __name__ == '__main__':
  main()