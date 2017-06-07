import matplotlib.pyplot as plt
import numpy as np
import csv


def load_csv(filename):
    # Load a CSV file
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def prepare_data(file_in, file_out):
    # load input data and output labels, shuffle randomly
    # and return an 80/20 cross-validation split

    # load data from files, set types
    dataset_in = load_csv(file_in)
    X = np.array(dataset_in, dtype=np.float64)

    # add column of leading 1's to balance shape of weights
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)

    dataset_out = load_csv(file_out)
    Y = np.array(dataset_out, dtype=np.int)

    # join input and output
    XY = np.concatenate((X, Y), axis=1)
    # shuffle data points toghether
    np.random.shuffle(XY)

    # split the input into training and validation parts
    split_index = int(0.8 * len(XY))
    (X, Y) = np.split(XY, [3], axis=1)
    train_in = np.split(X, [split_index])
    train_out = np.split(Y, [split_index])

    return (train_in, train_out)


def predict(X, W):

    activation = W.dot(X)
    if activation >= 0.0:
        return 1
    else:
        return -1


def train(X, Y, epochs, learning_rate):

    weights = np.zeros(X.shape[1], dtype=np.float64)    # improve

    for epoch in range(epochs):
        for j in range(len(X)):

            prediction = predict(X[j], weights)

            error = Y[j, 0] - prediction

            # adjust weights based on learning rate
            weights = weights + learning_rate * error * X[j]

    return weights


def validate(X, Y, W):
    # validate the data and compute the error

    errors = []
    for j in range(len(X)):

        error = 1 - int(Y[j, 0] == predict(X[j], W))
        errors.append(error)

    avg_error = sum(errors) / float(len(errors))

    return avg_error


def visualise(X, Y, W):

    # plot the input data and decision boundary
    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], marker="o", s=50,
                linewidths=0, c=Y, cmap=plt.cm.coolwarm)

    Y_line = (-W[0] - (W[1] * X[:, 1:])) / W[2]

    plt.plot(X[:, 1:], Y_line, "r-")
    plt.title('Classification boundary for the training data')
    plt.show()


def main():

    # load the input and output data
    (X, Y) = prepare_data('data/linsep-traindata.csv',
                          'data/linsep-trainclass.csv')
    # separate training and validation parts
    (X_train, X_val) = X
    (Y_train, Y_val) = Y

    learning_rate = 0.01
    epochs = 2

    weights = train(X_train, Y_train, epochs, learning_rate)
    print 'Learned weights: {}'.format(weights)

    avg_error = validate(X_val, Y_val, weights)
    print 'Prediction error: {}%'.format(avg_error * 100)

    visualise(X_train, Y_train, weights)


if __name__ == '__main__':
    main()
