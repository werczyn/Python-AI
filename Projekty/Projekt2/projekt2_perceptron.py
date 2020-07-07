import matplotlib.pyplot as plt
import perceptron as per
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    multi_classifier = MultiClassifier(x_train, y_train)

    plot_decision_regions(X=x_test, y=y_test, classifier=multi_classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


class MultiClassifier(object):
    def __init__(self, train_data, train_label):
        self.perceptrons = []

        for i in range(0, len(set(train_label))):
            etykiety = np.copy(train_label)
            etykiety[(etykiety != i)] = -1
            etykiety[(etykiety == i)] = 1

            perceptron = per.Perceptron(eta=0.1, n_iter=500)
            perceptron.fit(train_data, etykiety)
            self.perceptrons.append(perceptron)

    def predict(self, x):
        prep_arr = []
        for vector in x:
            temp = []
            for perceptron in self.perceptrons:
                temp.append(perceptron.net_input(vector))
            class_index = temp.index(max(temp))
            prep_arr.append(class_index)
        return np.array(prep_arr)


if __name__ == '__main__':
    main()
