import matplotlib.pyplot as plt
import reglog as regresja
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
    multi_classifier.write_probability(x_test)

    plot_decision_regions(X=x_test, y=y_test, classifier=multi_classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


class MultiClassifier(object):
    def __init__(self, train_data, train_etykiety):
        self.regressions = []

        for i in range(0, len(set(train_etykiety))):
            etykiety = train_etykiety.copy()
            etykiety[(etykiety == i)] = -1
            etykiety[(etykiety != -1)] = 0
            etykiety[(etykiety == -1)] = 1

            regression = regresja.LogisticRegressionGD(eta=0.005, n_iter=500, random_state=1)
            regression.fit(train_data, etykiety)
            self.regressions.append(regression)

    def predict(self, x):
        prep_arr = []
        for vector in x:
            temp = []
            for regression in self.regressions:
                temp.append(regression.net_input(vector))
            class_index = temp.index(max(temp))
            prep_arr.append(class_index)
        return np.array(prep_arr)

    def write_probability(self, x):
        prep_arr = []
        for vector in x:
            temp = []
            for regression in self.regressions:
                temp.append(regression.activation(regression.net_input(vector)))
            print(temp)
        return np.array(prep_arr)


if __name__ == '__main__':
    main()
