import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # standardyzacja cech
    sc = StandardScaler()
    sc.fit(X_train)

    # GINI depth4
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    score = tree.score(X_combined, y_combined)
    print(score)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree_gini_depth4')
    plt.show()


    # export_graphviz(tree, out_file='tree_gini_depth4.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    # ENTROPHY
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    score = tree.score(X_combined, y_combined)
    print(score)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree_entropy')
    plt.show()

    # export_graphviz(tree, out_file='tree_entropy.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    # GINI depth10
    tree = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    score = tree.score(X_combined, y_combined)
    print(score)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('tree_gini_depth10')
    plt.show()

    # export_graphviz(tree, out_file='tree_gini_depth10.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    # FOREST 1
    forest = RandomForestClassifier(criterion='gini', n_estimators=15, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    score = forest.score(X_combined, y_combined)
    print(score)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('randomforest1')
    plt.show()

    # FOREST 2
    forest = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    score = forest.score(X_combined, y_combined)
    print(score)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.savefig('randomforest2')
    plt.show()


if __name__ == '__main__':
    main()
