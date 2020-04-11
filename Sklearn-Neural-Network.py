import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPRegressor


def main():
    # /Users/johakim/Desktop/EPIC-Lab-Project/Feature-Extracted/OA08_03.csv
    # read feature extracted data with labels
    data = pd.read_csv(sys.argv[1])
    x = data.iloc[:, :-2]
    y = data['Gait Percent']

    # perform neural network training, fitting, testing
    neural_network(x, y)


def neural_network(x, y):
    # pick best values for parameters hidden_layer_sizes, activation, solver
    num_neurons = [5, 10, 20, 30]
    num_layers = [1, 2, 3]
    hl_permutations = []
    hl_scores = []

    for layers in num_layers:
        hl_permutations.extend(list(permutations(num_neurons, layers)))

    nn_df = pd.DataFrame(hl_permutations)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # train neural net with all of the generated hidden_layer_sizes
    for hls in hl_permutations:
        model = MLPRegressor(hidden_layer_sizes=hls, activation='relu', solver='adam')
        model.fit(x_train, y_train)
        test_score = model.score(x_test, y_test)
        hl_scores.append(test_score)

    # get scores of neural networks of different hidden layer sizes
    nn_df.insert(0, 'test_score', hl_scores)
    print("NN hidden layer sizes and scores:\n", nn_df)

    # get the hidden layer size of best score
    best_hidden_layer_size = hl_permutations[hl_scores.index(max(hl_scores))]
    print("Best hidden layer size:", best_hidden_layer_size)
    print("Score:", max(hl_scores))

    # plot learning curve
    model = MLPRegressor(hidden_layer_sizes=best_hidden_layer_size, activation='relu', solver='adam')
    train_sizes = np.linspace(.1, 1.0, 10)
    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    train_sizes, train_scores, validation_scores = learning_curve(estimator=model, X=x_train, y=y_train, train_sizes=train_sizes, scoring=scoring[0], cv=5)
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning Curves for MLP Model', fontsize=18, y=1.03)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()