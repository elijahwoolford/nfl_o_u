import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential

from model_pipeline import plot_sequential_data


def deep_learning_model(X_train, y_train, X_test, y_test):
    model = Sequential()

    n_cols = X_train.shape[1]
    y_train_vals = np.where(y_train == "Over", 1, 0)
    y_test_vals = np.where(y_test == "Over", 1, 0)

    model = Sequential()
    model.add(Dense(32, input_dim=775))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train_vals, epochs=3, validation_split=0.2)

    plot_sequential_data(history)

    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test_vals)
    print('test loss, test acc:', results)

    return model


def plot_learning_curve(model, title, X_train, y_train, axes, ylim, cv):
    n_jobs = 4
    train_sizes = np.linspace(.1, 1.0, 5)
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X_train, y_train, cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
