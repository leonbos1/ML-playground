# this file will use linear regression to predict housing prices
# the features are the following: surface_area, number_of_rooms, energy_label, price
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def energy_label_to_int(energy_label: str):
    energy_label = energy_label.upper().strip()
    if energy_label == 'A++++':
        return 1
    if energy_label == 'A+++':
        return 2
    if energy_label == 'A++':
        return 3
    if energy_label == 'A+':
        return 4
    if energy_label == 'A':
        return 5
    elif energy_label == 'B':
        return 6
    elif energy_label == 'C':
        return 7
    elif energy_label == 'D':
        return 8
    elif energy_label == 'E':
        return 9
    elif energy_label == 'F':
        return 10
    elif energy_label == 'G':
        return 11
    else:
        return 0


def main():
    dataset = pd.read_csv('./data.csv')

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    for i in range(len(X)):
        X[i][2] = energy_label_to_int(X[i][2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = LinearRegression()

    reg.fit(X_train, y_train)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    jet = plt.get_cmap('jet')

    t1 = np.linspace(0, 300, 100)
    t2 = np.linspace(0, 10, 100)
    t1, t2 = np.meshgrid(t1, t2)

    zs = np.array([reg.predict(np.array([[i, j, 0]]))[0]
                  for i, j in zip(np.ravel(t1), np.ravel(t2))])

    Z = zs.reshape(t1.shape)

    ax.plot_surface(t1, t2, Z, cmap=jet, alpha=0.6)

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', s=50)

    ax.set_xlabel('Surface area')

    ax.set_ylabel('Number of rooms')

    ax.set_zlabel('Price')

    plt.show()


if __name__ == '__main__':
    main()
