import numpy as np
import matplotlib.pylab as plt


def main():
    P = np.arange(-4, 4.1, 0.1)
    T = np.power(P, 2) + 1 * (np.random.rand(*P.shape) - 0.5)

    # siec
    S1 = 5
    W1 = np.random.rand(S1, 1) - 0.5
    B1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    B2 = np.random.rand(1, 1) - 0.5
    lr = 0.001

    for epoka in range(1, 200):
        # odpowiedz sieci
        X = W1 * P + B1 * np.ones(P.shape)
        A1 = np.fmax(X, 0)  # ReLu
        A2 = W2 @ A1 + B2

        # propagacja wsteczna
        E2 = T - A2
        E1 = np.transpose(W2) * E2

        dW2 = lr * E2 @ np.transpose(A1)
        dB2 = lr * E2 @ np.transpose(np.ones_like(E2))
        dW1 = lr * (np.divide(np.exp(X), np.exp(X) + 1)) * E1 * np.transpose(P)
        dB1 = lr * (np.divide(np.exp(X), np.exp(X) + 1)) * E1 * np.transpose(np.ones_like(P))

        W2 = W2 + dW2
        B2 = B2 + dB2
        W1 = W1 + dW1
        B1 = B1 + dB1

        plt.ion()
        if np.mod(epoka, 10) == 0:
            plt.clf()
            plt.plot(P, T, 'r*')
            plt.plot(P, *A2, 'b-')
            plt.pause(0.0001)
            plt.show()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
