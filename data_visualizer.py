import numpy as np
import matplotlib.pyplot as plt


def main():
    xlim = 15
    samples = 15

    valid_data = np.loadtxt(f'valid_data_{samples}x{samples}.csv', delimiter=',')
    invalid_data = np.loadtxt(f'invalid_data_{samples}x{samples}.csv', delimiter=',')

    # plot
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], c = valid_data[:, 2])

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('u')

    plt.subplot(1, 2, 2)
    plt.scatter(invalid_data[:, 0], invalid_data[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    # Add box constraint on x plot
    plt.plot([-xlim, -xlim], [-xlim, xlim], color="red")
    plt.plot([xlim, xlim], [-xlim, xlim], color="red")
    plt.plot([-xlim, xlim], [-xlim, -xlim], color="red")
    plt.plot([-xlim, xlim], [xlim, xlim], color="red", label="Constraints")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

