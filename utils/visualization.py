import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(predict, classes=2,
                           x_range=[0, 1], y_range=[0, 1], th=0.5):

    # Generate a grid with 200 points in each dimension
    hx = (x_range[1]-x_range[0])/200
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 200),
                         np.linspace(y_range[0], y_range[1], 200))

    # Compute decision for meshgrid
    Z = predict(np.hstack((xx.reshape([-1, 1]), yy.reshape([-1, 1]))))

    if classes < 3:
        Z = Z > th
        # Plot the contour and training examples
        Z = np.reshape(Z, xx.shape)
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='jet', alpha=0.6)

    else:
        Z = np.argmax(Z, axis=0)
        # Plot the contour and training examples
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z,
                     levels=np.arange(classes*2)/2,
                     cmap=plt.cm.Spectral,
                     alpha=0.6)

    return plt.gca()
