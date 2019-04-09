import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(predict, classes=2,
                           x_range=[0, 1], y_range=[0, 1], th=0.5):
    """Plots decision boundary on 2D.

    Creates a 2Dplot of size defined by `x_range` and `y_range`, and shadows
    regions according to the predicted class of the function `predict`.

    Parameters
    ----------
    predict : `Callable`
        Function that maps a numpy.nadarrar of shape (-1, 2) to a vector of
        probabilities os size classes if classes>2 or a single probability in
        case of binary classification.

    classes : `int`, optional
        Total number of classes. Defaults to 2, assumes binary classification.

    x_range : `numpy.ndarray` (2,)
        Specify the x-axis limits of the final plot. Defaults to [0, 1].

    y_range : `numpy.ndarray` (2,)
        The same as x_range but with the y-axis.

    th : `int`, optional
        Threshold used to determine whether the outputed probability belongs to
        class 1 in the case of binary classification. If classes>2 `th` is
        ignored. Defaults to 0.5.

    """

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
        if Z.ndim > 1:
            Z = np.argmax(Z, axis=1)
        # Plot the contour and training examples
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z,
                     levels=np.arange(classes*2)/2,
                     cmap='jet',
                     alpha=0.6)

    return plt.gca()
