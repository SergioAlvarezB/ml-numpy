# Machine Learning in Numpy

Collection of Numpy implementations of some traditional ML algorithms. Visual
examples are provided in the form of jupyter notebooks.


I came up with the idea for this project when I started learning machine learning,
from-scratch implementations helps getting a good intuition about how and why
algorithms work. As a result this repository is my personal repertoire of such
implementations.

## Structure

Algorithms implementations are under the [models](./models/) directory.
Each model is defined in its own class. These implementations are not meant to
be optimal, but clear and easy to read.

To test whether models are working correctly some usage examples are drawn in
jupyter notebooks under the [notebooks](./notebooks/) folder.

Lastly some helper functions are arranged into some modules under the [utils](./utils/)
directory. This methods encapsulate some code for trivial tasks like, loading a simple dataset or plotting a decision boundary, but that would result in an innecessary amount of code making notebooks harder to follow.
