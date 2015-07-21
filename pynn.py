"""pynn.py
~~~~~~~~~~
Python Neural Network
Author: Robert J Compton
http://github.com/robertcompton/pynn

A standard network class for simple supervised neural networks. Currently
supports supervised neural networks. Activation and cost functions, along with
the training method for the network are passed using wrapper classes. So if the
included classes do not fit the intended use case, user defines ones may be
passed instead.

General Notation:
    x: Input
    z: Weighted input (tx+b)
    a: Output (activation), sigma(z)
    y: Desired output for related x in the training set
    Sigma: Activation function, or class wrapper
    Theta: Weights of the network, parametrizing H() and J()
    Nabla: Gradient of a function (e.g, nabla_t is the gradient of J() in terms
    of theta)
    J: The cost function of the network
    H: The hypothesis function of the network
    Prime: A derivative function. E.g, sigma.prime
    Delta: The value of a derivative function. I.e, delta = sigma.prime(z), but
    not sigma.delta

"""

"""Third-party libraries"""
import numpy as np

"""Congruent imports"""
from utils import *

class Network():
    """A general class for neural networks. Is passed the desired structure for
    the network in the form of a list and classes containing the functions
    needed for computing the cost function and describing the activation of the
    network. See ``__init__`` for more details.
    """

    def __init__(self, layers,
                 sigma = SigmoidActivation,
                 cost  = CrossEntropyCost,
                 desc  = SGD):
        """Initializes the local variables for the network. ``layers`` is a list
        describing the structure for the network. Each element represents the
        number of neurons in the i'th layer where i is the index in the
        ``layers`` list. layers[0] is the input payer and layers[-1] is the
        output layer. Input and output layers are included in this list.
        ``sigma`` is a class containing the functions for the desired activation
        function used for the network. See ``utils.py`` for example structure.
        ``cost`` is a class containing the needed functions for the desired cost
        function to be used by the network. ``desc`` is a class describing the
        method for descent to be used by the network. See ``utils.py`` for example
        structures for these classes.
        """
        self.nl     = len(layers)
        self.layers = layers
        self.sigma  = sigma
        self.cost   = cost
        self.desc   = desc

        # Vectorize functions. Note this is not for performance, but instead for
        # syntax, as np.vectorize is just an element wise for loop.
        self.sigmaVec      = np.vectorize(sigma.fn)
        self.sigmaPrimeVec = np.vectorize(sigma.prime)
        self.costVec       = np.vectorize(cost.fn)
        self.costPrimeVec  = np.vectorize(cost.prime)
        self.initialize()

    def initialize(self):
        """Initializes the biases of the network using a Gaussian distribution
        with a mean of 0 and standard deviation of 1.

        Initializes the weights (``theta``) of the network using a Gaussian
        distribution wit a mean of 0 and standard deviation of 1 divided by the
        square root of the number of weights. This helps prevent saturation of
        the neurons.
        """
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.theta  = [np.random.randn(y, x) / np.sqrt(x)
                for x, y in zip(self.layers[:-1], self.layers[1:])]

    def feedforward(self, x):
        """Returns the output of the network for input ``x``."""
        a = x
        for theta, b in zip(self.theta, self.biases):
            a = self.sigmaVec(z(theta, a, b))
        return a

    def train(self, trainingData, epochs, miniBatchSize, alpha,
            lmbda    = 0.0,
            momentum = 0.0,
            verbose  = False):
        """Trained the network by calling the train(...) method of the class
        passed through the train parameter upon the network's initialization.

        ``trainingData`` is a set a list of tuples ``(x, y)`` where ``x`` is the
        input and ``y`` is the desired output. ``epochs`` is the number of
        training cycles to be completed. ``miniBatchSize`` is the size of the
        batch to be used on each epoch. ``alpha`` is the learning rate.
        ``lmbda`` is the regularization parameter. ``momentum`` is the momentum
        of the descent.
        """
        desc.train(self, trainingData, epochs, miniBatchSize, alpha,
                lmbda   = lmbda,
                verbose = verbose)

    def save(self, filename, compressed=True):
        """Saves theta and the biases of the current network in numpy format.
        If ``compressed`` is set to True, then output file is compressed.
        """
        if compressed:
            np.savez_compressed(filename, self.theta, self.biases)
        else:
            np.savez(filename, self.theta, self.biases)

    def load(self, filename):
        """Loads theta and the biases from the passed pyz file into the current
        network.
        """
        npzfile     = np.load(filename)
        self.theta  = npzfile['arr_0']
        self.biases = npzfile['arr_1']
