"""pynn.py
~~~~~~~~~~

Author: Robert Compton

A general library for neural networks with python. Currently supports supervices
networks. Activation and cost functions along with the training algorithm are
passed as wrapper classes. This allows for user defined classes to be passed if
the built in classes do not fit the intended use case.

http://github.com/robertcompton/pynn

"""

from utils import *
from network import *
