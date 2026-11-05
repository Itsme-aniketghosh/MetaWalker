"""metawalker: learned black-box optimizers for big, complex search spaces.

    from metawalker import minimize, register, minimize_auto, recommend, certify, importance, compare, train
    from metawalker.sklearn import MetaWalkerCV   # needs scikit-learn
"""
from .api import METHODS, Result, minimize, register
from .asktell import AskTell
from .certify import Certified, certify
from .compare import Comparison, compare
from .importance import Importance, importance
from .problems import FAMILIES, TestProblem, make_problem
from .recommend import Recommendation, minimize_auto, recommend
from .space import Categorical, Float, Int, Space
from .train import TrainedPolicy, train

__all__ = ['minimize', 'register', 'minimize_auto', 'recommend', 'Recommendation', 'certify', 'Certified', 'importance',
           'Importance', 'compare', 'Comparison', 'train', 'TrainedPolicy', 'AskTell', 'Result', 'METHODS',
           'make_problem', 'TestProblem', 'FAMILIES', 'Space', 'Float', 'Int', 'Categorical']
__version__ = '0.2.7'
