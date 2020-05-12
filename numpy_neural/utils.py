"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import numpy as np


def to_one_hot(labels, classes):
    ret = np.zeros([labels.shape[0], classes]).astype(np.float)
    ret[range(labels.shape[0]), labels] = 1.
    return ret


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    den = exps.sum(axis=1)
    den = np.tile(den, [exps.shape[1], 1]).T
    return exps / den

