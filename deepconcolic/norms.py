from abc import abstractmethod
from engine import Metric
import numpy as np


# ---


class Norm (Metric):
  '''
  Just an alias for norms.
  '''

  def close_to(self, refs, x):
    for r in refs:
      # print("Distance and factor", self.distance(r, x), self.factor)
      if self.distance (r, x) <= self.factor:
        return True;
    return False


# ---


class L0 (Norm):
  '''
  L0 norm.
  '''

  def __init__(self, scale = 255, **kwds):
    super().__init__(scale = scale, **kwds)


  def __repr__(self):
    return 'L0'


  @property
  def is_int(self):
    return True


  def distance(self, x, y):
    return np.count_nonzero (np.abs (x - y) * self.scale > 1)


class L1 (Norm):
  '''
  L1 norm.
  '''

  def __init__(self, scale = 255, **kwds):
    super().__init__(scale = scale, **kwds)


  def __repr__(self):
    return 'L0'


  @property
  def is_int(self):
    return True


  def distance(self, x, y):
    vector = (x-y).reshape(28, 28)
    return np.linalg.norm(vector, ord=1)

# ---

class Fro (Norm):
  '''
  Fro norm.
  '''

  def __init__(self, scale = 255, **kwds):
    super().__init__(scale = scale, **kwds)


  def __repr__(self):
    return 'L0'


  @property
  def is_int(self):
    return True


  def distance(self, x, y):
    vector = (x-y).reshape(28, 28)
    return np.linalg.norm(vector, ord='fro')

# ---

class Nuc (Norm):
  '''
  Fro norm.
  '''

  def __init__(self, scale = 255, **kwds):
    super().__init__(scale = scale, **kwds)


  def __repr__(self):
    return 'L0'


  @property
  def is_int(self):
    return True


  def distance(self, x, y):
    vector = (x-y).reshape(28, 28)
    return np.linalg.norm(vector, ord='nuc')

# ---


class LInf (Norm):
  '''
  L-inf norm.
  '''

  def __repr__(self):
    return 'Linf'


  def distance(self, x, y):
    return np.amax(np.absolute (x - y) * self.scale)


# ---
