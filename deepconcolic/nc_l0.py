from nc import NcAnalyzer
from l0_encoding import *

class NcL0Analyzer (GenericL0Analyzer, NcAnalyzer):
  """Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L0 norm.

  """
  pass

class NcL1Analyzer (GenericL1Analyzer, NcAnalyzer):
  """Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L1 norm.

  """
  pass

class NcFroAnalyzer (GenericFroAnalyzer, NcAnalyzer):
  """Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L1 norm.

  """
  pass

class NcNucAnalyzer (GenericNucAnalyzer, NcAnalyzer):
  """Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L1 norm.

  """
  pass
