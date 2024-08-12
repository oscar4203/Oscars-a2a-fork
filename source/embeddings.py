from gensim.models import KeyedVectors
from source.w2vloader import VectorsW2V

import numpy as np

class Embedding():
  def __init__(self, filename: str, custom: bool = True) -> None:
    self.custom = custom
    if self.custom:
      self.cvectors = VectorsW2V(filename)
    else:
      self.kvectors = KeyedVectors.load_word2vec_format(filename, binary=True)


  def get_vector(self, string: str) -> np.ndarray:
    vector: np.ndarray

    if self.custom:
      vector = self.cvectors[string]
    else:
      vector = self.kvectors[string]

    return vector
  
  def __getitem__(self, string: str) -> np.ndarray:
    return self.get_vector(string)

  