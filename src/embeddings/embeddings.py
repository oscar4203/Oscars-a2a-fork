from gensim.models import KeyedVectors
from src.embeddings.w2vloader import VectorsW2V
import time


import numpy as np

class Embedding():
  def __init__(self, filename: str, custom: bool = True) -> None:
    self.custom = custom

    start_time = time.perf_counter()
    if self.custom:
      self.cvectors = VectorsW2V(filename)
      self.vector_size: int = self.cvectors.get_vector_size()
    else:
      self.kvectors = KeyedVectors.load_word2vec_format(filename, binary=True)
      self.vector_size: int = self.kvectors.vector_size

    end_time = time.perf_counter()

  def get_vector(self, string: str) -> np.ndarray | None:
    vector: np.ndarray

    if self.custom:
      vector = self.cvectors[string]
    else:
      vector = self.kvectors[string]

    return vector

  def __getitem__(self, string: str) -> np.ndarray:
    vector = self.get_vector(string)
    if vector is None:
      raise KeyError(f"Vector for '{string}' not found.")
    return vector

  def deinit(self):
    if self.custom:
      self.cvectors.deinit()






