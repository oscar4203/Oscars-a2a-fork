import ctypes
import os
import numpy as np

import time

class HashEntry(ctypes.Structure):
#   struct entry_t {
#     struct entry_t *next;
#     char *name;
#     float *vector;
#   };

  _fields_ = [('next', ctypes.c_void_p),
              ('name', ctypes.c_char_p),
              ('vector', ctypes.POINTER(ctypes.c_float))]

class VectorsW2V():
  def __init__(self, path: str, normalize: bool = False) -> None:


    self.dll = ctypes.CDLL(os.getcwd() + "/w2vloader.dll")
    c_path = ctypes.create_string_buffer(path.encode())

    self.dll.load_binary(c_path, ctypes.c_char(normalize))

    # set up function stuff
    self.dll.lookup_entry.restype = ctypes.c_void_p
    self.dll.get_vector_size = ctypes.c_longlong
    self.dll.get_word_count = ctypes.c_longlong



  def get_vector(self, word: str) -> np.ndarray:
    # struct entry_t *lookup_entry(char *name) {

    c_string = ctypes.create_string_buffer(word.encode())

    entry = HashEntry.from_address(self.dll.lookup_entry(c_string))
    # v_size = self.dll.get_vector_size().value
    # print(type(v_size), v_size, "please work")
    vector = np.ctypeslib.as_array(entry.vector, (300,)).copy()

    return vector

  def __getitem__(self, word: str) ->np.ndarray:
    return self.get_vector(word)









if __name__ == "__main__":
  start = time.perf_counter()
  vectors = VectorsW2V("vectors.bin")
  end = time.perf_counter()

  print("Time elapsed:", end - start)

  fart = vectors["fart"]

  print(fart)
  pass
