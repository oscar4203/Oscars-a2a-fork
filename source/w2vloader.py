# Description: Load word2vec vectors from a binary file using a custom loader.

# Standard Libraries
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

    fullpath: str = os.getcwd() + "/w2vloader.dll"
    print("Full path", fullpath)
    self.dll = ctypes.CDLL(fullpath)
    c_path = ctypes.create_string_buffer(path.encode())

    self.dll.load_binary(c_path)


    # set up function stuff
    self.dll.lookup_entry.stype = ctypes.c_void_p
    self.dll.get_vector_size.restype = ctypes.c_longlong
    self.dll.get_word_count.restype = ctypes.c_longlong



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
  
  def get_vector_size(self) -> int:
    return int(self.dll.get_vector_size())
  
  def get_word_count(self) -> int:
    return int(self.dll.get_word_count())
  
  def deinit(self):
    self.dll.unload_binary()
    









if __name__ == "__main__":


  start = time.perf_counter()
  end = time.perf_counter()
  vectors = VectorsW2V("vectors.bin")
  print("Time elapsed:", end - start)
  fart = vectors["fart"]

  print(fart)
  pass
