import ctypes
import os

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


    self.dll = ctypes.CDLL(os.getcwd() + "\w2vloader.dll")
    c_path = ctypes.create_string_buffer(path.encode())
    self.dll.load_binary(c_path, ctypes.c_char(normalize))

    # set up function stuff
    self.dll.lookup_entry.restype = ctypes.c_void_p





  def get_vector(self, word: str) -> None:
    # struct entry_t *lookup_entry(char *name) {

    c_string = ctypes.create_string_buffer(word.encode())
    
    entry = HashEntry.from_address(self.dll.lookup_entry(c_string))

    print(entry.name)
    # entry.name








if __name__ == "__main__":
  start = time.perf_counter()
  vectors = VectorsW2V("vectors.bin")
  end = time.perf_counter()

  print("Time elapsed:", end-start)

  fart = vectors.get_vector("fart")
  pass