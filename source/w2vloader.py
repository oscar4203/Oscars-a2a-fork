# Description: Load word2vec vectors from a binary file using a custom loader.

# Standard Libraries
import ctypes
import os
import platform
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
    # Determine the correct shared object file extension
    system = platform.system()
    if system == "Windows":
      extension = "dll"
      print("System detected as Windows")
    elif system == "Linux":
      extension = "so"
      print("System detected as Linux")
    else:
      raise OSError(f"Unsupported operating system: {system}")

    # Construct the full path to the shared object file
    fullpath: str = os.path.join(os.getcwd(), f"w2vloader.{extension}")
    print("Full path", fullpath)
    self.dll = ctypes.CDLL(fullpath)
    c_path = ctypes.create_string_buffer(path.encode())

    # load the binary file
    self.dll.load_binary(c_path)

    # set up function stuff
    self.dll.lookup_entry.restype = ctypes.c_void_p
    self.dll.get_vector_size.restype = ctypes.c_longlong
    self.dll.get_word_count.restype = ctypes.c_longlong


  def get_vector(self, word: str) -> np.ndarray:
    # struct entry_t *lookup_entry(char *name) {
    vec_size = self.get_vector_size()
    c_string = ctypes.create_string_buffer(word.encode())
    pointer = self.dll.lookup_entry(c_string)
    if pointer == None:
      return np.zeros((vec_size,))
    entry = HashEntry.from_address(pointer)
    p_vector = np.ctypeslib.as_array(entry.vector, (vec_size,))

    vector = p_vector.copy()

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
  # Test the VectorsW2V class
  start = time.perf_counter()
  vectors = VectorsW2V("vectors.bin")
  end = time.perf_counter()
  print("Time elapsed:", end - start)
  fart = vectors["fart"]

  print(fart)
  pass
