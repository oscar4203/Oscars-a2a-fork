import time
from source.embeddings import Embedding

if __name__ == "__main__":
  print("Loading With Custom Vectors")
  start = time.perf_counter()
  e1 = Embedding("GoogleNews-vectors-negative300.bin", custom=True)
  stop = time.perf_counter()
  print("Loaded in", stop - start)

  print("Loading With Gensim Loader")
  start = time.perf_counter()
  e1 = Embedding("GoogleNews-vectors-negative300.bin", custom=False)
  stop = time.perf_counter()
  print("Loaded in", stop - start)
