import time
from src.embeddings.embeddings import Embedding

if __name__ == "__main__":
  print("Loading With Custom Vectors")
  start = time.perf_counter()
  e1 = Embedding("../data/embeddings/GoogleNews-vectors-negative300.bin", custom=True)
  stop = time.perf_counter()
  print("Loaded in", stop - start)
  some_vec = e1["poop"]
  e1.deinit()



  print("Loading With Gensim Loader")
  start = time.perf_counter()
  e1 = Embedding("../data/embeddings/GoogleNews-vectors-negative300.bin", custom=False)
  stop = time.perf_counter()
  print("Loaded in", stop - start)
  other_vec = e1["poop"]

