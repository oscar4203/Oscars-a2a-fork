import gensim


w2v_vectors = None

def load() -> None:
  w2v_vectors = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format("../data/vectors.bin", binary=True)
  print(w2v_vectors)
  print(w2v_vectors["sex"])