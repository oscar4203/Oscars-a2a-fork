#include <stdio.h>
#include <stdlib.h>

// we need to parse the binary and then put that 
// into a hash table, and provide hash table api
// features

struct {
  long long vectorSize;
  long long vectorCount;
  char **words;
  float **vectors;
} hashTable;


int load_binary(const char *filename) {
  FILE *fp = fopen(filename, "rb");

  long long vectorSize, vectorCount;
  // one of these needs to be changed to vectorSize
  fscanf(fp, "%lld", &vectorCount);
  fscanf(fp, "%lld", &vectorCount);
  
  long start = ftell(fp);

  fseek(fp, 0L, SEEK_END);
  long size = ftell(fp);

  rewind(fp);
  unsigned char *buffer = malloc(size);

  fread(buffer, 1, size, fp);
}

