#include <stdio.h>
#include <stdlib.h>

int load_binary(const char *filename) {
  FILE *fp = fopen(filename, "rb");

  long long vectorSize, vectorCount;
  fscanf(fp, "%lld", &vectorCount);
  fscanf(fp, "%lld", &vectorCount);
  
  long start = ftell(fp);

  fseek(fp, 0L, SEEK_END);
  long size = ftell(fp);

  rewind(fp);
  unsigned char *buffer = malloc(size);


  fread(buffer, 1, size, fp);






}