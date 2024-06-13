#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TABLE_SIZE 5000000
#define MAX_WORD_SIZE 50

struct entry_t {
  struct entry_t *next;
  char *name;
  float *vector;
};

struct {
  struct entry_t **lookup;
  long long lookup_size;
  long long vector_size;
  long long word_count;
} hash_table;


void init() {
  hash_table.lookup_size = TABLE_SIZE;
  hash_table.lookup = malloc(sizeof(struct entry_t *) * TABLE_SIZE);
}


void create_entry(char *name, float *vector) {

}

struct entry_t *lookup_entry(char *name) {

}


int bgetc(char *b, long long *cur) {
  int c = b[*cur];

  *cur += 1;

  return c;
}

void *bread(void *dest, int elemSize, int elemCount, char *b, long long *cur) {
  void *result = memcpy(dest, &b[*cur], elemCount * elemSize);

  *cur += elemCount * elemSize;

  return result;
}

void load_binary(const char *filename) {
  FILE *fp = fopen(filename, "rb");


  // Get dimension and word count
  long long vectorSize, vectorCount;
  fscanf(fp, "%lld", &vectorCount);
  fscanf(fp, "%lld", &vectorSize);
  hash_table.word_count = vectorCount;
  hash_table.vector_size = vectorSize;
  
  long start = ftell(fp);
  fpos_t pos;
  fgetpos(fp, &pos);

  fseek(fp, 0L, SEEK_END);
  long size = ftell(fp) - start;
  fsetpos(fp, &pos);

  



  char *buffer = malloc(size);
  fread(buffer, 1, size, fp);
  fclose(fp);

  long long cur = 0;

  for (int vectorIndex; vectorIndex < vectorCount; vectorIndex++) {
    char *name = &buffer[cur];
    for (int c = 0; c < MAX_WORD_SIZE && cur < size; c++) {
      int ch = bgetc(buffer, &cur);
      if (ch == ' ' || ch == '\n') break;
    }
    // Get the vector and move the cursor
    float *vector = (float*)&buffer[cur];
    cur += vectorSize * sizeof(float);
    // normalize step
    float len = 0;
    for (int i = 0; i < vectorSize; i++)
      len += vector[i] * vector[i];

    len = sqrtf(len);

    for (int i = 0; i < vectorSize; i++)
      vector[i] /= len;
    
    // now we create a hashtable entry

  }




}