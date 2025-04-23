#include "time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// add a makefile to produce a dll
#ifdef _WIN32
  #define FTELL64(pfile) _ftelli64(pfile)
  #define FSEEK64(pfile, offset, origin) _fseeki64(pfile, offset, origin)
#else
  #define FTELL64(pfile) ftello64(pfile)
  #define FSEEK64(pfile, offset, origin) fseeko64(pfile, offset, origin)
#endif


#define TABLE_SIZE 5000000
#define MAX_WORD_SIZE 50



struct entry_t {
  struct entry_t *next;
  char *name;
  float *vector;
};

struct {
  char *buffer;
  struct entry_t **lookup;
  long long lookup_size;
  long long vector_size;
  long long word_count;

} hash_table;


long long get_vector_size() {
  return hash_table.vector_size;
}

long long get_word_count() {
  return hash_table.word_count;
}




unsigned long long hash(char *str) {
    unsigned long long hash = 5381;
    int c;

    unsigned char *strr = (unsigned char *)str;

    while (c = *strr++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}


void create_entry(char *name, float *vector) {
  struct entry_t *entry = malloc(sizeof(struct entry_t));
  entry->name = name;
  entry->vector = vector;
  entry->next = NULL;

  unsigned long long index = hash(name);
  index = index % hash_table.lookup_size;

  struct entry_t *lookup = hash_table.lookup[index];

  while (lookup) {
    if (lookup->next == NULL) break;

    lookup = lookup->next;
  }

  if (lookup) {
    lookup->next = entry;
  } else {
    hash_table.lookup[index] = entry;
  }

}

struct entry_t *lookup_entry(char *name) {
  unsigned long long index = hash(name);
  index = index % hash_table.lookup_size;

  struct entry_t *entry = hash_table.lookup[index];

  while (entry && strcmp(name, entry->name) != 0)
    entry = entry->next;


  return entry;
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
  time_t tstart = time(NULL);


  FILE *fp = fopen(filename, "rb");


  // Get dimension and word count
  long long vectorSize, vectorCount;
  fscanf(fp, "%lld", &vectorCount);
  fscanf(fp, "%lld", &vectorSize);
  // printf("LOADING BINARY: %s Vector size %lld, Vector Count %lld\n", filename, vectorSize, vectorCount);

  // Set up the hash table
  hash_table.word_count = vectorCount;
  hash_table.vector_size = vectorSize;


  hash_table.lookup_size = (vectorCount * 3)/2;
  hash_table.lookup = malloc(sizeof(struct entry_t *) * hash_table.lookup_size);
  memset(hash_table.lookup, 0, sizeof(struct entry_t *) * hash_table.lookup_size);

  // Get the file size and allocate for it
  long long start = FTELL64(fp);
  fpos_t pos;
  fgetpos(fp, &pos);

  int dunno = FSEEK64(fp, 0L, SEEK_END);
  long long end = FTELL64(fp);
  long long size = end - start;
  fsetpos(fp, &pos);

  char *buffer = malloc(size);
  hash_table.buffer = buffer;
  // printf("Reading into Buffer with size of %lld\n", size);
  fread(buffer, 1, size, fp);
  fclose(fp);
  // printf("Done Reading buffer\n");


  time_t tend = time(NULL);




  long long cur = 0;

  for (int vectorIndex; vectorIndex < vectorCount; vectorIndex++) {
    // printf("got here");
    // clean the name
    char *name = &buffer[cur];
    for (int c = 0; c < MAX_WORD_SIZE && cur < size; c++) {
      int ch = bgetc(buffer, &cur);
      // if (ch == 0) break;
      if (ch == ' ' || ch == '\n') {
        buffer[cur - 1] = '\0';
        break;
      }
    }
    // Get the vector and move the cursor
    float *vector = (float*)&buffer[cur];
    cur += vectorSize * sizeof(float);

    create_entry(name, vector);
    // printf("Loaded Vector #%d: %s\n", vectorIndex, name);
  }
}



void unload_binary() {
  for (long long i = 0; i < hash_table.lookup_size; i++) {
    if (hash_table.lookup[i]) {
      struct entry_t *next = hash_table.lookup[i]->next;
      while (next) {
        next = next->next;
        struct entry_t *old = next;
        free(old);
      }
      free(hash_table.lookup[i]);
    }
  }

  free(hash_table.lookup);
  free(hash_table.buffer);
}


// int main() {

//   clock_t start = clock();

//   load_binary("data/vectors.bin");
//   clock_t end = clock();

//   double elapsed_time = (end-start)/(double)CLOCKS_PER_SEC ;

//   printf("Loaded Binary in %f seconds\n", elapsed_time);


//   start = clock();
//   unload_binary();

//   end = clock();

//   elapsed_time = (end-start)/(double)CLOCKS_PER_SEC ;

//   printf("Unloaded Binary in %f seconds\n", elapsed_time);



// }
