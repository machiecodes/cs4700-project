#pragma once

#define NUM_FILES 3
#define NUM_RADII 5

extern const char FILE_NAMES[3][8];
extern const int RADII[5];

typedef struct {
    int width;
    int height;
    unsigned char *data;
} PGMImage;

PGMImage *read_pgm(const char *filename);
void write_pgm(const char *filename, const PGMImage *img);
void free_pgm(PGMImage *img);

float *create_kernel(int radius);