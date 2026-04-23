#pragma once

#define NUM_FILES 3
#define NUM_RADII 5

extern const char INPUTS[3][9];
extern const int RADII[5];

typedef struct {
    int width;
    int height;
    unsigned char *data;
} PGMImage;

PGMImage *read_pgm(const char *filename);
PGMImage *copy_pgm(PGMImage *img);
void write_pgm(const char *filename, const PGMImage *img);
void free_pgm(PGMImage *img);

float *create_2d_kernel(int radius);
float *create_1d_kernel(int radius);