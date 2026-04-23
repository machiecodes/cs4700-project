#pragma once

typedef struct {
    int width;
    int height;
    unsigned char *data;
} PGMImage;

PGMImage *read_pgm(const char *filename);
void write_pgm(const char *filename, const PGMImage *img);
void free_pgm(PGMImage *img);