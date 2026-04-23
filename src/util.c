#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char FILE_NAMES[3][9] = {"1024.pgm", "2048.pgm", "4096.pgm"};
static const int RADII[5] = {3, 5, 7, 9, 11};

typedef struct {
    int width;
    int height;
    unsigned char *data;
} PGMImage;

PGMImage *read_pgm(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to read PGM file %s", filename);
        exit(1);
    }

    PGMImage *img = malloc(sizeof(PGMImage));

    // Consume "P5" line
    char pgm_header[3];
    fscanf(file, "%s", pgm_header);

    fscanf(file, "%d %d", &img->width, &img->height);

    // Consume "255" line
    char max_val[4];
    fscanf(file, "%s", max_val);

    const size_t num_pixels = img->width * img->height;
    img->data = malloc(num_pixels);
    fread(img->data, 1, num_pixels, file);

    fclose(file);
    return img;
}

void write_pgm(const char *filename, const PGMImage *img) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to write PGM file %s", filename);
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height, file);
    fclose(file);
}

void free_pgm(PGMImage *img) {
    free(img->data);
    free(img);
}

float *create_kernel(const int radius) {
    const float blur = (float) radius / 3.0f;
    const int size = 2 * radius + 1;

    float *kernel = malloc(size * size * sizeof(float));
    float sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            const float val = expf((float) -(j * j + i * i) / (2.0f * blur * blur));
            kernel[(i + radius) * size + (j + radius)] = val;
            sum += val;
        }
    }

    // Normalize so kernel weights sum to 1
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
    return kernel;
}
