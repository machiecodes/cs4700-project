#include <stdlib.h>
#include <math.h>

// FREE AFTER USE
float *gaussian(const int radius) {
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

static const float sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

static const float sobel_y[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};