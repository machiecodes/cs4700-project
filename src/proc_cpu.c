#include <stdio.h>
#include <stdlib.h>
#include "util.h"

int main() {
    for (int i = 0; i < 3; i++) {
        char input_file_name[64];
        sprintf(input_file_name, "src/images/input/%s", INPUTS[i]);
        PGMImage *input = read_pgm(input_file_name);
        PGMImage *output = copy_pgm(input);

        for (int j = 0; j < 5; j++) {
            const int radius = RADII[j];
            float *kernel = create_2d_kernel(radius);

            for (int y = 0; y < input->height; y++) {
                for (int x = 0; x < input->width; x++) {
                    float sum = 0.0f;

                    for (int ky = -radius; ky <= radius; ky++) {
                        for (int kx = -radius; kx <= radius; kx++) {
                            // Clamp border handling
                            int sample_x = x + kx;
                            int sample_y = y + ky;

                            if (sample_x < 0) sample_x = 0;
                            if (sample_y < 0) sample_y = 0;

                            if (sample_x >= input->width)  sample_x = input->width - 1;
                            if (sample_y >= input->height) sample_y = input->height - 1;

                            const float pixel = input->data[sample_y * input->width + sample_x];
                            const float weight = kernel[(ky + radius) * (2 * radius + 1) + (kx + radius)];
                            sum += pixel * weight;
                        }
                    }

                    output->data[y * input->width + x] = (unsigned char) sum;
                }
            }

            char output_file_name[64];
            sprintf(output_file_name, "src/images/correct/r%d-%s", radius, INPUTS[i]);
            write_pgm(output_file_name, output);

            free(kernel);
        }

        free_pgm(input);
        free_pgm(output);
    }
}