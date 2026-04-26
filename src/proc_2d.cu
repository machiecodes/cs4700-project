#include <cstdio>
#include <cstdlib>
#include "util.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define BLOCK_SIZE 16
#define MAX_KERNEL_DIM 23

__constant__ float c_kernel[MAX_KERNEL_DIM * MAX_KERNEL_DIM];

__global__ void gaussian_blur_2d(
    const unsigned char *input,
    unsigned char *output,
    const int width,
    const int height,
    const int radius
) {
    const int tile_size = BLOCK_SIZE + 2 * radius;
    extern __shared__ unsigned char tile[];

    const int out_x = BLOCK_SIZE * static_cast<int>(blockIdx.x) + static_cast<int>(threadIdx.x);
    const int out_y = BLOCK_SIZE * static_cast<int>(blockIdx.y) + static_cast<int>(threadIdx.y);

    // Top-left corner of this block's tile in global image coordinates
    const int tile_origin_x = BLOCK_SIZE * static_cast<int>(blockIdx.x) - radius;
    const int tile_origin_y = BLOCK_SIZE * static_cast<int>(blockIdx.y) - radius;

    // Cooperatively load the tile into shared memory.
    // The tile is larger than the block, so each thread may load more than one element.
    for (int i = static_cast<int>(threadIdx.y); i < tile_size; i += BLOCK_SIZE) {
        for (int j = static_cast<int>(threadIdx.x); j < tile_size; j += BLOCK_SIZE) {
            int src_x = tile_origin_x + j;
            int src_y = tile_origin_y + i;

            // Clamp to image bounds (same border handling as CPU)
            src_x = max(0, min(src_x, width - 1));
            src_y = max(0, min(src_y, height - 1));

            tile[i * tile_size + j] = input[src_y * width + src_x];
        }
    }

    __syncthreads();

    // Now compute the output pixel, reading from shared memory
    if (out_x >= width || out_y >= height) return;

    float sum = 0.0f;
    const int kernel_dim = 2 * radius + 1;

    for (int ky = 0; ky < kernel_dim; ky++) {
        for (int kx = 0; kx < kernel_dim; kx++) {
            const float weight = c_kernel[ky * kernel_dim + kx];
            const float pixel  = tile[(threadIdx.y + ky) * tile_size + (threadIdx.x + kx)];
            sum += pixel * weight;
        }
    }

    output[out_y * width + out_x] = static_cast<unsigned char>(sum);
}

int main() {
    FILE *csv = fopen("results.csv", "w");
    fprintf(csv, "Impl,File,Radius,Time 1,Time 2,Time 3,Time 4,Time 5,Average\n");

    for (const auto fileName : INPUTS) {
        char input_file_name[64];
        sprintf(input_file_name, "src/images/input/%s", fileName);
        PGMImage *input = read_pgm(input_file_name);

        const int image_size = input->width * input->height;

        unsigned char *h_input, *h_output;
        CHECK_CUDA(cudaMallocHost(&h_input, image_size));
        CHECK_CUDA(cudaMallocHost(&h_output, image_size));
        memcpy(h_input, input->data, image_size);

        unsigned char *d_input, *d_output;
        CHECK_CUDA(cudaMalloc(&d_input, image_size));
        CHECK_CUDA(cudaMalloc(&d_output, image_size));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));

        for (const int radius : RADII) {
            char correct_file_name[64];
            sprintf(correct_file_name, "src/images/correct/r%d-%s", radius, fileName);
            PGMImage *correct = read_pgm(correct_file_name);

            const int kernel_dim = 2 * radius + 1;
            const size_t kernel_size = kernel_dim * kernel_dim * sizeof(float);
            float *h_kernel = create_2d_kernel(radius);

            CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size));

            dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridDim(
                (input->width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (input->height + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            const int tile_size = BLOCK_SIZE + 2 * radius;
            const size_t shared_mem_bytes = tile_size * tile_size * sizeof(unsigned char);

            float times[5];

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // First iteration is just to warm up kernel
            for (int i = 0; i < 6; i++) {
                cudaEventRecord(start);

                gaussian_blur_2d<<<gridDim, blockDim, shared_mem_bytes>>>(
                    d_input, d_output, input->width, input->height, radius);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                CHECK_CUDA(cudaDeviceSynchronize());

                if (i != 0) {
                    float elapsed_time = 0;
                    cudaEventElapsedTime(&elapsed_time, start, stop);
                    times[i - 1] = elapsed_time;
                } else {
                    CHECK_CUDA(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));

                    for (int j = 0; j < image_size; j++) {
                        if (h_output[j] != correct->data[j]) {
                            printf("Mismatch at index %d: expected %hhu, got %hhu",
                                j, correct->data[j], h_output[j]);
                            exit(1);
                        }
                    }
                }
            }

            float average_time = 0.0f;
            for (const float time : times) {
                average_time += time;
            }
            average_time /= 5.0f;

            fprintf(csv, "2d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                fileName, radius, times[0], times[1], times[2], times[3], times[4], average_time);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            free(h_kernel);

            free_pgm(correct);
        }

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));

        CHECK_CUDA(cudaFreeHost(h_input));
        CHECK_CUDA(cudaFreeHost(h_output));

        free_pgm(input);
    }

    fclose(csv);
}