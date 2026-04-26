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

__constant__ float c_kernel[MAX_KERNEL_DIM];

__global__ void gaussian_blur_h(
    const unsigned char *input,
    float *intermediate,
    const int width,
    const int height,
    const int radius
) {
    const int tile_w = BLOCK_SIZE + 2 * radius;
    extern __shared__ unsigned char h_tile[];

    const int out_x = BLOCK_SIZE * static_cast<int>(blockIdx.x) + static_cast<int>(threadIdx.x);
    const int out_y = BLOCK_SIZE * static_cast<int>(blockIdx.y) + static_cast<int>(threadIdx.y);

    const int tile_origin_x = BLOCK_SIZE * static_cast<int>(blockIdx.x) - radius;

    for (int j = static_cast<int>(threadIdx.x); j < tile_w; j += BLOCK_SIZE) {
        const int src_x = max(0, min(tile_origin_x + j, width - 1));
        const int src_y = max(0, min(out_y, height - 1));
        h_tile[static_cast<int>(threadIdx.y) * tile_w + j] = input[src_y * width + src_x];
    }

    __syncthreads();

    if (out_x >= width || out_y >= height) return;

    float sum = 0.0f;
    const int kernel_dim = 2 * radius + 1;
    for (int kx = 0; kx < kernel_dim; kx++) {
        sum += c_kernel[kx] * static_cast<float>(
            h_tile[static_cast<int>(threadIdx.y) * tile_w + static_cast<int>(threadIdx.x) + kx]);
    }

    intermediate[out_y * width + out_x] = sum;
}

__global__ void gaussian_blur_v(
    const float *intermediate,
    unsigned char *output,
    const int width,
    const int height,
    const int radius
) {
    const int tile_h = BLOCK_SIZE + 2 * radius;
    extern __shared__ float v_tile[];

    const int out_x = BLOCK_SIZE * static_cast<int>(blockIdx.x) + static_cast<int>(threadIdx.x);
    const int out_y = BLOCK_SIZE * static_cast<int>(blockIdx.y) + static_cast<int>(threadIdx.y);

    const int tile_origin_y = BLOCK_SIZE * static_cast<int>(blockIdx.y) - radius;

    for (int i = static_cast<int>(threadIdx.y); i < tile_h; i += BLOCK_SIZE) {
        const int src_x = max(0, min(out_x, width - 1));
        const int src_y = max(0, min(tile_origin_y + i, height - 1));
        v_tile[i * BLOCK_SIZE + static_cast<int>(threadIdx.x)] = intermediate[src_y * width + src_x];
    }

    __syncthreads();

    if (out_x >= width || out_y >= height) return;

    float sum = 0.0f;
    const int kernel_dim = 2 * radius + 1;
    for (int ky = 0; ky < kernel_dim; ky++) {
        sum += c_kernel[ky] * v_tile[(static_cast<int>(threadIdx.y) + ky) * BLOCK_SIZE + static_cast<int>(threadIdx.x)];
    }

    output[out_y * width + out_x] = static_cast<unsigned char>(sum);
}

int main() {
    FILE *csv = fopen("results.csv", "a");
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

        // converting to unsigned char after the horizontal pass would introduce
        // rounding before vertical weights are applied
        float *d_intermediate;
        CHECK_CUDA(cudaMalloc(&d_input, image_size));
        CHECK_CUDA(cudaMalloc(&d_output, image_size));
        CHECK_CUDA(cudaMalloc(&d_intermediate, image_size * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));

        for (const int radius : RADII) {
            char correct_file_name[64];
            sprintf(correct_file_name, "src/images/correct/r%d-%s", radius, fileName);
            PGMImage *correct = read_pgm(correct_file_name);

            const int kernel_dim = 2 * radius + 1;
            const size_t kernel_size = kernel_dim * sizeof(float);
            float *h_kernel = create_1d_kernel(radius);

            CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size));

            dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridDim(
                (input->width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (input->height + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            const int tile_size = BLOCK_SIZE + 2 * radius;
            const size_t h_shared = static_cast<size_t>(BLOCK_SIZE) * tile_size * sizeof(unsigned char);
            const size_t v_shared = static_cast<size_t>(tile_size) * BLOCK_SIZE * sizeof(float);

            float times[5];

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // The first iteration is just to warm up kernels
            for (int i = 0; i < 6; i++) {
                cudaEventRecord(start);

                gaussian_blur_h<<<gridDim, blockDim, h_shared>>>(
                    d_input, d_intermediate, input->width, input->height, radius);
                gaussian_blur_v<<<gridDim, blockDim, v_shared>>>(
                    d_intermediate, d_output, input->width, input->height, radius);

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
                        if (abs(static_cast<int>(h_output[j]) - static_cast<int>(correct->data[j])) > 1) {
                            printf("Mismatch at index %d: expected %hhu, got %hhu\n",
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

            fprintf(csv, "sep,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                fileName, radius, times[0], times[1], times[2], times[3], times[4], average_time);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            free(h_kernel);

            free_pgm(correct);
        }

        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_intermediate));

        CHECK_CUDA(cudaFreeHost(h_input));
        CHECK_CUDA(cudaFreeHost(h_output));

        free_pgm(input);
    }

    fclose(csv);
}
