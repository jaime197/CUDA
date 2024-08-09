#include <stdio.h>
#include <stdint.h> 
#include <stdlib.h> 
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

typedef struct {
    uint32_t width;
    uint32_t height;
    uint16_t bpp;
    uint8_t* data;
} BMPImage;

BMPImage load_bmp(const char* filename) {
    BMPImage image;
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error al abrir el archivo BMP");
        exit(EXIT_FAILURE);
    }

    fseek(file, 18, SEEK_SET);
    fread(&image.width, sizeof(uint32_t), 1, file);
    fread(&image.height, sizeof(uint32_t), 1, file);

    fseek(file, 28, SEEK_SET);
    fread(&image.bpp, sizeof(uint16_t), 1, file);

    fseek(file, 54, SEEK_SET);
    unsigned long data_size = (size_t)(image.width * image.height * (image.bpp / 8));
    image.data = (uint8_t*)malloc(data_size);
    if (!image.data) {
        fclose(file);
        perror("Error de asignación de memoria para datos de píxeles");
        exit(EXIT_FAILURE);
    }

    fread(image.data, sizeof(uint8_t), data_size, file);
    fclose(file);
    return image;
}

__global__ void convolutionKernel(uint8_t* input, uint8_t* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        for (int c = 0; c < channels; ++c) {
            int pixel = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    pixel += input[(iy * width + ix) * channels + c];
                }
            }
            output[(y * width + x) * channels + c] = pixel / 9;
        }
    }
}

// Host function to call the CUDA kernel
void applyConvolutionCuda(uint8_t* input, uint8_t* output, int width, int height, int channels) {
    uint8_t* d_input, * d_output;
    size_t imageSize = width * height * channels * sizeof(uint8_t);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void save_bmp(const char* filename, BMPImage* image) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Error al abrir el archivo BMP para escritura");
        exit(EXIT_FAILURE);
    }

    uint32_t row_size = (image->width * (image->bpp / 8) + 3) & ~3;
    uint32_t pixel_data_size = row_size * image->height;
    uint32_t file_size = 54 + pixel_data_size;

    fwrite("BM", 1, 2, file);
    fwrite(&file_size, 4, 1, file);
    uint32_t reserved = 0;
    fwrite(&reserved, 4, 1, file);
    uint32_t offset = 54;
    fwrite(&offset, 4, 1, file);

    uint32_t dib_header_size = 40;
    fwrite(&dib_header_size, 4, 1, file);
    fwrite(&image->width, 4, 1, file);
    fwrite(&image->height, 4, 1, file);
    uint16_t planes = 1;
    fwrite(&planes, 2, 1, file);
    fwrite(&image->bpp, 2, 1, file);
    uint32_t compression = 0;
    fwrite(&compression, 4, 1, file);
    fwrite(&pixel_data_size, 4, 1, file);
    uint32_t resolution = 2835;
    fwrite(&resolution, 4, 1, file);
    fwrite(&resolution, 4, 1, file);
    uint32_t colors = 0;
    fwrite(&colors, 4, 1, file);
    fwrite(&colors, 4, 1, file);

    uint8_t padding[3] = { 0 };
    for (int y = 0; y < image->height; y++) {
        fwrite(image->data + (y * image->width * (image->bpp / 8)), 1, image->width * (image->bpp / 8), file);
        fwrite(padding, 1, (4 - (image->width * (image->bpp / 8)) % 4) % 4, file);
    }

    fclose(file);
}

int main() {
    const char* filename = "images/pikachuBMP.bmp";
    BMPImage image = load_bmp(filename);

    int width = image.width;
    int height = image.height;
    int channels = image.bpp / 8;

    uint8_t* output = (uint8_t*)malloc(width * height * channels * sizeof(uint8_t));

    applyConvolutionCuda(image.data, output, width, height, channels);

    save_bmp("images/output_cuda.bmp", &image);

    free(output);
    free(image.data);

    return 0;
}

