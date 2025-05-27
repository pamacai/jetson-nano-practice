__global__ void generateMasksKernel(float* output, unsigned char* masks, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        // Example thresholding to create a binary mask
        masks[index] = (output[index] > 0.5f) ? 255 : 0;
    }
}

extern "C" void generateMasks(float* output, unsigned char* masks, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    generateMasksKernel<<<gridSize, blockSize>>>(output, masks, width, height);
}