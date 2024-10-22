#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 1024 // Tamaño de las matrices

// Kernel CUDA para la multiplicación de matrices
__global__ void matrixMulCUDA(int* A, int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // índice de fila
    int col = blockIdx.x * blockDim.x + threadIdx.x; // índice de columna

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Función para inicializar las matrices con valores aleatorios
void initializeMatrix(int* matrix, int width) {
    for (int i = 0; i < width * width; i++) {
        matrix[i] = rand() % 100; // Números aleatorios entre 0 y 99
    }
}

// Función para comparar matrices y validar resultados
bool compareMatrices(int* C, int* C_seq, int width) {
    for (int i = 0; i < width * width; i++) {
        if (C[i] != C_seq[i]) {
            return false;
        }
    }
    return true;
}

// Multiplicación de matrices secuencial para validar resultados
void matrixMulSequential(int* A, int* B, int* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int sum = 0;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

int main() {
    srand(time(0));

    // Tamaño de las matrices
    int size = N * N * sizeof(int);

    // Reservar memoria en la CPU
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);
    int* h_C_seq = (int*)malloc(size);

    // Inicializar las matrices A y B con valores aleatorios
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Reservar memoria en la GPU
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiar matrices A y B a la memoria de la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir el número de bloques e hilos por bloque
    int blockSize = 16; // Tamaño del bloque
    dim3 threadsPerBlock(blockSize, blockSize); // Hilos por bloque
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize); // Número de bloques

    // Medir el tiempo de la operación en GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Llamar al kernel CUDA para la multiplicación de matrices
    matrixMulCUDA << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar el resultado C de la GPU a la CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Realizar la multiplicación secuencial para validar el resultado
    clock_t seq_start = clock();
    matrixMulSequential(h_A, h_B, h_C_seq, N);
    clock_t seq_end = clock();
    double seq_time = double(seq_end - seq_start) / CLOCKS_PER_SEC * 1000;

    // Validar los resultados
    if (compareMatrices(h_C, h_C_seq, N)) {
        std::cout << "La multiplicación de matrices es correcta." << std::endl;
    }
    else {
        std::cout << "Error en la multiplicación de matrices." << std::endl;
    }

    // Imprimir el tiempo tomado por GPU y CPU
    std::cout << "Tiempo en GPU: " << milliseconds << " ms" << std::endl;
    std::cout << "Tiempo en CPU: " << seq_time << " ms" << std::endl;

    // Liberar memoria
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_seq);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
