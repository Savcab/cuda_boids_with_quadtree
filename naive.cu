#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define spaceSize 1000
#define numBoids 1024 //has to be multiple of blockSize^2
#define blockSize 32
#define numIters 1000
#define visualRange 200
#define centerAttractionWeight 0.01

void checkCudaError(int id)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error(id=%d): %s\n", id, cudaGetErrorString(error));
    }
}

struct Boid
{
    double x;
    double y;
    double xVel;
    double yVel;
}

// Helper function to calculate distance between two boids
double calcDistance(Boid b1, Boid b1)
{
    return sqrt((b1.x - b2.x) * (b1.x - b2.x) + (b1.y - b2.y) * (b1.y - b2.y));
}

// Helper function to calculate the attraction to the center of mass of the flock
double centerAttraction(Boid b1, Boid* boids, int nBoids)
{
    // Find the center of mass
    double xSum = 0;
    double ySum = 0;
    int count = 0;
    int i;
    for(i = 0; i < nBoids; i++)
    {
        if(calcDistance(b1, boids[i]) < visualRange)
        {
            xSum += boids[i].x;
            ySum += boids[i].y;
            count++;
        }
    }
    xSum /= count;
    ySum /= count;
    // Calculate the force
    double distance = sqrt((xSum - b1.x) * (xSum - b1.x) + (ySum - b1.y) * (ySum - b1.y));
    double force = centerAttractionWeight * count / distance;
    return force;
}

// Runs naive implementation of Reynolds' Boids for a number of iterations
__global__ void runNaive(int iters, boid* initBoids, boid* finalBoids, int nBoids)
{
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;

    for(i = 0; i < nBoids; i++){
        
    }
}

int main(int argc, char **argv)
{
    // Allocate memory for host
    int i;
    Boid *init_boids, *final_boids;
    // Paged-locked memory doesn't get swapped back to disk
    cudaMallocHost((void**)&initBoids, sizeof(Boid) * numBoids);
    cudaMallocHost((void**)&finalBoids, sizeof(Boid) * numBoids);
	
    // Initialize boids
    int gap = spaceSize / numBoids;
    for(i = 0; i < numBoids; i++)
    {
        init_boids[i].x = i * gap;
        init_boids[i].y = i * gap;
        init_boids[i].xVel = random() % 10;
        init_boids[i].yVel = random() % 10;
    }

    // Allocate memory for device
    Boid* gpu_init_boids, *gpu_final_boids;
    cudaMalloc(&gpu_init_boids, sizeof(Boid) * numBoids);
    cudaMalloc(&gpu_final_boids, sizeof(Boid) * numBoids);
    
    // create events and streams
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Start calling the gpu
    cudaEventRecord(startEvent, 0);
    dim3 dimBlock(blockSize);
    dim3 dimGrid(numBoids/blockSize);
    // Iterate througha ll the streams
    cudaMemcpy(gpu_init_boids, init_boids, sizeof(Boid) * numBoids, cudaMemcpyHostToDevice);
    runNaive <<< dimGrid, dimBlock >>> (numIters, );
    cudaMemcpy(final_boids, gpu_final_boids, sizeof(Boid) * numBoids, cudaMemcpyDeviceToHost);
    checkCudaError(0);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("time is %f ms\n", ms);
    printf("final_boids[50].x is %f: \n", final_boids[50].x);

    // Free the memory
    cudaFreeHost(boids);
    cudaFree(gpu_boids);
    return 0;
}