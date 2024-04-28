#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <fstream>

#define numIters 2500
#define visualRange 40
#define boidMass 2.0f
#define maxSpeed 25.0f
#define minDistance 7.5f
#define centerAttrWeight 0.0005f
#define repulsionWeight 0.45f
#define alignmentWeight 0.10f

void checkCudaError(std::string str)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n%s\n\n", str, cudaGetErrorString(error));
    }
}

struct Boid
{
    double x;
    double y;
    double xVel;
    double yVel;
    double xAcc;
    double yAcc;
};

struct Force
{
    double Fx;
    double Fy;
};

// Helper function to calculate distance between two boids
__device__ double calcDist(Boid& b1, Boid& b2)
{
    return sqrt((b1.x - b2.x) * (b1.x - b2.x) + (b1.y - b2.y) * (b1.y - b2.y));
}

// Helper function to apply a force to a boid
__device__ void applyForce(Boid& boid, Force& force)
{
    boid.xAcc += force.Fx / boidMass;
    boid.yAcc += force.Fy / boidMass;
}

// Apply the attraction force to center and applies it to boid
__device__ void applyCenterAttr(Boid& boid, int currIdx, Boid* boids, int nBoids)
{
    // Find the center of mass
    double xSum = 0;
    double ySum = 0;
    int count = 0;
    for(int i = 0; i < nBoids; i++)
    {
        if(calcDist(boid, boids[i]) <= visualRange && i != currIdx)
        {
            xSum += boids[i].x;
            ySum += boids[i].y;
            count++;
        }
    }
    if(count == 0)
    {
        return;
    }
    // Calculate the force
    xSum /= count;
    ySum /= count;
    double distance = sqrt((xSum - boid.x) * (xSum - boid.x) + (ySum - boid.y) * (ySum - boid.y));
    if(distance == 0)
    {
        return;
    }
    double sinTheta = (ySum - boid.y) / distance;
    double cosTheta = (xSum - boid.x) / distance;
    Force force = {centerAttrWeight * count * cosTheta / distance, 
                    centerAttrWeight * count * sinTheta / distance};
    applyForce(boid, force);
}

// Apply the repulsion force to avoid other boids
__device__ void applyAvoidOthers(Boid& boid, int currIdx, Boid* boids, int nBoids)
{
    for(int i = 0; i < nBoids; i++)
    {
        double distance = calcDist(boid, boids[i]);
        if(distance < minDistance && i != currIdx && distance != 0)
        {
            double distance = calcDist(boid, boids[i]);
            double sinTheta = (boids[i].y - boid.y) / distance;
            double cosTheta = (boids[i].x - boid.x) / distance;
            Force force = {repulsionWeight * cosTheta * (distance - minDistance), 
                            repulsionWeight * sinTheta * (distance - minDistance)};
            applyForce(boid, force);
        }
    }
}

// Apply the alignment force to make this velocity match the central velocity
__device__ void applyAlignment(Boid& boid, int currIdx, Boid* boids, int nBoids)
{
    double vXSum = 0;
    double vYSum = 0;
    int count = 0;
    for(int i = 0; i < nBoids; i++)
    {
        if(calcDist(boid, boids[i]) <= visualRange && i != currIdx)
        {
            vXSum += boids[i].xVel;
            vYSum += boids[i].yVel;
            count++;
        }
    }
    if(count == 0)
    {
        return;
    }
    // Align the velocity slightly to average velocity
    vXSum /= count;
    vYSum /= count;
    Force force = {(vXSum - boid.xVel)*alignmentWeight, 
                    (vYSum - boid.yVel)*alignmentWeight};
    applyForce(boid, force);
}

// Helper function at the end of each iteration to update position based off velocity
// LIMITS SPEED
// BOUNCES OFF WALLS
// RESETS ACCELERATION
__device__ void updateBoid(Boid& boid, int spaceSize)
{
    // UPDATE POSITIONS
    boid.x += boid.xVel;
    boid.y += boid.yVel;
    // If hit wall, bounce off
    if(boid.x < 0 || boid.x > spaceSize)
    {
        boid.xVel *= -1;
        if(boid.x < 0)
        {
            boid.x = -1 * boid.x;
        }
        else
        {
            boid.x = spaceSize - (boid.x - spaceSize);
        }
    }
    if(boid.y < 0 || boid.y > spaceSize)
    {
        boid.yVel *= -1;
        if(boid.y < 0)
        {
            boid.y = -1 * boid.y;
        }
        else
        {
            boid.y = spaceSize - (boid.y - spaceSize);
        }
    }
    // UPDATE SPEED
    boid.xVel += boid.xAcc;
    boid.yVel += boid.yAcc;
    // Limit speed
    double speed = sqrt(boid.xVel * boid.xVel + boid.yVel * boid.yVel);
    if(speed > maxSpeed)
    {
        boid.xVel = boid.xVel * maxSpeed / speed;
        boid.yVel = boid.yVel * maxSpeed / speed;
    }
    // UPDATE ACCELERATION
    boid.xAcc = 0;
    boid.yAcc = 0;
}

// One step of naive: calculate the acceleration of each boid. DOESN'T apply it yet
__global__ void naiveCalcAcc(Boid* boids, int nBoids)
{
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Boid boidCpy = boids[currIdx];
    // Center attarction force
    // NOTE: could probably optimize by making memory accesses closer to each other
    applyCenterAttr(boidCpy, currIdx, boids, nBoids);
    applyAvoidOthers(boidCpy, currIdx, boids, nBoids);
    applyAlignment(boidCpy, currIdx, boids, nBoids);
    boids[currIdx] = boidCpy;
}

__global__ void naiveUpdateBoids(Boid* boids, int nBoids, int spaceSize)
{
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Boid boidCpy = boids[currIdx];
    updateBoid(boidCpy, spaceSize);
    boids[currIdx] = boidCpy;
}

void simulation(int spaceSize, int numBoids, int blockSize){
    // Allocate memory for host
    Boid *boids;
    // Paged-locked memory doesn't get swapped back to disk
    cudaMallocHost((void**)&boids, sizeof(Boid) * numBoids);
	
    // Initialize boids
    srand(0);

    for(int i = 0; i < numBoids; i++)
    {
        // Generate random coordinates within the space size
        boids[i].x = rand() % spaceSize;
        boids[i].y = rand() % spaceSize;
        
        // Generate random velocities (-10 to 10)
        boids[i].xVel = rand() % 21 - 10;
        boids[i].yVel = rand() % 21 - 10;
        
        // Initialize acceleration to 0
        boids[i].xAcc = 0;
        boids[i].yAcc = 0;
    }

    // Allocate memory for device
    Boid* gpu_boids;
    cudaMalloc(&gpu_boids, sizeof(Boid) * numBoids);
    
    // start time
    struct timespec start, stop; 
    double time;

    // Start calling the gpu
    // cudaEventRecord(startEvent, 0);
    dim3 dimBlock(blockSize);
    dim3 dimGrid(numBoids/blockSize);
    // Run all the timesteps
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    cudaMemcpy(gpu_boids, boids, sizeof(Boid) * numBoids, cudaMemcpyHostToDevice);
    for(int i = 0; i < numIters; i++)
    {
        naiveCalcAcc <<< dimGrid, dimBlock >>> (gpu_boids, numBoids);
        checkCudaError("After naiveCalcAcc");
        naiveUpdateBoids <<< dimGrid, dimBlock >>> (gpu_boids, numBoids, spaceSize);
        checkCudaError("After naiveUpdateBoids");
        cudaMemcpy(boids, gpu_boids, sizeof(Boid) * numBoids, cudaMemcpyDeviceToHost);
        checkCudaError("After cudaMemcpy device to host");
    }
   if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("%d, %d, %f.6\n", numBoids, blockSize, time);

    // Free the memory
    cudaFreeHost(boids);
    cudaFree(gpu_boids);
}

int main(int argc, char **argv)
{
    simulation(10000, 200000, 32);
    simulation(10000, 100000, 128);
    simulation(10000, 100000, 512);
    simulation(10000, 100000, 32);
    // for(int numBoids = 1000; numBoids <= 100000; numBoids*=10){
    //     for(int i = 2; i <= 16; i*=2){
    //         int blockSize = 32 * i;
    //         int spaceSize;
    //         if (numBoids < 2000) {
    //             spaceSize = 1000;
    //         }
    //         else {
    //             spaceSize= spaceSize / 2;
    //         }
    //         simulation(spaceSize, numBoids, blockSize);
    //     }
    // }

    return 0;
}