#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <fstream>

#define spaceSize 10000
#define numBoids  128//has to be multiple of blockSize
#define numIters 5000
#define visualRange 10000
#define boidMass 1
#define maxSpeed 100
#define minDistance 5
#define centerAttrWeight 0.05
#define repulsionWeight 0.5
#define alignmentWeight 0.05

#define dimBlock 32
#define dimGrid (numBoids / dimBlock)

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
__device__ void updateBoid(Boid& boid)
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

__global__ void naiveUpdateBoids(Boid* boids, int nBoids)
{
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Boid boidCpy = boids[currIdx];
    updateBoid(boidCpy);
    boids[currIdx] = boidCpy;
}

int main(int argc, char **argv)
{
    // Allocate memory for host
    Boid *boids;
    // Paged-locked memory doesn't get swapped back to disk
    cudaMallocHost((void**)&boids, sizeof(Boid) * numBoids);
	
    // Initialize boids
    int gap = spaceSize / numBoids;
    for(int i = 0; i < numBoids; i++)
    {
        boids[i].x = i * gap;
        boids[i].y = i * gap;
        boids[i].xVel = rand() % 20 - 10;
        boids[i].yVel = rand() % 20 - 10;
        boids[i].xAcc = 0;
        boids[i].yAcc = 0;
    }

    // Allocate memory for device
    Boid* gpu_boids;
    cudaMalloc(&gpu_boids, sizeof(Boid) * numBoids);
    
    // start time
    struct timespec start, stop; 
    double time;
    std::ofstream ofile("output.txt");
    std::ofstream oTestFile("test.txt");

    // Start calling the gpu
    // cudaEventRecord(startEvent, 0);
    dim3 dimBlock(dimBlock);
    dim3 dimGrid(dimGrid);
    // Run all the timesteps
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    cudaMemcpy(gpu_boids, boids, sizeof(Boid) * numBoids, cudaMemcpyHostToDevice);
    for(int i = 0; i < numIters; i++)
    {
        naiveCalcAcc <<< dimGrid, dimBlock >>> (gpu_boids, numBoids);
        checkCudaError("After naiveCalcAcc");
        naiveUpdateBoids <<< dimGrid, dimBlock >>> (gpu_boids, numBoids);
        checkCudaError("After naiveUpdateBoids");
        cudaMemcpy(boids, gpu_boids, sizeof(Boid) * numBoids, cudaMemcpyDeviceToHost);
        checkCudaError("After cudaMemcpy device to host");
        // Print out the all the boids
        ofile << "ITERATION " << i << "\n";
        for(int j = 0; j < numBoids; j++)
        {
            ofile << "Boid " << j << ": " << boids[j].x << ", " << boids[j].y << "\n";
        }
        oTestFile << boids[0].x << " " << boids[0].y << "\n";
    }
   if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    ofile.close();
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %.9f\n", time*1e9);

    // Free the memory
    cudaFreeHost(boids);
    cudaFree(gpu_boids);
    return 0;
}