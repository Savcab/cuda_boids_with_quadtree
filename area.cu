#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define INT_MAX 2147483647

#define spaceSize 10000
#define numBoids  128
#define numIters 5000
#define visualRange 10000
#define boidMass 1
#define maxSpeed 100
#define minDistance 5
#define centerAttrWeight 0.05
#define repulsionWeight 0.5
#define alignmentWeight 0.05

// Blocks and grids are 2D now, these are parameters along one of the edges
#define grid1Dim 10
#define block1Dim 10

// Dimensions for finding the Start & End for boids in areaIds
#define gridDimSE 10
#define blockDimSE 10

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

// For convienience
struct BoidsContext
{
    Boid* boids;
    int* startIdx;
    int* endIdx;
}

// Helper function to return which area this boid belongs to
__device__ __host__ int areaId(Boid& boid)
{
    int numAreas1D = block1Dim * grid1Dim;
    int areaSide = spaceSize / numAreas1D;
    return ((int)boid.x / areaSide) + numAreas1D * ((int)boid.y / areaSide);
}

// Helper function to convert xy thread coordinate to areaId
__device__ int areaId(int myThreadX, int myThreadY)
{
    int numAreas1D = block1Dim * grid1Dim;
    return myThreadX + myThreadY * numAreas1D;
}

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

// NOTE: 
//  startIdx is inclusive
//  endIdx is exclusive
__global__ void parallelFindStartEnd(Boid* boids, int* startIdx, int* endIdx)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startingIdx = numBoids * threadId / (gridDim.x * blockDim.x);
    int endingIdx  = numBoids * (threadId + 1) / (gridDim.x * blockDim.x);
    // Initialize the conditions
    int prevVal = areaId(boids[startingIdx]);
    atomicMin(&(startIdx[prevVal]), startingIdx);
    atomicMax(&(endIdx[prevVal]), startingIdx);
    // Iterating through the whole thing linearly
    // CAN BE OPTIMIZED BY USING BINARY SEARCH
    for(int i = startingIdx + 1; i < endingIdx; i++)
    {
        int currVal = areaId(boids[i]);
        if(currVal != prevVal)
        {
            atomicMin(&(startIdx[currVal]), i);
            atomicMax(&(endIdx[currVal]), i);
            prevVal = currVal;
        }
    }
}

// One step of naive: calculate the acceleration of each boid. DOESN'T apply it yet
__global__ void areaCalcAcc(Boid* boids, int* startIdx, int* endIdx)
{
    int currX = blockIdx.x * blockDim.x + threadIdx.x;
    int currY = blockIdx.y * blockDim.y + threadIdx.y;
    int aId = areaId(currX, currY);
    // Edge case: empty area
    if(startIdx[aId] == INT_MAX && endIdx[aId] == -1)
    {
        return;
    }

    // Find areas around it that we need to inlude in calculations
    // L = neighborhood radius(neighborhood is a square block)
    int L = visualRange / (spaceSize / (grid1Dim * block1Dim)) + 1;
    int neighArea = (2*L+1) * (2*L+1);
    int neighIds[neighArea];
    int count = 0;
    for(int y = currY - L; y <= currY + L; y++)
    {
        for(int x = currX - L; x <= currX + L; x++)
        {
            neighIds[count] = areaId(x, y);
            count++;
        }
    }

    // Iterate through all boids in this area and do all calculations
    for(int i = startIdx[aId]; i < endIdx[aId]; i++)
    {
        Boid boidCpy = boids[i];
        // Itearte through whole neighborhood to calculate forces
        for(int nIdx = 0; nIdx < neighArea; nIdx++)
        {
            // TODO: need to fix currIdx thing!
            int nStartIdx = startIdx[neighIds[nIdx]];
            int nEndIdx = endIdx[neighIds[nIdx]];
            int nBoids = nEndIdx - nStartIdx;
            if(nStartIdx == INT_MAX && nEndIDx == -1)
            {
                continue;
            }
            Boid* nStartPtr = boids + nStartIdx;
            applyCenterAttr(boidCpy, currIdx, nStartPtr, nBoids);
            applyAvoidOthers(boidCpy, currIdx, nStartPtr, nBoids);
            applyAlignment(boidCpy, currIdx, nStartPtr, nBoids);
        }
        boids[currIdx] = boidCpy;
    }
}

// TODO: need to implement
__global__ void areaUpdateBoids(Boid* boids, int* startIdx, int* endIdx)
{
    int currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Boid boidCpy = boids[currIdx];
    updateBoid(boidCpy);
    boids[currIdx] = boidCpy;
}

// Customer comparator functor for sorting boids according to which area they land in
struct CompareByAreaId {
    __host__ __device__
    bool operator()(const Boid& boid1, const Boid& boid2) const
    {
        int aId1 = areaId(boid1);
        int aId2 = areaId(boid2);
        return aId1 <= aId2;
    }
}

int main(int argc, char **argv)
{
    // Allocate memory for host
    thrust::host_vector<Boid> boids;
	
    // Initialize boids
    int gap = spaceSize / numBoids;
    for(int i = 0; i < numBoids; i++)
    {
        Boid temp;
        temp.x = i * gap;
        temp.y = i * gap;
        temp.xVel = rand() % 20 - 10;
        temp.yVel = rand() % 20 - 10;
        temp.xAcc = 0;
        temp.yAcc = 0;
        boids.push_back(temp);
    }

    // Allocate memory for device
    thrust::device_vector<Boid> gpu_boids = boids;
    thrust::device_vector<int> gpu_startIdx(grid1Dim * block1Dim * grid1Dim * block1Dim, INT_MAX);
    thrust::device_vector<int> gpu_endIdx(grid1Dim * block1Dim * grid1Dim * block1Dim, -1);

    Boid* gpu_boidsPtr = thrust::raw_pointer_cast(gpu_boids.data());
    int* gpu_startIdxPtr = thrust::raw_pointer_cast(gpu_startIdx.data());
    int* gpu_endIdxPtr = thrust::raw_pointer_cast(gpu_endIdx.data());
    
    // start time
    struct timespec start, stop; 
    double time;
    std::ofstream ofile("output.txt");
    std::ofstream oTestFile("test.txt");

    // Start calling the gpu
    dim3 dimBlock(block1Dim, block1Dim);
    dim3 dimGrid(grid1Dim, grid1Dim);

    dim3 dimBlockLinear(block1Dim * block1Dim);
    dim3 dimGridLinear(grid1Dim * grid1Dim);
    // Run all the timesteps
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    for(int i = 0; i < numIters; i++)
    {
        // Sort the boids such that boids in the same area are next to each other
        thrust::sort(gpu_boids.begin(), gpu_boids.end(), CompareByAreaId());
        // Find start and end indices in "boids" for every area 
        thrust::fill(gpu_startIdx.begin(), gpu_startIdx.end(), INT_MAX);
        thrust::fill(gpu_endIdx.begin(), gpu_endIdx.end(), -1);
        parallelFindStartEnd <<< dimGridLinear, dimBlockLinear >>> (gpu_boidsPtr, gpu_startIdxPtr, gpu_endIdxPtr);

        areaCalcAcc <<< dimGrid, dimBlock >>> (gpu_boidsPtr, gpu_startIdxPtr, gpu_endIdxPtr);
        checkCudaError("After areaCalcAcc");
        areaUpdateBoids <<< dimGrid, dimBlock >>> (gpu_boidsPtr, gpu_startIdxPtr, gpu_endIdxPtr);
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