#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

// #define INT_MAX 2147483647

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
#define grid1Dim 20
#define block1Dim 5

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
};

// Helper function to return which area this boid belongs to
__device__ __host__ int areaId(const Boid& boid)
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
__global__ void parallelFindStartEnd(BoidsContext* context)
{
    // Unpack the context
    Boid* boids = context->boids;
    int* startIdx = context->startIdx;
    int* endIdx = context->endIdx;

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

// Helper function to assign data in the shared memory within blocks
__device__ void fillSharedMem(BoidsContext* context, Boids*** nghBoids, int** nghBoidsLen,
        int nRelX, int nRelY, int nGlobalX, int nGlobalY)
{
    int nghId = areaId(nGlobalX, nGlobalY);
    nghBoidsLen[nRelX][nRelY] = (context->startIdx[nghId] == INT_MAX && context->endIdx[nghId] == -1) ? 
        0 : context->endIdx[nghId] - context->startIdx[nghId];
    nghBoids[nRelX][nRelY] = (context->startIdx[nghId] == INT_MAX && context->endIdx[nghId] == -1) ? 
        nullptr: context->boids + context->startIdx[nghId];
}

// Helper function for Rule 2 in fulling shared memory
// NOTE: 
//  xDir = direction of x in the line(-1 = left, 0 = no move, 1 = right)
//  yDir = direction of y in the line(-1 = up, 0 = no move, 1 = down)
__device__ void fillSharedMemLine(BoidsContext* context, Boids*** nghBoids, int** nghBoidsLen, int L,
        int relX, int relY, int globalX, int globalY, int xDir, int yDir)
{
    for(int i = 1; i <= L 
        && globalX+(i*xDir) >= 0 
        && globalX+(i*xDir) < blockDim.x * gridDim.x
        && globalY+(i*yDir) >= 0
        && globalY+(i*yDir) < blockDim.y * gridDim.y
        ; i++)
    {
        fillSharedMem(context, nghBoids, nghBoidsLen, relX+(i*xDir), relY+(i*yDir), globalX+(i*xDir), globalY+(i*yDir));
    }
}

// Helper function for Rule 3 in fulling shared memory
// NOTE: 
//  xDir = direction of x in the line(-1 = left, 0 = no move, 1 = right)
//  yDir = direction of y in the line(-1 = up, 0 = no move, 1 = down)
__device__ void fillSharedMemSquare(BoidsContext* context, Boids*** nghBoids, int** nghBoidsLen, int L,
        int relX, int relY, int gloalX, int globalY, int xDir, int yDir)
{
    for(int i = 1; i <= L 
        && globalX+(i*xDir) >= 0 
        && globalX+(i*xDir) < blockDim.x * gridDim.x
        && globalY+(i*yDir) >= 0
        && globalY+(i*yDir) < blockDim.y * gridDim.y
        ; i++)
    {
        // Go along the x-axis first
        fillSharedMemLine(context, nghBoids, nghBoidsLen, L, relX+(i*xDir), relY, globalX+(i*xDir), globalY, 0, yDir);
    }
}

// One step of naive: calculate the acceleration of each boid. DOESN'T apply it yet
__global__ void areaCalcAcc(BoidsContext* context)
{
    // Unpack the context
    Boids* boids = context->boids;
    int* startIdx = context->startIdx;
    int* endIdx = context->endIdx;

    // Global index of XY on the grid
    int currX = blockIdx.x * blockDim.x + threadIdx.x;
    int currY = blockIdx.y * blockDim.y + threadIdx.y;
    int aId = areaId(currX, currY);
    // Edge case: empty area
    if(startIdx[aId] == INT_MAX && endIdx[aId] == -1)
    {
        return;
    }

    // Find the neighborhood around it within it's visual range
    // NOTE: as in the whole block's neighborhood. We use shared memory here for efficiency
    int L = visualRange / (spaceSize / (grid1Dim * block1Dim)) + 1; // the visual range in term of blocks
    __shared__ Boid* nghBoids[2*L + blockDim.x][2*L + blockDim.y];
    __shared__ int nghBoidsLen[2*L + blockDim.x][2*L + blockDim.y];
    // Rules for which neighboring grids to fill out for each thread in the block
    //  1: each thread fills out it's own block
    //  2: threads on the perimeter of the block fills out the lines of neighbors expending beyond it
    //  3: threads on the corners fills in the square of neighbors in the corners
    int currRelX = L + threadIdx.x; //Relative position of XY on the neighbor shared memory
    int currRelY = L + threadIdx.y;
    // Rule 1
    fillSharedMem(context, nghBoids, nghBoidsLen, nghId, currRelX, currRelY, currX, currY);
    // Rule 2
    if(threadIdx.x == 0)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, -1, 0);
    } 
    if(threadIdx.x == blockDim.x-1)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, 1, 0);
    }
    if(threadIdx.y == 0)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, 0, -1);
    }
    if(threadIdx.y == blockDim.y-1)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, 0, 1);
    }
    // Rule 3
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, -1, -1);
    }
    if(threadIdx.x == 0 && threadIdx.y == blockDim.y-1)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, -1, 1);
    }
    if(threadIdx.x == blockDim.x-1 && threadIdx.y == 0)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, 1, -1);
    }
    if(threadIdx.x == blockDi.x-1 && threadIdx.y == blockDim.y-1)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, L, currRelX, currRelY, currX, currY, 1, 1);
    }
    __syncthreads();

    // Iterate through all boids in this area and do all calculations
    for(int i = startIdx[aId]; i < endIdx[aId]; i++)
    {
        Boid boidCpy = boids[i];
        // Find neighbor upperbounds in nghBoids
        int lowerX = (currX - L < 0) ? currRelX - currX : currRelX - L; //inclusive
        int lowerY = (currY - L < 0) ? currRelY - currY : currRelY - L;
        int upperX = (currX + L >= gridDim.x * blockDim.x) ? currRelX + (gridDim.x*blockDim.x-1-currX): currRelX + L; //inclusive
        int upperY = (currY + L >= gridDim.y * blockDim.y) ? currRelY + (gridDim.y*blockDim.y-1-currY): currRelY + L;

        // Go through each neighboring grid in shared memory
        for(int x = lowerX; x <= upperX; x++)
        {
            for(int y = lowerY; y <= upperY; y++)
            {
                int currIdx = (currRelX == x && currRelY == y) ? i - startIdx[aId] : -1;
                Boid* nStartPtr = nghBoids[x][y];
                int nghNumBoids = nghBoidsLen[x][y];
                applyCenterAttr(boidCpy, currIdx, nStartPtr, nghNumBoids);
                applyAvoidOthers(boidCpy, currIdx, nStartPtr, nghNumBoids);
                applyAlignment(boidCpy, currIdx, nStartPtr, nghNumBoids);
            }
        }
        boids[currIdx] = boidCpy;
    }
}

// Updates all the boids in this area
__global__ void areaUpdateBoids(Boid* boids, int* startIdx, int* endIdx)
{
    int currX = blockIdx.x * blockDim.x + threadIdx.x;
    int currY = blockIdx.y * blockDim.y + threadIdx.y;
    int aId = areaId(currX, currY);

    int starting = startIdx[aId];
    int ending = endIdx[aId];
    for(int i = starting; i < ending; i++)
    {
        Boid boidCpy = boids[currIdx];
        updateBoid(boidCpy);
        boids[currIdx] = boidCpy;
    }
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
};

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
    BoidsContext* gpu_context;
    cudaMalloc(&gpu_context, sizeof(BoidsContext));

    Boid* gpu_boidsPtr = thrust::raw_pointer_cast(gpu_boids.data());
    int* gpu_startIdxPtr = thrust::raw_pointer_cast(gpu_startIdx.data());
    int* gpu_endIdxPtr = thrust::raw_pointer_cast(gpu_endIdx.data());

    // Construct the context and copy it over to GPU
    BoidsContext context;
    context.boids = gpu_boidsPtr;
    context.startIdx = gpu_startIdxPtr;
    context.endIdx = gpu_endIdxPtr;
    cudaMemcpy(gpu_context, &context, sizeof(BoidsContext), cudaMemcpyHostToDevice);
    
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

        areaCalcAcc <<< dimGrid, dimBlock >>> (gpu_context);
        checkCudaError("After areaCalcAcc");
        areaUpdateBoids <<< dimGrid, dimBlock >>> (gpu_context);
        checkCudaError("After naiveUpdateBoids");
        boids = gpu_boids;
        checkCudaError("After copying gpu_boids back to host");
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

    // Free cuda memory used
    cudaFree(gpu_context);

    return 0;
}