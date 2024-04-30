#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for srand()
GLuint vbo;  // Handle for the VBO
struct cudaGraphicsResource *cuda_vbo_resource; // CUDA Graphics Resource for mapping

// #define INT_MAX 2147483647

#define spaceSize 1000
#define numBoids  128
#define numIters 5000
#define visualRange 40
#define boidMass 2.0f
#define maxSpeed 50.0f
#define minDistance 10.0f
#define centerAttrWeight 0.30f
#define repulsionWeight 1.0f
#define alignmentWeight 0.10f

// Blocks and grids are 2D now, these are parameters along one of the edges
#define grid1Dim 2
#define block1Dim 2

// For ease of programming and compiling
#define L (visualRange / (spaceSize / (grid1Dim * block1Dim)) + 1) // visual range in terms of areas
#define nghSide (2*L + block1Dim) // side length of neighborhood shared memory block

void checkCudaError(std::string str)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n%s\n\n", str, cudaGetErrorString(error));
    }
}

struct Boid
{
    int id; // Unique identifier for the boid
    float x, y, xVel, yVel, xAcc, yAcc;
};

struct Force
{
    float Fx, Fy;
};

// For convienience
struct BoidsContext
{
    Boid* boids;
    int* startIdx;
    int* endIdx;
};

void initOpenGL(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 800);
    glutCreateWindow("CUDA Boids Simulation");

    glewInit();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, spaceSize, 0, spaceSize, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 800, 800);

    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numBoids * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register the VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

__global__ void updateVBO(float *vbo, Boid *boids, int nBoids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBoids) {
        vbo[2 * idx] = boids[idx].x;
        vbo[2 * idx + 1] = boids[idx].y;
    }
}

// Helper function to return which area this boid belongs to
__device__ __host__ int areaId(const Boid& boid)
{
    int numAreas1D = block1Dim * grid1Dim;
    int areaSide = spaceSize / numAreas1D;
    return ((int)boid.x / areaSide) + numAreas1D * ((int)boid.y / areaSide);
}

// Helper function to convert xy thread coordinate to areaId
__device__ __host__ int areaId(int myThreadX, int myThreadY)
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
__device__ void fillSharedMem(BoidsContext* context, Boid* nghBoids[nghSide][nghSide], int nghBoidsLen[nghSide][nghSide],
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
__device__ void fillSharedMemLine(BoidsContext* context, Boid* nghBoids[nghSide][nghSide], int nghBoidsLen[nghSide][nghSide],
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
__device__ void fillSharedMemSquare(BoidsContext* context, Boid* nghBoids[nghSide][nghSide], int nghBoidsLen[nghSide][nghSide],
        int relX, int relY, int globalX, int globalY, int xDir, int yDir)
{
    for(int i = 1; i <= L 
        && globalX+(i*xDir) >= 0 
        && globalX+(i*xDir) < blockDim.x * gridDim.x
        && globalY+(i*yDir) >= 0
        && globalY+(i*yDir) < blockDim.y * gridDim.y
        ; i++)
    {
        // Go along the x-axis first
        fillSharedMemLine(context, nghBoids, nghBoidsLen, relX+(i*xDir), relY, globalX+(i*xDir), globalY, 0, yDir);
    }
}

// One step of area: calculate the acceleration of each boid. DOESN'T apply it yet
__global__ void areaCalcAcc(BoidsContext* context)
{
    // Unpack the context
    Boid* boids = context->boids;
    int* startIdx = context->startIdx;
    int* endIdx = context->endIdx;

    // Global index of XY on the grid
    int currX = blockIdx.x * blockDim.x + threadIdx.x;
    int currY = blockIdx.y * blockDim.y + threadIdx.y;
    int aId = areaId(currX, currY);

    // Find the neighborhood around it within it's visual range
    // NOTE: as in the whole block's neighborhood. We use shared memory here for efficiency
    // NOTE: L is the visual range in terms of blocks
    __shared__ Boid* nghBoids[nghSide][nghSide];
    __shared__ int nghBoidsLen[nghSide][nghSide];
    // Rules for which neighboring grids to fill out for each thread in the block
    //  1: each thread fills out it's own block
    //  2: threads on the perimeter of the block fills out the lines of neighbors expending beyond it
    //  3: threads on the corners fills in the square of neighbors in the corners
    int currRelX = L + threadIdx.x; //Relative position of XY on the neighbor shared memory
    int currRelY = L + threadIdx.y;
    // Rule 1
    fillSharedMem(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY);
    // Rule 2
    if(threadIdx.x == 0)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, -1, 0);
    } 
    if(threadIdx.x == blockDim.x-1)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, 1, 0);
    }
    if(threadIdx.y == 0)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, 0, -1);
    }
    if(threadIdx.y == blockDim.y-1)
    {
        fillSharedMemLine(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, 0, 1);
    }
    // Rule 3
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, -1, -1);
    }
    if(threadIdx.x == 0 && threadIdx.y == blockDim.y-1)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, -1, 1);
    }
    if(threadIdx.x == blockDim.x-1 && threadIdx.y == 0)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, 1, -1);
    }
    if(threadIdx.x == blockDim.x-1 && threadIdx.y == blockDim.y-1)
    {
        fillSharedMemSquare(context, nghBoids, nghBoidsLen, currRelX, currRelY, currX, currY, 1, 1);
    }
    __syncthreads();

    // Edge case: empty area
    if(startIdx[aId] == INT_MAX && endIdx[aId] == -1)
    {
        return;
    }
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
                if(nStartPtr != nullptr)
                {
                    applyCenterAttr(boidCpy, currIdx, nStartPtr, nghNumBoids);
                    applyAvoidOthers(boidCpy, currIdx, nStartPtr, nghNumBoids);
                    applyAlignment(boidCpy, currIdx, nStartPtr, nghNumBoids);
                }
            }
        }
        boids[i] = boidCpy;
    }
}

// Updates all the boids in this area
__global__ void areaUpdateBoids(BoidsContext* context)
{
    // Unpack the context
    Boid* boids = context->boids;
    int* startIdx = context->startIdx;
    int* endIdx = context->endIdx;

    int currX = blockIdx.x * blockDim.x + threadIdx.x;
    int currY = blockIdx.y * blockDim.y + threadIdx.y;
    int aId = areaId(currX, currY);

    int starting = startIdx[aId];
    int ending = endIdx[aId];
    for(int i = starting; i < ending; i++)
    {
        Boid boidCpy = boids[i];
        updateBoid(boidCpy);
        boids[i] = boidCpy;
        // int targetID = 2;
        // if (boids[i].id == targetID) {
        //     printf("Ker Boid %d: Pos(%.2f, %.2f) Vel(%.2f, %.2f)\n", targetID,
        //            boids[i].x, boids[i].y,
        //            boids[i].xVel, boids[i].yVel);
        // }
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


class SimulationState {
public:
    static SimulationState& getInstance() {
        static SimulationState instance;
        return instance;
    }

    // Prevent copy and move semantics
    SimulationState(const SimulationState&) = delete;
    SimulationState& operator=(const SimulationState&) = delete;
    SimulationState(SimulationState&&) = delete;
    SimulationState& operator=(SimulationState&&) = delete;

    thrust::host_vector<Boid> boids;
    thrust::device_vector<Boid> gpu_boids;
    thrust::device_vector<int> gpu_startIdx;
    thrust::device_vector<int> gpu_endIdx;
    BoidsContext* gpu_context;

    dim3 dimBlock;
    dim3 dimGrid;
    dim3 dimBlockLinear;
    dim3 dimGridLinear;

private:
    // Private constructor
    SimulationState() {
        init();
    }

    ~SimulationState() {
        cleanup();
    }

    void init() {
        srand(0);  // Seed for random number generation

        // Boid initialization
        boids = thrust::host_vector<Boid>(numBoids);
        for (int i = 0; i < numBoids; ++i) {
            Boid temp = {
                i,
                static_cast<float>(rand() % spaceSize), // Explicit cast to float
                static_cast<float>(rand() % spaceSize), // Explicit cast to float
                static_cast<float>(rand() % 20 - 10),   // Explicit cast to float
                static_cast<float>(rand() % 20 - 10),   // Explicit cast to float
                0.0f,  // Already float, no cast needed
                0.0f   // Already float, no cast needed
            };
            boids.push_back(temp);
        }
        gpu_boids = thrust::device_vector<Boid>(boids);
        gpu_startIdx = thrust::device_vector<int>(grid1Dim * block1Dim * grid1Dim * block1Dim, INT_MAX);
        gpu_endIdx = thrust::device_vector<int>(grid1Dim * block1Dim * grid1Dim * block1Dim, -1);

        // Context for CUDA kernels
        cudaMalloc(&gpu_context, sizeof(BoidsContext));
        checkCudaError("Failed to allocate device memory for BoidsContext");
        BoidsContext context = {thrust::raw_pointer_cast(gpu_boids.data()), thrust::raw_pointer_cast(gpu_startIdx.data()), thrust::raw_pointer_cast(gpu_endIdx.data())};
        cudaMemcpy(gpu_context, &context, sizeof(BoidsContext), cudaMemcpyHostToDevice);
        checkCudaError("Failed to copy BoidsCOntext to device");
        dimBlock = dim3(block1Dim, block1Dim);
        dimGrid = dim3(grid1Dim, grid1Dim);
        dimBlockLinear = dim3(block1Dim * block1Dim);
        dimGridLinear = dim3(grid1Dim * grid1Dim);
    }

    void cleanup() {
        // Cleanup logic, e.g., free memory
        cudaFree(gpu_context);
    }
};

#define DEG_PER_RAD (180.0f / M_PI)

void display() {
    auto& instance = SimulationState::getInstance();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // printBoidDetails<<<1, 1>>>(instance.gpu_context, 2);
    cudaDeviceSynchronize();
    // Draw boids
    for (int i = 0; i < numBoids; ++i) {
        float angle = atan2f(instance.boids[i].yVel, instance.boids[i].xVel) * DEG_PER_RAD;
        float boidSize = 3.0f;

        glPushMatrix();
        glTranslatef(instance.boids[i].x, instance.boids[i].y, 0.0f);
        glRotatef(angle - 90.0f, 0.0f, 0.0f, 1.0f);

        glBegin(GL_TRIANGLES);
        glColor3f(1.0, 0.0, 0.0); // Red color for visibility
        glVertex2f(0.0f, boidSize * 2.0f); // Point of the triangle
        glVertex2f(-boidSize, -boidSize); // Base - Left Vertex
        glVertex2f(boidSize, -boidSize); // Base - Right Vertex
        glEnd();

        glPopMatrix();
    }

    // Draw grid lines
    int numLines = grid1Dim * block1Dim;
    float spacing = spaceSize / static_cast<float>(numLines);
    glColor3f(0.5f, 0.5f, 0.5f); // Gray color for grid lines

    glBegin(GL_LINES);
    // Vertical lines
    for (int i = 0; i <= numLines; ++i) {
        glVertex2f(i * spacing, 0.0f);
        glVertex2f(i * spacing, spaceSize);
    }
    // Horizontal lines
    for (int i = 0; i <= numLines; ++i) {
        glVertex2f(0.0f, i * spacing);
        glVertex2f(spaceSize, i * spacing);
    }
    glEnd();

    glutSwapBuffers();
}


__global__ void printBoidDetails(BoidsContext* context, int targetID) {
    for (int i = 0; i < numBoids; i++) {
        if (context->boids[i].id == targetID) {
            printf("GPU Boid %d: Pos(%.2f, %.2f) Vel(%.2f, %.2f)\n", targetID,
                   context->boids[i].x, context->boids[i].y,
                   context->boids[i].xVel, context->boids[i].yVel);
            break;
        }
    }
}

__global__ void printGpuBoids(Boid* boids, int targetID) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numBoids) {
        if (boids[idx].id == targetID) {
            printf("GPU_boids Boid %d: Pos(%.2f, %.2f) Vel(%.2f, %.2f)\n",
                   boids[idx].id, boids[idx].x, boids[idx].y, boids[idx].xVel, boids[idx].yVel);
        }
    }
}
__global__ void checkPointerConsistency(Boid* directPointer, BoidsContext* context) {
    if (threadIdx.x == 0) {  // Use only one thread to print
        printf("Direct vector pointer: %p\n", directPointer);
        printf("Context vector pointer: %p\n", context->boids);
    }
}


void printBoids(thrust::host_vector<Boid> boids)
{
    for(int i = 0; i < numBoids; i++)
    {
        std::cout << "area ID: " << areaId(boids[i]) << "\n";
        std::cout << "XY coord: ()" << boids[i].x << ", " << boids[i].y << ")\n";
    }
}

void timer(int value) {
    auto& instance = SimulationState::getInstance();
    thrust::sort(instance.gpu_boids.begin(), instance.gpu_boids.end(), CompareByAreaId());
    thrust::fill(instance.gpu_startIdx.begin(), instance.gpu_startIdx.end(), INT_MAX);
    thrust::fill(instance.gpu_endIdx.begin(), instance.gpu_endIdx.end(), -1);
    // parallelFindStartEnd <<< instance.dimGridLinear, instance.dimBlockLinear >>> (instance.gpu_context);
    // cudaDeviceSynchronize();
    // areaCalcAcc <<< instance.dimGrid, instance.dimBlock >>> (instance.gpu_context);
    // cudaDeviceSynchronize();
    // checkCudaError("After areaCalcAcc");
    // areaUpdateBoids <<< instance.dimGrid, instance.dimBlock >>> (instance.gpu_context);
    // cudaDeviceSynchronize();

    //printBoidDetails<<<1, 1>>>(instance.gpu_context, 2);
    // cudaDeviceSynchronize();
    int targetID = 2;  // Example: Check for Boid with ID 2
    //printGpuBoids<<<(numBoids + 255) / 256, 256>>>(thrust::raw_pointer_cast(instance.gpu_boids.data()), targetID);
    // cudaDeviceSynchronize();  // Synchronize to ensure the print statements complete
    checkCudaError("After printing gpu_boids");
    checkCudaError("After naiveUpdateBoids");

    thrust::copy(instance.gpu_boids.begin(), instance.gpu_boids.end(), instance.boids.begin());
    cudaDeviceSynchronize();

    // DEBUG
    printBoids(instance.boids);


    checkCudaError("After copying gpu_boids back to host");
    // for (int i = 0; i < numBoids; i++) {
    //     int targetID = 2;
    //     if (instance.boids[i].id == targetID) { // targetID is the ID of the boid you want to track
    //         std::cout << "Boid " << targetID << ": Pos(" << instance.boids[i].x << ", " << instance.boids[i].y << ") Vel(" << instance.boids[i].xVel << ", " << instance.boids[i].yVel << ")\n";
    //         break;
    //     }
    // }

    // Boid* directPointer = thrust::raw_pointer_cast(instance.gpu_boids.data());
    // checkPointerConsistency<<<1, 1>>>(directPointer, instance.gpu_context);
    // cudaDeviceSynchronize();
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     fprintf(stderr, "CUDA error during pointer consistency check: %s\n", cudaGetErrorString(error));
    // }

    glutPostRedisplay();
    glutTimerFunc(16, timer, 0); // Call this timer function again after 16 milliseconds   
}


int main(int argc, char **argv)
{
    initOpenGL(argc, argv);
    glutDisplayFunc(display);
    

    auto& state = SimulationState::getInstance();

    glutTimerFunc(0, timer, 0);
    glutMainLoop();
    
    return 0;
}