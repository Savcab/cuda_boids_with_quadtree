#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <fstream>

// Include CUDA headers
#include <cublas.h>

GLuint vbo;  // Handle for the VBO
struct cudaGraphicsResource *cuda_vbo_resource; // CUDA Graphics Resource for mapping

#define spaceSize 1028
#define numBoids  512
#define blockSize 32
#define numIters 5000
#define visualRange 20
#define boidMass 1.0f
#define maxSpeed 0.5f
#define minDistance 10.0f
#define centerAttrWeight 0.005f
#define repulsionWeight 0.05f
#define alignmentWeight 0.75f

struct Boid
{
    float x, y, xVel, yVel, xAcc, yAcc;
};

struct Force
{
    float Fx, Fy;
};

void checkCudaError(const std::string& str)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n%s\n\n", str.c_str(), cudaGetErrorString(error));
    }
}




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


struct SimulationState {
    dim3 dimBlock;
    dim3 dimGrid;
    Boid* gpu_boids = nullptr;
    Boid* host_boids = nullptr; // Add this line to hold the host-side boids
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    
    static SimulationState& getInstance() {
        static SimulationState instance; // Guaranteed to be destroyed and instantiated on first use.
        return instance;
    }

    SimulationState() : dimBlock(0, 0, 0), dimGrid(0, 0, 0), gpu_boids(nullptr), cuda_vbo_resource(nullptr) {
        // You could initialize other members here if needed
    }

    // Make sure constructors and operators are as required
    SimulationState(const SimulationState&) = delete; // Delete Copy constructor
    void operator=(const SimulationState&) = delete; // Delete assignment operator

    // Idle function that can be called with no captures
   // Idle function that can be called with no captures
    static void idleFunction() {
        auto& instance = SimulationState::getInstance();

        naiveCalcAcc<<<instance.dimGrid, instance.dimBlock>>>(instance.gpu_boids, numBoids);
        cudaDeviceSynchronize();
        naiveUpdateBoids<<<instance.dimGrid, instance.dimBlock>>>(instance.gpu_boids, numBoids);
        cudaDeviceSynchronize();

        static int frameCounter = 0;
        if (frameCounter++ % 60 == 0) {  // For example, print every 60 frames
            cudaMemcpy(instance.host_boids, instance.gpu_boids, sizeof(Boid) * numBoids, cudaMemcpyDeviceToHost);
            checkCudaError("Failed to copy boids back to host");

            // // Print the positions of the first few boids for debugging
            // for (int i = 0; i < std::min(numBoids, 10); ++i) {
            //     printf("Boid %d: Position (%.2f, %.2f), Velocity (%.2f, %.2f)\n",
            //         i, instance.host_boids[i].x, instance.host_boids[i].y, instance.host_boids[i].xVel, instance.host_boids[i].yVel);
            // }
        }
            // In your idle functio
        glutPostRedisplay();
    }

};

void display() {
    auto& instance = SimulationState::getInstance(); // Access the singleton instance

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Clear with a dark gray to see the points
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity(); // Reset the current modelview matrix

    // For each boid, draw a triangle pointed in the direction of its velocity
    for (int i = 0; i < numBoids; ++i) {
        // Compute the angle of rotation based on the velocity vector
        float angle = atan2f(instance.host_boids[i].yVel, instance.host_boids[i].xVel);

        glPushMatrix(); // Save the current transformation
        glTranslatef(instance.host_boids[i].x, instance.host_boids[i].y, 0.0f); // Translate to the boid's position
        glRotatef(angle * 180.0f / M_PI, 0.0f, 0.0f, 1.0f); // Rotate to align with the velocity vector

        glBegin(GL_TRIANGLES);
        glColor3f(1.0, 0.0, 0.0); // Red color for visibility
        // Define the vertices of the triangle relative to the boid's position
        glVertex2f(0.0f, 5.0f); // Point of the triangle
        glVertex2f(-2.5f, -2.5f); // Base - Left Vertex
        glVertex2f(2.5f, -2.5f); // Base - Right Vertex
        glEnd();

        glPopMatrix(); // Restore the transformation
    }

    glutSwapBuffers();
}

void timer(int value) {
    auto& instance = SimulationState::getInstance();
    
    naiveCalcAcc<<<instance.dimGrid, instance.dimBlock>>>(instance.gpu_boids, numBoids);
    cudaDeviceSynchronize();
    naiveUpdateBoids<<<instance.dimGrid, instance.dimBlock>>>(instance.gpu_boids, numBoids);
    cudaDeviceSynchronize();
    
    cudaMemcpy(instance.host_boids, instance.gpu_boids, sizeof(Boid) * numBoids, cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy boids back to host");

    glutPostRedisplay();
    glutTimerFunc(16, timer, 0); // Call this timer function again after 16 milliseconds
}


int main(int argc, char **argv) {
    initOpenGL(argc, argv);
    glutDisplayFunc(display);

    // Existing CUDA setup...
    // Allocate memory for host
    Boid* host_boids;
    cudaMallocHost((void**)&host_boids, sizeof(Boid) * numBoids);
    
	
    // Initialize boids
    int gap = spaceSize / numBoids;
    for(int i = 0; i < numBoids; i++)
    {
        host_boids[i].x = i * gap;
        host_boids[i].y = i * gap;
        host_boids[i].xVel = rand() % 20 - 10;
        host_boids[i].yVel = rand() % 20 - 10;
        host_boids[i].xAcc = 0;
        host_boids[i].yAcc = 0;
    }

    // Allocate memory for device
    Boid* gpu_boids;
    cudaMalloc(&gpu_boids, sizeof(Boid) * numBoids);
    // start time
    struct timespec start, stop; 
    double time;
    std::ofstream ofile("output.txt");
    std::ofstream oTestFile("test.txt");
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    cudaMemcpy(gpu_boids, host_boids, sizeof(Boid) * numBoids, cudaMemcpyHostToDevice);

    // Set up the singleton instance
    auto& state = SimulationState::getInstance();
    state.host_boids = host_boids; // Set the host_boids here
    state.dimBlock = dim3(blockSize);
    state.dimGrid = dim3(numBoids/blockSize);
    state.gpu_boids = gpu_boids;
    state.cuda_vbo_resource = cuda_vbo_resource; // Make sure to initialize this

    glutTimerFunc(0, timer, 0);
    
    glutMainLoop();
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    ofile.close();
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %.9f\n", time*1e9);

    cudaFreeHost(host_boids);
    cudaFree(gpu_boids);
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);
    return 0;
}
