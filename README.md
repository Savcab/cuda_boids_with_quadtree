# CUDA Analysis with Naive and Grid Methods

This is the final research project for Felix Chen and Brice Brown for EE451 - Parallel and Distributed Systems at USC.

We used CARC as our platform of choice.

To run the jobs you have to load the modules and compile with a command like
nvcc -o naive naive_testing.cu

and then submit the job by doing sbatch job.sl

For the OpenGL version you would need to get a interactive desktop running and then use the commands in the interactive_job file to propely compile and then you just do ./boids_simulation
