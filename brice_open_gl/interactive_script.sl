module purge
module load nvidia-hpc-sdk
module load gcc/11.3.0 cuda/11.6.2 cmake gmake glew/2.2.0 freeglut/3.2.2 mesa/22.3.2 mesa-glu/9.0.2 

nvcc -o boids_simulation opengl_naive.cu -I$GLEW_ROOT/include -I$MESA_GLU_ROOT/include -I$MESA_ROOT/include -I$FREEGLUT_ROOT/include -L$GLEW_ROOT/lib64 -L$MESA_GLU_ROOT/lib -L$MESA_ROOT/lib -L$FREEGLUT_ROOT/lib64 -lglut -lGL -lGLEW -lGLU -lcudart -O2
