#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>

#define TILE_I 16
#define TILE_J 16
#define I2D(ni,i,j) (((ni)*(j)) + i)

// arrays on host //
float *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8;
float *u,*v;
int *solid;

// arrays on device //
float *f0_data, *f1_data, *f2_data, *f3_data, *f4_data, *f5_data, *f6_data, *f7_data, *f8_data;
float *u_data,*v_data;
int *solid_data;

// textures on device //
texture<float, 2> f1_tex, f2_tex, f3_tex, f4_tex, f5_tex, f6_tex, f7_tex, f8_tex;

// CUDA arrays on device //
cudaArray *f1_array, *f2_array, *f3_array, *f4_array, *f5_array, *f6_array, *f7_array, *f8_array; 

// scalars //
float tau,faceq1,faceq2,faceq3; 
float vxin, roout;
float width, height;
int ni,nj;
size_t pitch;

// CUDA kernel prototypes //
__global__ void stream_kernel (int pitch, float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                               float *f7_data, float *f8_data);

__global__ void collide_kernel (int pitch, float tau, float faceq1, float faceq2, float faceq3,
                                float *f0_data, float *f1_data, float *f2_data,
                                float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                                float *f7_data, float *f8_data, float *plot_data);

__global__ void apply_BCs_kernel (int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data, int* solid_data);

// CUDA kernel C wrappers //
void stream(void);
void collide(void);
void apply_BCs(void);

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int array_size_2d,totpoints,i,j,k,i0;
    char filename[20];
    const int max_step = 10000;
    FILE *fp_col;
    cudaChannelFormatDesc desc;

	// time clock //
	cudaEvent_t start, stop;
	float time = 0.f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // parameter list //
    ni=420;
    nj=160;
    vxin=0.04;
    roout=1.0;
    tau=0.51;
    
    printf ("ni = %d\n", ni);
    printf ("nj = %d\n", nj);
    printf ("vxin = %f\n", vxin);
    printf ("roout = %f\n", roout);
    printf ("tau = %f\n", tau);

    // allocate memory on host //
    totpoints = ni*nj;    
    array_size_2d=ni*nj*sizeof(float);

    f0 = (float *)malloc(array_size_2d);
    f1 = (float *)malloc(array_size_2d);
    f2 = (float *)malloc(array_size_2d);
    f3 = (float *)malloc(array_size_2d);
    f4 = (float *)malloc(array_size_2d);
    f5 = (float *)malloc(array_size_2d);
    f6 = (float *)malloc(array_size_2d);
    f7 = (float *)malloc(array_size_2d);
    f8 = (float *)malloc(array_size_2d);

    u = (float *)malloc(array_size_2d);
    v = (float *)malloc(array_size_2d);
    solid = (int *)malloc(ni*nj*sizeof(int));

    // allocate memory on device //
    pitch = ni*sizeof(float);

    checkCudaErrors(cudaMallocPitch((void **)&f0_data, &pitch, sizeof(float)*ni, nj)); 
    checkCudaErrors(cudaMallocPitch((void **)&f1_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f2_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f3_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f4_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f5_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f6_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f7_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&f8_data, &pitch, sizeof(float)*ni, nj));

	checkCudaErrors(cudaMallocPitch((void **)&u_data, &pitch, sizeof(float)*ni, nj));
	checkCudaErrors(cudaMallocPitch((void **)&v_data, &pitch, sizeof(float)*ni, nj));
    checkCudaErrors(cudaMallocPitch((void **)&solid_data, &pitch, sizeof(int)*ni, nj));

    desc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&f1_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f2_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f3_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f4_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f5_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f6_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f7_array, &desc, ni, nj));
    checkCudaErrors(cudaMallocArray(&f8_array, &desc, ni, nj));

    // some coeffients //
    faceq1 = 4.f/9.f;
    faceq2 = 1.f/9.f;
    faceq3 = 1.f/36.f;

    // initiall equilibrium f //
	for (i = 0; i < ni; i++) {
	for (j = 0; j < nj; j++) {
		i0 = I2D(ni, i, j);
		f0[i0] = faceq1 * roout * (1.f - 1.5f*vxin*vxin);
		f1[i0] = faceq2 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f2[i0] = faceq2 * roout * (1.f - 1.5f*vxin*vxin);
		f3[i0] = faceq2 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f4[i0] = faceq2 * roout * (1.f - 1.5f*vxin*vxin);
		f5[i0] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f6[i0] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f7[i0] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f8[i0] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		if ((i - ni / 4)*(i - ni / 4) + (j - nj / 2)*(j - nj / 2) < nj*nj / 16) {
			solid[i0] = 0;
		}
		else {
			solid[i0] = 1;
		}
	}
	}

    // Transfer initial data from host to device //
    checkCudaErrors(cudaMemcpy2D((void *)f0_data, pitch, (void *)f0, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f1_data, pitch, (void *)f1, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f2_data, pitch, (void *)f2, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f3_data, pitch, (void *)f3, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f4_data, pitch, (void *)f4, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f5_data, pitch, (void *)f5, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f6_data, pitch, (void *)f6, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f7_data, pitch, (void *)f7, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)f8_data, pitch, (void *)f8, sizeof(float)*ni, sizeof(float)*ni, nj,
                                cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy2D((void *)u_data, pitch, (void *)u, sizeof(float)*ni, sizeof(float)*ni, nj,
								cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D((void *)v_data, pitch, (void *)v, sizeof(float)*ni, sizeof(float)*ni, nj,
								cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D((void *)solid_data, pitch, (void *)solid, sizeof(int)*ni, sizeof(int)*ni, nj,
                                cudaMemcpyHostToDevice));


    for (k=0; k<max_step; k++)
    {
        stream();
        apply_BCs();
        collide();
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("GPU Consuming Time = %f s\n",time/1000);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    

    checkCudaErrors(cudaMemcpy2D((void *)u, sizeof(float)*ni, (void *)u_data, pitch, sizeof(float)*ni, nj,
                                cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D((void *)v, sizeof(float)*ni, (void *)v_data, pitch, sizeof(float)*ni, nj,
                                cudaMemcpyDeviceToHost));

    sprintf(filename,"./%d.plt",k);
    fp_col = fopen(filename,"w");
    fprintf (fp_col,"VARIABLES = X, Y, u, v, solid\n");
    fprintf (fp_col,"ZONE I= %d , J= %d, F=POINT \n",ni,nj);
    
    for (j=0; j<nj; j++){
    for (i=0; i<ni; i++){
        i0 = I2D(ni,i,j);
        fprintf (fp_col,"%5d %5d %15.7f %15.7f %5d\n",i,j,u[i0],v[i0],solid[i0]);
    }
    }
    fclose(fp_col);

    cudaFree(f0_data);
	cudaFree(f1_data);
	cudaFree(f2_data);
	cudaFree(f3_data);
	cudaFree(f4_data);
	cudaFree(f5_data);
	cudaFree(f6_data);
	cudaFree(f7_data);
	cudaFree(f8_data);
	cudaFree(u_data);
	cudaFree(v_data);
	cudaFree(solid_data);

	cudaFree(f1_array);
	cudaFree(f2_array);
	cudaFree(f3_array);
	cudaFree(f4_array);
	cudaFree(f5_array);
	cudaFree(f6_array);
	cudaFree(f7_array);
	cudaFree(f8_array);

	system("pause");

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void stream_kernel (int pitch, float *f1_data, float *f2_data,
                            float *f3_data, float *f4_data, float *f5_data,
			                float *f6_data, float *f7_data, float *f8_data)
{
    int i, j, i2d;

    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    f1_data[i2d] = tex2D(f1_tex, (float) (i-1)  , (float) j);
    f2_data[i2d] = tex2D(f2_tex, (float) i      , (float) (j-1));
    f3_data[i2d] = tex2D(f3_tex, (float) (i+1)  , (float) j);
    f4_data[i2d] = tex2D(f4_tex, (float) i      , (float) (j+1));
    f5_data[i2d] = tex2D(f5_tex, (float) (i-1)  , (float) (j-1));
    f6_data[i2d] = tex2D(f6_tex, (float) (i+1)  , (float) (j-1));
    f7_data[i2d] = tex2D(f7_tex, (float) (i+1)  , (float) (j+1));
    f8_data[i2d] = tex2D(f8_tex, (float) (i-1)  , (float) (j+1));
}

// C wrapper //
void stream(void)
{
    // Device-to-device mem-copies to transfer data from linear memory (f1_data) to CUDA format memory (f1_array)
    checkCudaErrors(cudaMemcpy2DToArray(f1_array, 0, 0, (void *)f1_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f2_array, 0, 0, (void *)f2_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f3_array, 0, 0, (void *)f3_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f4_array, 0, 0, (void *)f4_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f5_array, 0, 0, (void *)f5_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f6_array, 0, 0, (void *)f6_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f7_array, 0, 0, (void *)f7_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2DToArray(f8_array, 0, 0, (void *)f8_data, pitch, sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice));

    // Bind cuda array to texture //
    f1_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f1_tex, f1_array));
    f2_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f2_tex, f2_array));
    f3_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f3_tex, f3_array));
    f4_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f4_tex, f4_array));
    f5_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f5_tex, f5_array));
    f6_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f6_tex, f6_array));
    f7_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f7_tex, f7_array));
    f8_tex.filterMode = cudaFilterModePoint;
    checkCudaErrors(cudaBindTextureToArray(f8_tex, f8_array));

    dim3 grid = dim3((ni+TILE_I-1)/TILE_I, (nj+TILE_J-1)/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    stream_kernel<<<grid, block>>>(pitch, f1_data, f2_data, f3_data, f4_data, f5_data, f6_data, f7_data, f8_data);
    getLastCudaError("stream failed.");

    checkCudaErrors(cudaUnbindTexture(f1_tex));
    checkCudaErrors(cudaUnbindTexture(f2_tex));
    checkCudaErrors(cudaUnbindTexture(f3_tex));
    checkCudaErrors(cudaUnbindTexture(f4_tex));
    checkCudaErrors(cudaUnbindTexture(f5_tex));
    checkCudaErrors(cudaUnbindTexture(f6_tex));
    checkCudaErrors(cudaUnbindTexture(f7_tex));
    checkCudaErrors(cudaUnbindTexture(f8_tex));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void collide_kernel (int pitch, float tau, float faceq1, float faceq2, float faceq3, float *u_data, float *v_data,
                                float *f0_data, float *f1_data, float *f2_data, float *f3_data, float *f4_data, 
								float *f5_data, float *f6_data, float *f7_data, float *f8_data)
{
    int i, j, i2d;
    float ro, vx, vy, v_sq_term, rtau, rtau1;
    float f0now, f1now, f2now, f3now, f4now, f5now, f6now, f7now, f8now;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq;
    
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    rtau = 1.f/tau;
    rtau1 = 1.f - rtau;    

    // Read all f's and store in registers //
    f0now = f0_data[i2d];
    f1now = f1_data[i2d];
    f2now = f2_data[i2d];
    f3now = f3_data[i2d];
    f4now = f4_data[i2d];
    f5now = f5_data[i2d];
    f6now = f6_data[i2d];
    f7now = f7_data[i2d];
    f8now = f8_data[i2d];

    // Macroscopic flow props //
    ro =  f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
    vx = (f1now - f3now + f5now - f6now - f7now + f8now)/ro;
    vy = (f2now - f4now + f5now + f6now - f7now - f8now)/ro;
	u_data[i2d] = vx;
	v_data[i2d] = vy;
	       
    // Calculate equilibrium f's //
    v_sq_term = 1.5f*(vx*vx + vy*vy);
    f0eq = ro * faceq1 * (1.f - v_sq_term);
    f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
    f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
    f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
    f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

    // Do collisions //
    f0_data[i2d] = rtau1 * f0now + rtau * f0eq;
    f1_data[i2d] = rtau1 * f1now + rtau * f1eq;
    f2_data[i2d] = rtau1 * f2now + rtau * f2eq;
    f3_data[i2d] = rtau1 * f3now + rtau * f3eq;
    f4_data[i2d] = rtau1 * f4now + rtau * f4eq;
    f5_data[i2d] = rtau1 * f5now + rtau * f5eq;
    f6_data[i2d] = rtau1 * f6now + rtau * f6eq;
    f7_data[i2d] = rtau1 * f7now + rtau * f7eq;
    f8_data[i2d] = rtau1 * f8now + rtau * f8eq;
}

// C wrapper //
void collide(void)
{
	dim3 grid = dim3((ni + TILE_I - 1) / TILE_I, (nj + TILE_J - 1) / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    collide_kernel<<<grid, block>>>(pitch, tau, faceq1, faceq2, faceq3, u_data, v_data,
                                    f0_data, f1_data, f2_data, f3_data, f4_data, f5_data, f6_data, f7_data, f8_data);
    
    getLastCudaError("collide failed.");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void apply_BCs_kernel (int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data,
				                  int* solid_data)
{
    int i, j, i2d, i2d2;
    float v_sq_term;
    float f1old, f2old, f3old, f4old, f5old, f6old, f7old, f8old;
    
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    // Solid BC: "bounce-back" //
    if (solid_data[i2d] == 0) {
      f1old = f1_data[i2d];
      f2old = f2_data[i2d];
      f3old = f3_data[i2d];
      f4old = f4_data[i2d];
      f5old = f5_data[i2d];
      f6old = f6_data[i2d];
      f7old = f7_data[i2d];
      f8old = f8_data[i2d];
      
      f1_data[i2d] = f3old;
      f2_data[i2d] = f4old;
      f3_data[i2d] = f1old;
      f4_data[i2d] = f2old;
      f5_data[i2d] = f7old;
      f6_data[i2d] = f8old;
      f7_data[i2d] = f5old;
      f8_data[i2d] = f6old;
    }

    // Inlet BC //
    if (i == 0) {
      v_sq_term = 1.5f*(vxin * vxin);
      f1_data[i2d] = roout * faceq2 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f5_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f8_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
    }
        
    // Exit BC //
    if (i == (ni-1)) {
      i2d2 = i2d - 1;
      f3_data[i2d] = f3_data[i2d2];
      f6_data[i2d] = f6_data[i2d2];
      f7_data[i2d] = f7_data[i2d2];
    }

    // periodic BC //
    if (j == 0 ) {
        i2d2 = i + (nj-1)*pitch/sizeof(float);
        f2_data[i2d] = f2_data[i2d2];
        f5_data[i2d] = f5_data[i2d2];
        f6_data[i2d] = f6_data[i2d2];
    }
    if (j == (nj-1)) {
        i2d2 = i;
        f4_data[i2d] = f4_data[i2d2];
        f7_data[i2d] = f7_data[i2d2];
        f8_data[i2d] = f8_data[i2d2];
    }
}

void apply_BCs(void)
{
	dim3 grid = dim3((ni + TILE_I - 1) / TILE_I, (nj + TILE_J - 1) / TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    apply_BCs_kernel<<<grid, block>>> (ni, nj, pitch, vxin, roout, faceq2, faceq3,
							f0_data, f1_data, f2_data, f3_data, f4_data, f5_data, f6_data, f7_data, f8_data, solid_data);
    
    getLastCudaError("apply_Periodic_BC failed.");
}
