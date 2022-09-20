#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define I2D(ni,i,j) (((ni)*(j)) + i)

// arrays //
float *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8;
float *tmpf0,*tmpf1,*tmpf2,*tmpf3,*tmpf4,*tmpf5,*tmpf6,*tmpf7,*tmpf8;
float *u,*v;
int *solid;

// scalars //
float tau,faceq1,faceq2,faceq3; 
float vxin, roout;
float width, height;
int ni,nj;

////////////////////////////////////////////////////////////////////////////////
// Lattice Boltzmann function prototypes //
void stream(void);
void collide(void);
void apply_BCs(void);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	int array_size_2d,totpoints,i,j,k,i0;
    clock_t start,end;
    char filename[20];
    const int max_step = 10000;
    FILE *fp_col;

	omp_set_num_threads(4);
	

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

    totpoints=ni*nj;    
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

    tmpf0 = (float *)malloc(array_size_2d);
    tmpf1 = (float *)malloc(array_size_2d);
    tmpf2 = (float *)malloc(array_size_2d);
    tmpf3 = (float *)malloc(array_size_2d);
    tmpf4 = (float *)malloc(array_size_2d);
    tmpf5 = (float *)malloc(array_size_2d);
    tmpf6 = (float *)malloc(array_size_2d);
    tmpf7 = (float *)malloc(array_size_2d);
    tmpf8 = (float *)malloc(array_size_2d);

	u = (float *)malloc(array_size_2d);
	v = (float *)malloc(array_size_2d);
    solid = malloc(ni*nj*sizeof(int));

    faceq1 = 4.f/9.f;
    faceq2 = 1.f/9.f;
    faceq3 = 1.f/36.f;

	#pragma omp parallel for  schedule(static) private(i,j,i0)
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

	start = clock();	
    for (k=0; k<max_step; k++)
    {
        stream();
        apply_BCs();
        collide();

    }
    end = clock();
    printf("CPU Comsuming Time=%f s\n", (double)(end - start) / CLOCKS_PER_SEC);
  
	sprintf(filename, "./%d.plt", k);
	fp_col = fopen(filename, "w");
	fprintf(fp_col, "VARIABLES = X, Y, u, v, solid\n");
	fprintf(fp_col, "ZONE I= %d , J= %d, F=POINT \n", ni, nj);

	for (j = 0; j < nj; j++) {
		for (i = 0; i < ni; i++) {
			i0 = I2D(ni, i, j);
			fprintf(fp_col, "%5d %5d %15.7f %15.7f %5d\n", i, j, u[i0], v[i0], solid[i0]);
		}
	}
	fclose(fp_col);

	system("pause");
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void stream(void)
{
    int i,j,im1,ip1,jm1,jp1,i0;

	#pragma omp parallel for  schedule(static) private(i,j,jm1,jp1,im1,ip1,i0)
    for (j=0; j<nj; j++) {
	for (i=1; i<ni; i++) {
		jm1 = j - 1;
		jp1 = j + 1;
		if (j == 0) jm1 = 0;
		if (j == (nj - 1)) jp1 = nj - 1;
		im1 = i-1;
		ip1 = i+1;
		if (i==0) im1=0;
		if (i==(ni-1)) ip1=ni-1;

		i0 = I2D(ni, i, j);
		tmpf1[i0] = f1[I2D(ni,im1,j)];
		tmpf2[i0] = f2[I2D(ni,i,jm1)];
		tmpf3[i0] = f3[I2D(ni,ip1,j)];
		tmpf4[i0] = f4[I2D(ni,i,jp1)];
		tmpf5[i0] = f5[I2D(ni,im1,jm1)];
		tmpf6[i0] = f6[I2D(ni,ip1,jm1)];
		tmpf7[i0] = f7[I2D(ni,ip1,jp1)];
		tmpf8[i0] = f8[I2D(ni,im1,jp1)];
	}
    }

	#pragma omp parallel for  schedule(static) private(i,j,i0)
    for (j=0; j<nj; j++) {
	for (i=1; i<ni; i++) {
	    i0 = I2D(ni,i,j);
	    f1[i0] = tmpf1[i0];
	    f2[i0] = tmpf2[i0];
	    f3[i0] = tmpf3[i0];
	    f4[i0] = tmpf4[i0];
	    f5[i0] = tmpf5[i0];
	    f6[i0] = tmpf6[i0];
	    f7[i0] = tmpf7[i0];
	    f8[i0] = tmpf8[i0];
	}
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void collide(void)
{
    int i,j,i0;
    float ro, rovx, rovy, vx, vy, v_sq_term;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq;
    float rtau, rtau1;

    rtau = 1.f/tau;
    rtau1 = 1.f - rtau;

	#pragma omp parallel for  schedule(static) private(i,j,i0,ro,rovx,rovy,vx,vy,v_sq_term,f0eq,f1eq,f2eq,f3eq,f4eq,f5eq,f6eq,f7eq,f8eq)
    for (j=0; j<nj; j++) {
	for (i=0; i<ni; i++) {

	    i0 = I2D(ni,i,j);

	    // Do the summations needed to evaluate the density and components of velocity
	    ro = f0[i0] + f1[i0] + f2[i0] + f3[i0] + f4[i0] + f5[i0] + f6[i0] + f7[i0] + f8[i0];
	    rovx = f1[i0] - f3[i0] + f5[i0] - f6[i0] - f7[i0] + f8[i0];
	    rovy = f2[i0] - f4[i0] + f5[i0] + f6[i0] - f7[i0] - f8[i0];
	    vx = rovx/ro;
	    vy = rovy/ro;
        u[i0] = vx;
        v[i0] = vy;
	    v_sq_term = 1.5f*(vx*vx + vy*vy);

	    // Evaluate the local equilibrium f values in all directions
	    f0eq = ro * faceq1 * (1.f - v_sq_term);
	    f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
	    f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
	    f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
	    f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
	    f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
	    f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
	    f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
	    f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

	    // Simulate collisions by "relaxing" toward the local equilibrium
	    f0[i0] = rtau1 * f0[i0] + rtau * f0eq;
	    f1[i0] = rtau1 * f1[i0] + rtau * f1eq;
	    f2[i0] = rtau1 * f2[i0] + rtau * f2eq;
	    f3[i0] = rtau1 * f3[i0] + rtau * f3eq;
	    f4[i0] = rtau1 * f4[i0] + rtau * f4eq;
	    f5[i0] = rtau1 * f5[i0] + rtau * f5eq;
	    f6[i0] = rtau1 * f6[i0] + rtau * f6eq;
	    f7[i0] = rtau1 * f7[i0] + rtau * f7eq;
	    f8[i0] = rtau1 * f8[i0] + rtau * f8eq;
	}
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void apply_BCs(void)
{
    int i,j,i2d,i2d2;
    float f1old,f2old,f3old,f4old,f5old,f6old,f7old,f8old,v_sq_term;

	#pragma omp parallel for  schedule(static) private(i,j,i2d,i2d2,f1old,f2old,f3old,f4old,f5old,f6old,f7old,f8old,v_sq_term)
	for (j = 0; j < nj; j++) {
	for (i = 0; i < ni; i++) {

		i2d = I2D(ni, i, j);
		// Solid BC: "bounce-back" //
		if (solid[i2d] == 0) {
			f1old = f1[i2d];
			f2old = f2[i2d];
			f3old = f3[i2d];
			f4old = f4[i2d];
			f5old = f5[i2d];
			f6old = f6[i2d];
			f7old = f7[i2d];
			f8old = f8[i2d];

			f1[i2d] = f3old;
			f2[i2d] = f4old;
			f3[i2d] = f1old;
			f4[i2d] = f2old;
			f5[i2d] = f7old;
			f6[i2d] = f8old;
			f7[i2d] = f5old;
			f8[i2d] = f6old;
		}

		// Inlet BC //
		if (i == 0) {
			v_sq_term = 1.5f*(vxin * vxin);
			f1[i2d] = roout * faceq2 * (1.f + 3.f*vxin + 3.f*v_sq_term);
			f5[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
			f8[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
		}

		// Exit BC //
		if (i == (ni - 1)) {
			i2d2 = i2d - 1;
			f3[i2d] = f3[i2d2];
			f6[i2d] = f6[i2d2];
			f7[i2d] = f7[i2d2];
		}

		// periodic BC //
		if (j == 0) {
			i2d2 = I2D(ni, i, nj - 1);
			f2[i2d] = f2[i2d2];
			f5[i2d] = f5[i2d2];
			f6[i2d] = f6[i2d2];
		}
		if (j == (nj - 1)) {
			i2d2 = i;
			f4[i2d] = f4[i2d2];
			f7[i2d] = f7[i2d2];
			f8[i2d] = f8[i2d2];
		}
	}
	}
}
