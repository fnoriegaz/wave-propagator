#include<iostream>
#include<vector>
#include<cuda.h>
#include<cuda_runtime.h>


__constant__ float fdtd_coeff[4]={1225./1024, 245./3072, 49./5120, 5./7168};


__global__
void compute_dPdt(float *P1, float *P_1, float *Vx, float *Vy, float *vel, float *rho,
				  int width, int depth, float dx, float dt){

	int ix = threadIdx.x + threadIdx.x * blockIdx.x;
	int iy = threadIdx.y + threadIdx.y * blockIdx.y;
	int tid = ix + iy * width;

	if(ix > 3 && ix < (width - 3) && iy > 3 && iy < (depth - 3)){
		P1[tid] += -1 * dt * dt * vel[tid] * vel[tid] * rho[tid] * (
			fdtd_coeff[0] * (Vx[tid] - Vx[tid-1] + Vy[tid] - Vy[tid-1*width]) -
			fdtd_coeff[1] * (Vx[tid+1] - Vx[tid-2] + Vy[tid+1*width] - Vy[tid-2*width]) +
			fdtd_coeff[2] * (Vx[tid+2] - Vx[tid-3] + Vy[tid+2*width] - Vy[tid-3*width]) -
			fdtd_coeff[3] * (Vx[tid+3] - Vx[tid-4] + Vy[tid+3*width] - Vy[tid-4*width]) ) / dx;
	}
}


int main(){

	int model_width = 512;
	int model_depth = 512;

	float dx = 12.5;
	float dy = 12.5;
	float dt = 1e-3;
	float total_time = 3.0;
	int time_samples = total_time / dt;

	std::vector<float> vel_model_h = std::vector<float>(model_width * model_depth);
	std::vector<float> rho_model_h = std::vector<float>(model_width * model_depth);

	float *vel_model_d, *rho_model_d;
	float *P1, *P_1, *dP_dx, *dP_dy, *Vx, *Vy, *dVx_dt, *dVy_dt, *dVx_dx, *dVy_dy;
	cudaMalloc(&vel_model_d, model_width * model_depth * sizeof(float));
	cudaMalloc(&rho_model_d, model_width * model_depth * sizeof(float));
	cudaMalloc(&P1, model_width * model_depth * sizeof(float));
	cudaMalloc(&Vx, model_width * model_depth * sizeof(float));
	cudaMalloc(&Vy, model_width * model_depth * sizeof(float));
	cudaMalloc(&P_1, model_width * model_depth * sizeof(float));
	cudaMalloc(&dP_dx, model_width * model_depth * sizeof(float));
	cudaMalloc(&dP_dy, model_width * model_depth * sizeof(float));
	cudaMalloc(&dVx_dt, model_width * model_depth * sizeof(float));
	cudaMalloc(&dVx_dx, model_width * model_depth * sizeof(float));
	cudaMalloc(&dVy_dt, model_width * model_depth * sizeof(float));
	cudaMalloc(&dVy_dy, model_width * model_depth * sizeof(float));

	for(int c=0;c<model_width*model_depth;c++){
		vel_model_h[c] = 1500.;
		rho_model_h[c] = 2600.;
	}

	cudaMemcpy(vel_model_d, vel_model_h.data(), model_width * model_depth * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(rho_model_d, rho_model_h.data(), model_width * model_depth * sizeof(float), cudaMemcpyHostToDevice);



	return 0;
}
