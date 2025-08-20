#include<iostream>
#include<vector>
#include<source_location>
#include<fstream>
#include<cuda.h>
#include<cuda_runtime.h>


__constant__ float fdtd_coeff[4]={1225./1024, 245./3072, 49./5120, 5./7168};


void CHECK_CALL(){
	cudaError err = cudaGetLastError();
	if(err  != cudaSuccess){
		//std::source_location location = std::source_location::current();
		std::cout << "Cuda error: " << err << std::endl;
			/*":"
			<< location.file_name() << ":"
			<< location.line() << std::endl;*/
	}
}


__global__
void kernel_dPdt(float *P1, float *P_1, float *Vx, float *Vy, float *vel, float *rho,
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


__global__
void kernel_dVdt(float *P, float *Vx, float *Vy, float *rho,
				  int width, int depth, float dx, float dy, float dt){

	int ix = threadIdx.x + threadIdx.x * blockIdx.x;
	int iy = threadIdx.y + threadIdx.y * blockIdx.y;
	int tid = ix + iy * width;

	if(ix > 2 && ix < (width - 4) && iy < depth){
		Vx[tid] += -1 * (2. / (rho[tid] + rho[tid+1])) * dt * (
			fdtd_coeff[0] * (P[tid+1] - P[tid]) - 
			fdtd_coeff[1] * (P[tid+2] - P[tid-1]) +
			fdtd_coeff[2] * (P[tid+3] - P[tid-2]) -
			fdtd_coeff[3] * (P[tid+4] - P[tid-3]) ) / dx;
	}
	
	if(iy > 2 && iy < (depth - 4) && ix < width){
		Vy[tid] += -1 * (2. / (rho[tid] + rho[tid+1*width])) * dt * (
			fdtd_coeff[0] * (P[tid+1*width] - P[tid]) - 
			fdtd_coeff[1] * (P[tid+2*width] - P[tid-1*width]) +
			fdtd_coeff[2] * (P[tid+3*width] - P[tid-2*width]) -
			fdtd_coeff[3] * (P[tid+4*width] - P[tid-3*width]) ) / dy;
	}
}


__global__
void kernel_add_source(float *P, float *source, int time_sample, int sloc_x, int sloc_y,
					   int width){
	int ix = threadIdx.x;
	P[ix + sloc_x + width * sloc_y] += 1.;
}


void save_wavefield(float *P_h, float *P_d, int width, int depth, int it){

	cudaMemcpy(P_h, P_d, width * depth * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream out_stream("wavefield.bin", std::ios::binary);
	out_stream.seekp(it*width*depth*sizeof(float));

}


void propagate(float *P1, float *P_1, float *Vx, float *Vy, float *rho, float *vel,
			   float dx, float dy, float dt, int width, int depth, int time_samples){

	dim3 block_size(16,16);
	dim3 grid_size(width/16+1, depth/16+1);

	for(int c=0;c<time_samples;c++){
		std::cout << "iteration: " << c << std::endl;
		kernel_dVdt<<<grid_size, block_size>>>(P1,Vx,Vy,rho,width,depth,dx,dy,dt);
		CHECK_CALL();
		kernel_dPdt<<<grid_size, block_size>>>(P1,P_1,Vx,Vy,vel,rho,width,depth,dx,dt);
		CHECK_CALL();
		kernel_add_source<<<1,1>>>(P1,P1,c,width/2,depth/2,width);
		CHECK_CALL();
	}
}


int main(){

	int model_width = 1024;
	int model_depth = 1024;

	float dx = 12.5;
	float dy = 12.5;
	float dt = 1e-3;
	float total_time = 3.0;
	int time_samples = total_time / dt + 1;

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

	propagate(P1,P_1,Vx,Vy,rho_model_d,vel_model_d,dx,dy,dt,model_width,model_depth,time_samples);

	cudaFree(vel_model_d);
	cudaFree(rho_model_d);
	cudaFree(P1);
	cudaFree(Vx);
	cudaFree(Vy);
	cudaFree(P_1);
	cudaFree(dP_dx);
	cudaFree(dP_dy);
	cudaFree(dVx_dt);
	cudaFree(dVx_dx);
	cudaFree(dVy_dt);
	cudaFree(dVy_dy);

	return 0;
}
