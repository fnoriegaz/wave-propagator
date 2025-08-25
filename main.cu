#include<iostream>
#include<vector>
#include<source_location>
#include<fstream>
#include<iomanip>
#include<stdexcept>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>


#define PI 3.141592653589793


__constant__ float fdtd_coeff[4]={1225./1024, 245./3072, 49./5120, 5./7168};


#define CUDA_CHECK(call) do {                                           \
	cudaError_t err__ = (call);                                           \
	if (err__ != cudaSuccess) {                                           \
		fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
				cudaGetErrorString(err__));                                 \
		exit(1);                                                            \
	}                                                                     \
}while(0)


void writeVTI(const float *data, int width, int depth, 
		const std::string& filename, float dx = 1.0f, float dy = 1.0f, float dz = 1.0f){

	std::ofstream ofs(filename);
	if(!ofs){
		throw std::runtime_error("Could not open file for writing: " + filename);
	}

	int nx = width;
	int ny = depth;
	int nz = 1; // 2D slice, but stored as 3D grid with 1 layer

	ofs << R"(<?xml version="1.0"?>)" << "\n";
	ofs << R"(<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">)" << "\n";
	ofs << "  <ImageData WholeExtent=\"0 " << (nx-1) 
		<< " 0 " << (ny-1) 
		<< " 0 " << (nz-1) 
		<< "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
	ofs << "    <Piece Extent=\"0 " << (nx-1) 
		<< " 0 " << (ny-1) 
		<< " 0 " << (nz-1) << "\">\n";
	ofs << "      <PointData Scalars=\"field\">\n";
	ofs << "        <DataArray type=\"Float32\" Name=\"field\" format=\"ascii\">\n";

	ofs << std::fixed << std::setprecision(6);
	for (int j = 0; j < ny; ++j) {
		for (int i = 0; i < nx; ++i) {
			ofs << data[j * nx + i] << " ";
		}
		ofs << "\n";
	}

	ofs << "        </DataArray>\n";
	ofs << "      </PointData>\n";
	ofs << "      <CellData/>\n";
	ofs << "    </Piece>\n";
	ofs << "  </ImageData>\n";
	ofs << "</VTKFile>\n";
}


void CHECK_CALL(){
	cudaError err = cudaGetLastError();
	if(err  != cudaSuccess){
		std::cout << "Cuda error: " << err << std::endl;
	}
}


// compute vector a and b later needed by psi kernels
// same kernel can be used for both dimensions x and y
__global__
void kernel_cpml(float *ax, float *bx, float freq, float dx, float dt, float r, int cpml_width, 
		float courant_vel, int length){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	float lx = cpml_width * dx;
	float d0 = -3.0 * log(1./r) / (2. * lx * courant_vel);
	float alphax = 0;
	float sigma = 0.;
	float fx = 0.;

	if(ix < cpml_width){
		fx = (cpml_width - ix - 1) * dx;
		float s = fx / lx;
		alphax = PI * freq * (1. - s);
		sigma = d0 * s * s;
		ax[ix] = expf(-(sigma + alphax) * dt);
		bx[ix] = (sigma / (sigma + alphax)) * (ax[ix] - 1.);
	}

	if(ix >= cpml_width && ix <= (length - cpml_width - 1)){
		ax[ix] = 0.;
		bx[ix] = 0.;
	}

	if(ix > (length - cpml_width -1) &&  ix < length){
		fx = (ix - length + cpml_width) * dx;
		float s = fx / lx;
		alphax = PI * freq * (1. - s);
		sigma = d0 * s * s;
		ax[ix] = expf(-(sigma + alphax) * dt);
		bx[ix] = (sigma / (sigma + alphax)) * (ax[ix] - 1.);
	}
}


__global__
void kernel_psiv(float *psivx, float *psivy, float *Vx, float *Vy, float *ax, float *bx, float *ay, float *by,
				 float dx, float dy, int width, int depth, int cpml_width){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int tid = ix + iy * width;

	if( ((ix > 3 && ix < cpml_width) || (ix > (width - cpml_width) && ix < (width - 3) )) && iy < depth){
		psivx[tid] = ax[ix]*psivx[tid] + bx[ix]*(
			fdtd_coeff[0] * (Vx[tid] - Vx[tid-1]) -
			fdtd_coeff[1] * (Vx[tid+1] - Vx[tid-2]) +
			fdtd_coeff[2] * (Vx[tid+2] - Vx[tid-3]) -
			fdtd_coeff[3] * (Vx[tid+3] - Vx[tid-4])) / dx;
	}
	
	if( ((iy > 3 && iy < cpml_width) || (iy > (depth - cpml_width) && iy < (depth - 3) )) && ix < width){
		psivy[tid] = ay[iy]*psivy[tid] + by[iy]*(
			fdtd_coeff[0] * (Vy[tid] - Vy[tid-1*width]) -
			fdtd_coeff[1] * (Vy[tid+1*width] - Vy[tid-2*width]) +
			fdtd_coeff[2] * (Vy[tid+2*width] - Vy[tid-3*width]) -
			fdtd_coeff[3] * (Vy[tid+3*width] - Vy[tid-4*width])) / dy;
	}

}


__global__
void kernel_psi(float *psix, float *psiy, float *P, float *ax, float *bx, float *ay, float *by,
				float dx, float dy, int width, int depth, int cpml_width){
	
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int tid = ix + iy * width;

	if( ((ix > 2 && ix < cpml_width) || (ix > (width-cpml_width)  && ix < (width-4))) && iy < depth ){
		psix[tid] = ax[ix]*psix[tid] + bx[ix]*(
			fdtd_coeff[0] * (P[tid+1] - P[tid]) -
			fdtd_coeff[1] * (P[tid+2] - P[tid-1]) +
			fdtd_coeff[2] * (P[tid+3] - P[tid-2]) -
			fdtd_coeff[3] * (P[tid+4] - P[tid-3])) / dx;
	}

	if( ((iy > 2 && iy < cpml_width) || (iy > (depth-cpml_width)  && iy < (depth-4))) && ix < width ){
		psiy[tid] = ay[iy]*psiy[tid] + by[iy]*(
			fdtd_coeff[0] * (P[tid+1*width] - P[tid]) -
			fdtd_coeff[1] * (P[tid+2*width] - P[tid-1*width]) +
			fdtd_coeff[2] * (P[tid+3*width] - P[tid-2*width]) -
			fdtd_coeff[3] * (P[tid+4*width] - P[tid-3*width])) / dy;
	}

}


__global__
void kernel_dPdt(float *P1, float *P_1, float *Vx, float *Vy, float *vel, float *rho,
				  int width, int depth, float dx, float dy, float dt, float *psivx, float *psivy){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int tid = ix + iy * width;

	if(ix > 3 && ix < (width - 3) && iy > 3 && iy < (depth - 3)){

		float dvx_dx = (
			fdtd_coeff[0] * (Vx[tid] - Vx[tid-1]) -
			fdtd_coeff[1] * (Vx[tid+1] - Vx[tid-2]) +
			fdtd_coeff[2] * (Vx[tid+2] - Vx[tid-3]) -
			fdtd_coeff[3] * (Vx[tid+3] - Vx[tid-4]) ) / dx;
		float dvy_dy = (
			fdtd_coeff[0] * (Vy[tid] - Vy[tid-1*width]) -
			fdtd_coeff[1] * (Vy[tid+1*width] - Vy[tid-2*width]) +
			fdtd_coeff[2] * (Vy[tid+2*width] - Vy[tid-3*width]) -
			fdtd_coeff[3] * (Vy[tid+3*width] - Vy[tid-4*width]) ) / dy;

		P1[tid] += -1 * dt * vel[tid] * vel[tid] * rho[tid] *(dvx_dx + dvy_dy) + psivx[tid] + psivy[tid];
	}
}


__global__
void kernel_dVdt(float *P, float *Vx, float *Vy, float *rho,
				  int width, int depth, float dx, float dy, float dt, float *psix, float *psiy){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int tid = ix + iy * width;

	if(ix > 2 && ix < (width - 4) && iy < depth){
		Vx[tid] = Vx[tid] - (2. / (rho[tid] + rho[tid+1])) * dt * (
			fdtd_coeff[0] * (P[tid+1] - P[tid]) - 
			fdtd_coeff[1] * (P[tid+2] - P[tid-1]) +
			fdtd_coeff[2] * (P[tid+3] - P[tid-2]) -
			fdtd_coeff[3] * (P[tid+4] - P[tid-3]) ) / dx + psix[tid];
	}
	
	if(iy > 2 && iy < (depth - 4) && ix < width){
		Vy[tid] = Vy[tid] - (2. / (rho[tid] + rho[tid+1*width])) * dt * (
			fdtd_coeff[0] * (P[tid+1*width] - P[tid]) - 
			fdtd_coeff[1] * (P[tid+2*width] - P[tid-1*width]) +
			fdtd_coeff[2] * (P[tid+3*width] - P[tid-2*width]) -
			fdtd_coeff[3] * (P[tid+4*width] - P[tid-3*width]) ) / dy + psiy[tid];
	}
}


__global__
void kernel_add_source(float *P, float *source, int time_sample, int sloc_x, int sloc_y,
					   int width){
	P[sloc_x + width * sloc_y] += 10.;
}


void propagate(float *P1, float *P_1, float *Vx, float *Vy, float *rho, float *vel,
			   float dx, float dy, float dt, int width, int depth, int time_samples){

	int cpml_width = 24;
	dim3 block_size(16,16);
	dim3 grid_size((width+block_size.x-1)/block_size.x, (depth+block_size.y-1)/block_size.y);
	
	//float courant_velx = (dx / dt) * (1 / (sqrt(3)) * (1225./1024 + 245./3072 + 49./5120 + 5./7168));
	//float courant_vely = (dy / dt) * (1 / (sqrt(3)) * (1225./1024 + 245./3072 + 49./5120 + 5./7168));
	float courant_velx = 1500.;
	float courant_vely = 1500.;

	float r = 1e-4;
	float freq = 3;
	float *P_h = (float *)malloc(width*depth*sizeof(float));
	float *psivx, *psivy, *psix, *psiy, *ax, *bx, *ay, *by;
	cudaMalloc(&psivx, width*depth*sizeof(float));
	cudaMalloc(&psivy, width*depth*sizeof(float));
	cudaMalloc(&psix, width*depth*sizeof(float));
	cudaMalloc(&psiy, width*depth*sizeof(float));
	cudaMalloc(&ax, width*sizeof(float));
	cudaMalloc(&bx, width*sizeof(float));
	cudaMalloc(&ay, depth*sizeof(float));
	cudaMalloc(&by, depth*sizeof(float));

	cudaMemset(psix, 0, width*depth*sizeof(float));
	cudaMemset(psiy, 0, width*depth*sizeof(float));
	cudaMemset(psivx, 0, width*depth*sizeof(float));
	cudaMemset(psivy, 0, width*depth*sizeof(float));
	cudaMemset(ax, 0,width*sizeof(float));
	cudaMemset(bx, 0,width*sizeof(float));
	cudaMemset(ay, 0,depth*sizeof(float));
	cudaMemset(by, 0,depth*sizeof(float));

	kernel_cpml<<<(width+block_size.x-1)/block_size.x, block_size.x>>>(ax,bx,freq,dx,dt,r,cpml_width,
	courant_velx,width);
	kernel_cpml<<<(depth+block_size.y-1)/block_size.y, block_size.y>>>(ay,by,freq,dy,dt,r,cpml_width,
	courant_vely,depth);

	float *h_ax = (float *)malloc(width*sizeof(float));
	float *h_bx = (float *)malloc(width*sizeof(float));
	float *h_ay = (float *)malloc(depth*sizeof(float));
	float *h_by = (float *)malloc(depth*sizeof(float));
	cudaMemcpy(h_ax, ax, width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_bx, bx, width*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i=0;i<cpml_width;i++) printf("%f %f\n", h_ax[i], h_bx[i]);
	for (int i=width-cpml_width;i<width;i++) printf("%f %f\n", h_ax[i], h_bx[i]);

	for(int c=0;c<time_samples;c++){
		std::cout << "iteration: " << c << std::endl;
		kernel_psi<<<grid_size, block_size>>>(psix,psiy,P1,ax,bx,ay,by,dx,dy,width,depth,cpml_width);
		CHECK_CALL();
		CUDA_CHECK(cudaDeviceSynchronize());
		kernel_psiv<<<grid_size, block_size>>>(psivx,psivy,Vx,Vy,ax,bx,ay,by,dx,dy,width,depth,cpml_width);
		CHECK_CALL();
		CUDA_CHECK(cudaDeviceSynchronize());
		kernel_dVdt<<<grid_size, block_size>>>(P1,Vx,Vy,rho,width,depth,dx,dy,dt,psix,psiy);
		CHECK_CALL();
		CUDA_CHECK(cudaDeviceSynchronize());
		kernel_dPdt<<<grid_size, block_size>>>(P1,P_1,Vx,Vy,vel,rho,width,depth,dx,dy,dt,psivx,psivy);
		CHECK_CALL();
		CUDA_CHECK(cudaDeviceSynchronize());

		if(c==100){
			kernel_add_source<<<1,1>>>(P1,P1,c,width/2,depth/2,width);
			CHECK_CALL();
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		cudaMemcpy(P_h,P1, width*depth*sizeof(float), cudaMemcpyDeviceToHost);
		writeVTI(P_h,width,depth,"file_" + std::to_string(c) + ".vti",dx,dy);
	}

	free(P_h);
	cudaFree(psivx);
	cudaFree(psivy);
	cudaFree(psix);
	cudaFree(psiy);
	cudaFree(ax);
	cudaFree(bx);
	cudaFree(ay);
	cudaFree(by);
}


int main(){

	int model_width = 128;
	int model_depth = 128;

	float dx = 25;
	float dy = 25;
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

	cudaMemset(P1, 0, model_width*model_depth*sizeof(float));
	cudaMemset(P_1, 0, model_width*model_depth*sizeof(float));
	cudaMemset(Vx, 0, model_width*model_depth*sizeof(float));
	cudaMemset(Vy, 0, model_width*model_depth*sizeof(float));
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
