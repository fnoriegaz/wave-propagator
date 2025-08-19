#include<iostream>
#include<vector>

int main(){

	int model_width = 512;
	int model_depth = 512;

	float dx = 12.5;
	float dy = 12.5;
	float dt = 1e-3;
	float total_time = 3.0;
	int time_samples = total_time / dt;

	std::vector<float> vel_model = std::vector<float>(model_width * model_depth);
	std::vector<float> rho_model = std::vector<float>(model_width * model_depth);

	for(int c=0;c<model_width*model_depth;c++){
		vel_model[c] = 1500.;
		rho_model[c] = 2600.;
	}


	

	return 0;
}
