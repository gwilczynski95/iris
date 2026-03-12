#pragma once

// *** *** *** *** ***

#include "Header.cuh"

// *** *** *** *** ***

class CPyOptiXIrisRenderer {
	public:
		CPyOptiXIrisRenderer(float chi_square_squared_radius, const char *ptx_path);

		void SetGeometry(const float *m, const float *s, const float *q, int number_of_Gaussians);

		void Sample(
			const float *O, const float *v, int max_Gaussians_per_ray,
			int number_of_rays,
			float *t_hit, float *delta, int *indices
		);

	private:
		OptixDeviceContext optixContext;
			
		OptixModule module_Sample;
			
		OptixProgramGroup missPG;
		OptixProgramGroup raygenPG_Sample;
		OptixProgramGroup hitgroupPG_Sample;

		OptixPipeline pipeline;
		OptixShaderBindingTable *sbt_Sample;

		void *missRecordsBuffer;
		void *raygenRecordsBuffer_Sample;
		void *hitgroupRecordsBuffer_Sample;

		float3 *Gaussian_as_icosahedron_vertices;
		int3 *Gaussian_as_icosahedron_indices;
		OptixTraversableHandle GAS;

		void *GASBuffer;
		void *instancesBuffer;
		
		OptixTraversableHandle IAS;

		void *IASBuffer;
		float chi_square_squared_radius;
		void *launchParamsBuffer;
		float max_R;

		// *** *** *** *** ***

		void SetGeometry_CUDA(
			float *m, float *s, float *q,
			int number_of_Gaussians
		);

		void Sample_CUDA(
			float *O, float *v, int max_Gaussians_per_ray,
			int number_of_rays,
			float *t_hit, float *delta, int *indices
		);
};

// *** *** *** *** ***

extern "C" {
	CPyOptiXIrisRenderer *CreateRenderer(float chi_square_squared_radius, const char *ptx_path);
	int DestroyRenderer(CPyOptiXIrisRenderer *renderer);
	int SetGeometry(CPyOptiXIrisRenderer *renderer, const float *m, const float *s, const float *q, int number_of_Gaussians);
	int Sample(
		CPyOptiXIrisRenderer *renderer,
		const float *O, const float *v, int max_Gaussians_per_ray,
		int number_of_rays,
		float *t_hit, float *delta, int *indices
	);
}
