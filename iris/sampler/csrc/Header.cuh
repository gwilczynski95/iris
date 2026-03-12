#pragma once

#include "constants.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

// *************************************************************************************************

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *************************************************************************************************

struct SLaunchParams_Sample {
	float3 *O;
	float3 *v;
	int max_Gaussians_per_ray;
	OptixTraversableHandle AS;
	float *t_hit;
	float *delta;
	int *indices;
	float chi_square_squared_radius;
	float max_R;
};

// *************************************************************************************************

struct SLaunchParams {
	union {
		SLaunchParams_Sample Sample;
	};
};