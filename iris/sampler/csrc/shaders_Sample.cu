#include "Header.cuh"

// *** *** *** *** ***

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *** *** *** *** ***

struct SRayPayload {
	float t_min;
	int number_of_hits;
	float3 data[HIT_BUFFER_SIZE];
	int hit_buffer_size;
};

// *** *** *** *** ***

extern "C" __global__ void __raygen__() {
	uint3 launch_index      = optixGetLaunchIndex();
	uint3 launch_dimensions = optixGetLaunchDimensions();

	int ray_ind        = launch_index.x;
	int number_of_rays = launch_dimensions.x;

	float3 O    = optixLaunchParams.Sample.O    [ray_ind];
	float3 v    = optixLaunchParams.Sample.v    [ray_ind];

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	float t_min = 0.0f;
	int element_ind = ray_ind;

	// *********************************************************************************************

	for (int i = 0; i < optixLaunchParams.Sample.max_Gaussians_per_ray; i += HIT_BUFFER_SIZE) {
		// !!! !!! !!!
		rp.number_of_hits = 0;
		// !!! !!! !!!

		int hit_buffer_size = min(HIT_BUFFER_SIZE, optixLaunchParams.Sample.max_Gaussians_per_ray - i);

		// *****************************************************************************************

		// !!! !!! !!!
		if (isfinite(t_min)) {
		// !!! !!! !!!
			rp.t_min = t_min;
			rp.hit_buffer_size = hit_buffer_size;

			optixTrace(
				optixLaunchParams.Sample.AS,
				O,
				v,
				t_min,
				INFINITY,
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
				0,
				1,
				0,

				rp_addr_lo,
				rp_addr_hi
			);
		}

		// *****************************************************************************************
		
		for (int j = 0; j < hit_buffer_size; ++j) {
			float3 tmp = rp.data[j];

			// !!! !!! !!!
			t_min         = ((j < rp.number_of_hits) ?                 tmp.x : INFINITY);
			float delta   = ((j < rp.number_of_hits) ?                 tmp.y : INFINITY);
			int Gauss_ind = ((j < rp.number_of_hits) ? __float_as_int(tmp.z) :       -1);
			// !!! !!! !!!

			optixLaunchParams.Sample.t_hit  [element_ind] = t_min;
			optixLaunchParams.Sample.delta  [element_ind] = delta;
			optixLaunchParams.Sample.indices[element_ind] = Gauss_ind;

			element_ind += number_of_rays;
		}

		// *****************************************************************************************

		// !!! !!! !!!
		t_min = nextafter(t_min, INFINITY);
		// !!! !!! !!!
	}
}

// *** *** *** *** ***

extern "C" __global__ void __anyhit__() {
	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();

	// *********************************************************************************************

	float O_dot_v = __fmaf_rn(O.x, v.x, __fmaf_rn(O.y, v.y, O.z * v.z));
	float v_dot_v = __fmaf_rn(v.x, v.x, __fmaf_rn(v.y, v.y, v.z * v.z));

	// !!! !!! !!!
	float v_dot_v_inv;
	asm volatile (
		"rcp.approx.ftz.f32 %0, %1;" :
		"=f"(v_dot_v_inv) :
		"f" (v_dot_v)
	);
	// !!! !!! !!!

	float t_hit = -O_dot_v * v_dot_v_inv;

	float3 O_perp = make_float3(
		__fmaf_rn(v.x, t_hit, O.x),
		__fmaf_rn(v.y, t_hit, O.y),
		__fmaf_rn(v.z, t_hit, O.z)
	);

	float O_perp_squared_norm = __fmaf_rn(O_perp.x, O_perp.x, __fmaf_rn(O_perp.y, O_perp.y, O_perp.z * O_perp.z));
	float delta = optixLaunchParams.Sample.chi_square_squared_radius - O_perp_squared_norm;

	// *********************************************************************************************

	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *********************************************************************************************

	if ((delta >= 0.0f) && (t_hit >= rp->t_min)) {
		unsigned Gauss_ind = optixGetInstanceIndex();
				
		// *****************************************************************************************

		// !!! !!! !!!
		asm volatile (
			"sqrt.approx.ftz.f32 %0, %1;" :
			"=f"(delta) :
			"f" (v_dot_v_inv * delta)
		);
		// !!! !!! !!!

		// *****************************************************************************************

		float3 tmp1 = make_float3(t_hit, delta, __int_as_float(Gauss_ind));
		float3 tmp2;

		for (int i = 0; i < rp->number_of_hits; ++i) {
			tmp2 = rp->data[i];

			if (tmp1.x < tmp2.x) {
				rp->data[i] = tmp1;
				tmp1 = tmp2;
			}
		}

		if (rp->number_of_hits < rp->hit_buffer_size) {
			rp->data[rp->number_of_hits++] = tmp1;
			optixIgnoreIntersection();
		} else {
			if (t_hit <= tmp2.x + optixLaunchParams.Sample.max_R) optixIgnoreIntersection();
		}
	} else
		optixIgnoreIntersection();
}