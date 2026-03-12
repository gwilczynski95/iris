#include "CPyOptiXIrisRenderer.h"

// *** *** *** *** ***

void CPyOptiXIrisRenderer::SetGeometry(const float *m, const float *s, const float *q, int number_of_Gaussians) {
	SetGeometry_CUDA(
		(float *)m, (float *)s, (float *)q,
		number_of_Gaussians
	);
}

// *** *** *** *** ***

void CPyOptiXIrisRenderer::Sample(
	const float *O, const float *v, int max_Gaussians_per_ray,
	int number_of_rays,
	float *t_hit, float *delta, int *indices
) {
	Sample_CUDA(
		(float *)O, (float *)v, max_Gaussians_per_ray,
		number_of_rays,
		t_hit, delta, indices
	);
}

// *** *** *** *** ***

extern "C" {
	CPyOptiXIrisRenderer *CreateRenderer(float chi_square_squared_radius, const char *ptx_path) {
		try {
			return new CPyOptiXIrisRenderer(chi_square_squared_radius, ptx_path);
		} catch (...) {
			return nullptr;
		}
	}

	int DestroyRenderer(CPyOptiXIrisRenderer *renderer) {
		try {
			delete renderer;
			return 0;
		} catch (...) {
			return -1;
		}
	}

	int SetGeometry(CPyOptiXIrisRenderer *renderer, const float *m, const float *s, const float *q, int number_of_Gaussians) {
		if (renderer == nullptr) {
			return -1;
		}
		try {
			renderer->SetGeometry(m, s, q, number_of_Gaussians);
			return 0;
		} catch (...) {
			return -1;
		}
	}

	int Sample(
		CPyOptiXIrisRenderer *renderer,
		const float *O, const float *v, int max_Gaussians_per_ray,
		int number_of_rays,
		float *t_hit, float *delta, int *indices
	) {
		if (renderer == nullptr) {
			return -1;
		}
		try {
			renderer->Sample(O, v, max_Gaussians_per_ray, number_of_rays, t_hit, delta, indices);
			return 0;
		} catch (...) {
			return -1;
		}
	}
}