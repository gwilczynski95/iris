import ctypes
import os
from typing import Optional, Tuple

import torch


class CPyOptiXIrisRenderer:
    """ctypes wrapper around the OptiX renderer shared library."""

    def __init__(
        self,
        chi_square_squared_radius: float,
        max_gaussians_per_ray: int = 40,
        lib_name: str = "optix_sampler_core.so",
        ptx_name: str = "shaders_Sample.ptx",
    ) -> None:
        base_dir = os.path.dirname(__file__)

        self._lib_path = os.path.join(base_dir, lib_name)
        if not os.path.exists(self._lib_path):
            alt_lib_path = os.path.join(base_dir, "build", lib_name)
            if os.path.exists(alt_lib_path):
                self._lib_path = alt_lib_path
            else:
                raise FileNotFoundError(f"Could not find {lib_name} at {self._lib_path} or {alt_lib_path}. Build the library first.")

        self._ptx_path = os.path.join(base_dir, ptx_name)
        if not os.path.exists(self._ptx_path):
            alt_ptx_path = os.path.join(base_dir, "build", ptx_name)
            if os.path.exists(alt_ptx_path):
                self._ptx_path = alt_ptx_path
            else:
                raise FileNotFoundError(f"Could not find {ptx_name} at {self._ptx_path} or {alt_ptx_path}. Run build_optix.sh to generate it.")

        self.max_gaussians_per_ray = max_gaussians_per_ray
        self._lib = ctypes.CDLL(self._lib_path)

        self._lib.CreateRenderer.argtypes = [ctypes.c_float, ctypes.c_char_p]
        self._lib.CreateRenderer.restype = ctypes.c_void_p

        self._lib.DestroyRenderer.argtypes = [ctypes.c_void_p]
        self._lib.DestroyRenderer.restype = ctypes.c_int

        self._lib.SetGeometry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.SetGeometry.restype = ctypes.c_int

        self._lib.Sample.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.Sample.restype = ctypes.c_int

        self._renderer: Optional[int] = self._lib.CreateRenderer(
            ctypes.c_float(chi_square_squared_radius), self._ptx_path.encode("utf-8")
        )
        if not self._renderer:
            raise RuntimeError("Failed to create OptiX renderer. Ensure the PTX was built and CUDA/OptiX are available.")

    def __del__(self) -> None:
        try:
            if getattr(self, "_renderer", None):
                self._lib.DestroyRenderer(ctypes.c_void_p(self._renderer))
                self._renderer = None
        except Exception:
            # Avoid noisy destructor errors on interpreter shutdown
            pass

    @staticmethod
    def _ensure_cuda_float(t: torch.Tensor, name: str) -> torch.Tensor:
        if t.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA device")
        if t.dtype != torch.float32:
            t = t.float()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def SetGeometry(self, m: torch.Tensor, s: torch.Tensor, q: torch.Tensor) -> None:
        if self._renderer is None:
            raise RuntimeError("Renderer has not been created.")

        m = self._ensure_cuda_float(m, "m")
        s = self._ensure_cuda_float(s, "s")
        q = self._ensure_cuda_float(q, "q")

        number_of_gaussians = int(m.shape[0])
        status = self._lib.SetGeometry(
            ctypes.c_void_p(self._renderer),
            ctypes.c_void_p(m.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_int(number_of_gaussians),
        )
        if status != 0:
            raise RuntimeError(f"SetGeometry failed with status {status}.")

    def Sample(self, O: torch.Tensor, v: torch.Tensor, max_Gaussians_per_ray: Optional[int] = None, *_unused: object) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._renderer is None:
            raise RuntimeError("Renderer has not been created.")

        O = self._ensure_cuda_float(O, "O")
        v = self._ensure_cuda_float(v, "v")

        number_of_rays = int(O.shape[0])
        buffer_size = int(max_Gaussians_per_ray if max_Gaussians_per_ray is not None and not isinstance(max_Gaussians_per_ray, torch.Tensor) else self.max_gaussians_per_ray)

        t_hit = torch.empty((buffer_size, number_of_rays), device=O.device, dtype=torch.float32)
        delta = torch.empty_like(t_hit)
        indices = torch.empty((buffer_size, number_of_rays), device=O.device, dtype=torch.int32)

        status = self._lib.Sample(
            ctypes.c_void_p(self._renderer),
            ctypes.c_void_p(O.data_ptr()),
            ctypes.c_void_p(v.data_ptr()),
            ctypes.c_int(buffer_size),
            ctypes.c_int(number_of_rays),
            ctypes.c_void_p(t_hit.data_ptr()),
            ctypes.c_void_p(delta.data_ptr()),
            ctypes.c_void_p(indices.data_ptr()),
        )
        if status != 0:
            raise RuntimeError(f"Sample failed with status {status}.")

        return t_hit, delta, indices
