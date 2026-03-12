import numpy as np;
import torch;
from optix_sampler import CPyOptiXIrisRenderer;

# ### ### ### ### ###

def GenerateRays(
    R, D, F,
    width, height,
    fov_X, fov_Y
):
    double_tan_half_fov_X = 2.0 * np.tan(0.5 * fov_X);
    double_tan_half_fov_Y = 2.0 * np.tan(0.5 * fov_Y);

    indices = torch.arange(height * width, dtype=torch.float32, device="cuda");
    indices = indices.unsqueeze(1);
    y = indices // width;
    x = indices % width;
    
    d_x = (-0.5 + ((x + 0.5) / width)) * double_tan_half_fov_X;
    d_y = (-0.5 + ((y + 0.5) / height)) * double_tan_half_fov_Y;
    d_z = 1.0;

    R = R.expand(height * width, -1);
    D = D.expand(height * width, -1);
    F = F.expand(height * width, -1);
    
    v = (R * d_x) + (D * d_y) + (F * d_z);
    
    return v;

# ### ### ### ### ###

renderer = CPyOptiXIrisRenderer(11.3449);

number_of_Gaussians = 1000000;

m = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32, device="cuda") + (torch.rand(number_of_Gaussians, 3, dtype=torch.float32, device="cuda") * 2.0);
s = torch.Tensor.repeat(torch.tensor([0.05, 0.025, 0.01], dtype=torch.float32, device="cuda"), number_of_Gaussians, 1);
q = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device="cuda") + (torch.rand(number_of_Gaussians, 4, dtype=torch.float32, device="cuda") * 2.0);

renderer.SetGeometry(m, s, q);

R = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda");
D = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda");
F = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda");
width = 800;
height = 800;
FOV = np.pi / 2;

v = GenerateRays(
    R, D, F,
    width, height,
    FOV, FOV
);

O = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device="cuda");
O = O.unsqueeze(0).repeat(height * width, 1);
max_Gaussians_per_ray = 40;

t_samples, delta, indices_samples = renderer.Sample(O, v, max_Gaussians_per_ray);

print(t_samples);
print(delta);
print(indices_samples);
