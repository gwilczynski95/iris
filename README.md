# IRIS: Intersection-aware Ray-based Implicit Editable Scenes


<p align="center">
  <img src="images/teaser.jpg" width="1000"/>
</p>

<p align="center">
  <img src="images/counter.gif" width="32%" />
  <img src="images/garden.gif" width="33%" />
  <img src="images/bicycle.gif" width="32%" />
</p>

# ⚙️ Installation

This project is developed as an extension for Nerfstudio. To get started, please install [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/2adcc380c6c846fe032b1fe55ad2c960e170a215) along with its dependencies. <br>

<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nerf.studio/_images/logo.png">
    <!-- /pypi-strip -->
    <img alt="nerfstudio" src="https://docs.nerf.studio/_images/logo.png" width="400">
    <!-- pypi-strip -->
    </picture>
    <!-- /pypi-strip -->
</p>

Then, install this repo with:

```bash
pip install -e .
ns-install-cli
```

🚀 This will install the package in editable mode and kick off the Nerfstudio CLI installer to get you all set up and ready to go! 🎉

> Note: If for some reason the method is not working right away for you it may mean tat you have to compile the OptiX code for your specific machine. In order to do so please refer to this [instruction](docs/installation.md).

# Data preparation
### Synthetic data
By default, our method supports the NeRF Synthetic format. If you want to use your own data you need to put `sparse_pc.ply` in the dataset folder. For synthetic data you can use point cloud generated with [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

### Real data
For real data please follow [Nerfstudio data format](https://docs.nerf.studio/quickstart/data_conventions.html). Like with synthetic data, you'll also need to place `sparse_pc.ply` in the dataset folder to initialize the network with a sparse point cloud.

# Training the network

Example train commands:
``` bash
# For nerf synthetic
ns-train iris --data <path_to_dataset>

# For MiP-NeRF but also other real data
ns-train iris_real --data <path_to_dataset>
```

# Evaluation

To evaluate trained model use this example command:
```
ns-eval \
--load_config <path_to_config_of_trained_model> \
--output-path <output_path_for_metrics_json> \
--render-output-path <output_path_for_renders>
```