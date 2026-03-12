# Advanced Installation

If you landed in here it measn that for some reaseon the code is not working for you as it should. This usually means that parts of our code that are written in cuda require compiling for your specific machine. We have prepared easy and hard option for manual compilaiton to choose from.

## Easy

Start from installing [OptiX SDK 9.0.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) in `iris/sampler` directory.
We have prepared a script that will run docker and compile everything for you. It solves the problems in most of the cases we encountered. To build the optix sampler simply run

```bash
bash build_sampler.sh
```

## Hard

Similarly to easy start from installing [OptiX SDK 9.0.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) in `iris/sampler` directory. Now you can run the compilation with the script:

```bash
bash build_optix.sh
```

You need all the dependencies for compiling c++ and cuda code on oyur machine already setup for the comand above to work. In the worst case you can tweak the compillation commands in the script. Often the paths to the libraries differ across the systems.