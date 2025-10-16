# DiffuXion

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

Interactive interface for microstructure generation and diffusion simulation in polycrystaline materials - powered by [MSUtils](https://github.com/DataAnalyticsEngineering/MSUtils) and [FANS](https://github.com/DataAnalyticsEngineering/FANS)

## Table of contents
- [Features](#features)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)

## Features

We provide an interactive easy-to-use interface to directly configure parameters, run simulations and visualize results. This includes:

- generating Voronoi tessellations and corresponding voxel-based microstructures with prescribed grain boundary thickness
- defining diffusion coefficients for bulk and transversely isotropic grain boundaries either directly or via Arrhenius parameters and temperature
- computing full-field solutions (concentration, gradient and flux fields) as well as effective diffusivities

Note that in-app visualization is limited to low resolution results. For high-resolution results and for advanced investigations we recommend viewing the results in [ParaView](https://www.paraview.org/) using the generated xdmf-file.

More detailed information is provided [below](#usage).

## Getting started

Clone this repository to your system:

```bash
git clone https://github.com/DataAnalyticsEngineering/DiffuXion
cd DiffuXion
```

We recommend using [pixi](https://pixi.sh/) to effortlessly set up an isolated environment with all required dependencies:

```bash
# Install pixi if not done already
curl -fsSL https://pixi.sh/install.sh | sh

# Create the environment with all dependencies
pixi install
```

In general you can open a pixi environment shell via `pixi shell`.


Start the interactive interface to configure, run and visualize your simulations:

```bash
pixi run start
```

This will execute `marimo run examples/run.py` in the background and should open a browser tab.

## Usage

... some more details coming soon ...

## Acknowledgements

Funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support of the Stuttgart Center for Simulation Science ([SimTech](https://www.simtech.uni-stuttgart.de/)).

## Contributors

- [Lena Scholz](https://github.com/strinner213)
- [Sanath Keshav](https://github.com/sanathkeshav)
