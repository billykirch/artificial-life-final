# Artificial Life — Ciliate Evolution

## Overview

This project uses Taichi, a high-performance numerical computation framework, to simulate soft-body dynamics of evolving shapes or creatures. The simulation and evaluation are split into two main Python files:

**evaluate.py**: Contains Taichi kernels and functions for running particle-based simulations, evaluating performance, and visualizing results, pulled from [this Taichi example repository](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py).

**processing.py** (main control script): Runs evolutionary cycles, mutating the shapes and evaluating their performance.

## evaluate.py

Its main purpose is to run the core physics simulation, calculate the loss (performance measure), and visualize particle-based soft-body movements.

**Initialization & Configuration**: 
Imports necessary modules and sets parameters like particle count, grid size (`n_grid`), time step (`dt`), and physical properties (`gravity`, elasticity `E`).

**Field Allocation** (`allocate_fields`): 
Allocates memory and field data structures needed by the Taichi framework for particles, grid velocities, actuators, and other simulation attributes.

**Kernel Definitions** (`@ti.kernel`):

- `clear_grid()`: Resets the grid state before each simulation step.

- `clear_particle_grad()` and `clear_actuation_grad()`: Clear gradients for optimization purposes.

- `p2g(f)`: Converts particle velocities and states to grid representations *accordint to sinusoidal waveforms*.

- `grid_op()`: Processes grid operations, including applying gravity and handling collisions.

- `g2p(f)`: Transfers data from the grid back to particles, updating their positions and velocities.

- `compute_actuation(t)`: Calculates actuation signals to drive actuator movements.

- `compute_x_avg()` and `loss`: Measures performance (e.g., how far the simulated shape moves).

**Visualization** (`visualize(s, folder)`): 
Displays the simulation state at step `s`, saving output images for visualization.

## processing.py

Its main purpose is to manage evolutionary algorithms to optimize creature shapes. It creates generations of shapes, evaluates them, mutates them, and tracks their performance.

**Evolution Configuration**: 
Parameters defining mutation strength, height, and number of cilia (appendages).

**Shape Creation** (`createCiliate(height, n)`): 
Generates initial soft-body configurations with a main body and appendages (cilia).

**Mutation Function** (`mutatedCiliate(height, n)`): 
Applies controlled mutations to shapes based on parameters, ensuring practical bounds.

**Main Evolution Loop**:

- Initializes a first-generation set of shapes.

- Iteratively evaluates shapes by calling `evaluate.py`, measures performance, and selects the best-performing individuals.

- Applies mutations to create new generations, striving for optimized performance.

## How to Run the Simulation

### Setup

Ensure you've installed a recent version of Python (3.8 or newer) as well as Taichi ('pip install taichi').

Also, **your system must support GPU computation** (Taichi configured to use GPU by default in the provided scripts).

### Execution

Run the evolutionary process by executing `processing.py`. This script automatically handles multiple iterations, invoking the simulation through evaluate.py:

`python processing.py`

Parameters for the initial generation (number of cilia, mutation strength, generations) can be adjusted within `processing.py`.

### Output

Performance results for each shape are printed and visualized.

The loss for each iteration for a ciliate is printed out so you may track progress while the simulation runs. After the entire simulation completes, all ciliates from every generation, as well as their respective performances, are printed as well.

The number of ciliates, their complexity, the number of learning iterations, and the evolutionary generations can all be adjusted in the code, but with the given parameters, **this simulation can easily take 30-45 minutes to run**.

Visual output for each simulation step is generated, allowing analysis of creature movement. Loss graphs (`losses-plot.png`) show performance across iterations.

## Running Simulation TLDR

Execute from a command line or terminal:

`python processing.py`

Both `evaluate.py` and `processing.py` must be in the same directory with the required permissions and dependencies to run GPU-accelerated simulations.

Adjusting mutation parameters in `processing.py` will influence evolutionary strategies, while the physics simulation parameters (`gravity`, `dt`, actuator strengths) in `evaluate.py` can be tuned for different simulation behaviors.

