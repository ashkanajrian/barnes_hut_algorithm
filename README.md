# 3D Barnes-Hut Galaxy Simulation

![Galaxy Simulation Demo](boodies.gif)

An optimized 3D implementation of the Barnes-Hut algorithm for simulating gravitational galaxy formation, developed as part of the "Simulation and Modeling of Natural Processes" course at University of Geneva.

## Key Features
- **3D galaxy simulation** with realistic gravitational interactions
- **Performance optimizations** including:
  - Numba-accelerated force calculations
  - Iterative tree traversal
  - Memory-efficient node structures
- **Visualization** using Matplotlib 3D

## Requirements
- Python 3.6+
- NumPy
- Matplotlib
- Numba (for acceleration)

## Usage
```bash
python barnes_hut_3d.py
```
Configuration options available in the script:
- Number of bodies
- Time step (dt)
- Simulation duration
- Visualization settings

## Academic Background
Developed by Ashkan Ajrian as course homework, based on Dr. Jonas Latt's original 2D implementation. Extended to 3D with performance optimizations.