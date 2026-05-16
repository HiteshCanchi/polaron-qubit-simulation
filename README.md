# Variational Study of Polaron Effects in GaAs Quantum Dots

## Overview
This project implements the **Lee-Low-Pines-Huybrechts (LLPH)** variational method to simulate energy level spacing and decoherence times for a qubit in a GaAs semiconductor quantum dot. The work is based on the theoretical framework of *Physica B: Condensed Matter*, with critical mathematical and physical corrections applied to the original publication's results.

## Key Features
- **Deterministic Integration:** Replaced stochastic methods with a 2D Simpson's rule grid for numerical stability.
- **Variational Optimization:** Utilizes the `scipy.optimize` L-BFGS-B algorithm to find optimal wavefunction parameters.
- **Physics Corrections:** Verified and corrected dimensional inconsistencies in the spontaneous emission formulas.

## Verified Corrections
During the reproduction phase, the following discrepancies in the source paper were identified and corrected:
1. **Oscillation Period Scale:** Identified a missing 4π² factor in the author's calculations, correcting the scale from ~1.6 fs to the physically accurate ~64 fs.
2. **Decoherence Rate Formula:** Found the published decoherence formula to be dimensionally invalid. Applied **Fermi's Golden Rule** with proper dielectric constants, shifting the decoherence time from an unrealistic ~18 ns to ~1000 ns.

## Results
The simulation generates high-resolution plots for:
- **Figure 1:** Energy Level Spacing vs. Confinement Length.
- **Figure 2:** Period of Oscillation vs. Confinement Length.
- **Figure 3:** Decoherence Time vs. Magnetic Field.

## Requirements
```bash
pip install numpy scipy matplotlib

```

## Usage

```bash
python src/main_parabolic.py

```

## Future Work

* Implementation of **Gaussian Confinement Potentials** to model finite-well depth and electron escape thresholds.
