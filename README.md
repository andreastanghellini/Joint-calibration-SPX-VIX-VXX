# A joint framework for SPX, VIX and VXX

This repository contains the Python implementation of a model which uses path signatures for the joint pricing and calibration of SPX, VIX and VXX options.

The model uses signatures of time-augmented Ornstein–Uhlenbeck factors to describe volatility dynamics within a common framework. The implementation includes Monte Carlo simulation, joint calibration to market option prices, and evaluation on independently simulated paths.

## Requirements

The code requires Python 3 and the following packages:

- NumPy
- SciPy
- Matplotlib
- iisignature
- tqdm

Install the dependencies using:

```bash
python -m pip install numpy scipy matplotlib iisignature tqdm
```

## Repository contents

functions_highL_fee_roll6d_fit_pubblicazione.py
Shared functions for simulation, signature calculations, pricing, calibration and diagnostics.

main_restart_highL_fee_roll6d_fit_pubblicazione.py
Model configuration, market quotes and joint calibration.

out_of_sample_highL_fee_roll6d_fit_pubblicazione.py
Pricing and diagnostics using fixed parameters and independently simulated paths.

Keep these three files in the same directory.


## Market data

The example market data are defined directly in the calibration and evaluation scripts as NumPy arrays. They include strikes and bid–ask quotes for SPX, VIX and VXX options, together with spot prices, maturities and the interest rate.

To analyse another market snapshot, update the corresponding inputs in both scripts. Keep valuation dates, option maturities, futures expiries and rolling conventions consistent.

## Running the calibration

Before running the calibration, review the configuration in the main script, including:

- Factor parameters and correlations.
- Signature coordinates used by the model.
- Market quotes and maturities.
- Number of Monte Carlo paths and time steps.
- Calibration weights and optimizer settings.
- Variance constraints, futures floor and VXX fee.

From the directory containing the scripts, run:

```bash
python main_restart_highL_fee_roll6d_fit_pubblicazione.py
```

The script prints the calibrated signature coefficients and correlation parameters, reports calibration diagnostics, plots the fitted smiles, and saves the calibration results to a NumPy archive.

## Evaluating the calibrated model

To evaluate a new calibration, copy the resulting `ell_optimal` and `rho_b_optimal` into `ell_fixed` and `rho_b_fixed` in the evaluation script.

Ensure that the signature basis, model parameters, market data and numerical conventions match those used for calibration.

Then run:

```bash
python out_of_sample_highL_fee_roll6d_fit_pubblicazione.py
```

The script evaluates option prices on newly simulated paths and reports Monte Carlo uncertainty and model diagnostics. Here, out-of-sample evaluation refers to an independent simulation sample.

Adjust `total_paths` and `batch_size` to control the simulation size and memory usage. Use `show_plots` and `save_results` to control plotting and result saving.

## Numerical reproducibility

The supplied configuration does not fix a random seed, so numerical results vary between runs. The number of simulated paths and the time discretization affect Monte Carlo accuracy and computational cost.
