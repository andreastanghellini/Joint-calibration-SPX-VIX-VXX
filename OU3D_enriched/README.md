# Enriched three-factor OU model (OU3D)

Python implementation of a model which uses path signatures for the joint pricing and calibration of SPX, VIX and VXX options.

## Requirements

Python 3 and the dependencies listed in requirements.txt. Install them with:

```bash
python -m pip install -r requirements.txt
```

## Repository contents

- functions_level3_enrichment.py: shared simulation, signature, pricing and calibration functions.
- main_level3_enrichment.py: model configuration and joint calibration.
- validation_level3_enrichment.py: evaluation on independent Monte Carlo paths and implied-volatility plots.
- market_data_template.py: empty template for local market inputs.

Keep the scripts in the same directory and retain these filenames so their imports resolve correctly.

## Local market data

Market quotes are supplied through a local file named market_data_private.py. This file is excluded from version control by .gitignore.

To use your own data:

1. Copy market_data_template.py to market_data_private.py in the same directory as the scripts.
2. Fill in the spot prices, interest rate, strikes, and call and put bid-ask arrays for each asset. All quote arrays for an asset must match its strike array in length and order.
3. Keep market_data_private.py on your computer. When uploading through a browser, select only the public repository files.
4. Update the valuation date, option maturities and futures expiries in both scripts to match your market snapshot.

The imports in the scripts show which variable names are required. Model forwards and implied volatilities are computed from the local inputs. The empty template must be completed before either script can run.

## Calibration

Review the model parameters, signature basis, Monte Carlo settings, calibration weights and constraints in the main script. From the scripts directory, run:

```bash
python main_level3_enrichment.py
```

The calibrated coefficients and correlations are printed as ell_optimal and rho_b_optimal.

## Out-of-sample evaluation

Copy the calibration results into ell_fixed and rho_b_fixed in the out-of-sample script. Match the model settings, market inputs, maturities and numerical conventions to the calibration configuration, then run:

```bash
python validation_level3_enrichment.py
```

The script computes prices on independent Monte Carlo paths and displays the implied-volatility plots. Control variates are enabled by use_price_control_variates. The sample size and memory usage can be adjusted through total_paths and batch_size.

## Reproducibility

The supplied configuration uses simulation_seed = None, so Monte Carlo samples and numerical results vary between runs.

[Back to both models](../README.md)
