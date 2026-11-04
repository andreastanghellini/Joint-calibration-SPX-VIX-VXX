# Joint SPX VIX and VXX calibration with two OU factors and one CIR

Python implementation using a generator-closed level-three signature enrichment.

## Files

- functions_level3_enrichment_CIR.py: simulation, signatures, pricing, calibration and plotting functions.
- main_level3_enrichment_CIR.py: model configuration and constrained joint calibration.
- validation_level3_enrichment_CIR.py: call-option pricing on independent Monte Carlo paths and implied-volatility plots.
- market_data_template_CIR.py: empty template for local market inputs.

## Local market inputs

Copy market_data_template_CIR.py to market_data_private_CIR.py and fill in the valuation and previous-future expiry dates, option and futures maturities, spot prices, interest rate, strikes, and call and put bid-ask arrays. Maturities are year fractions on an ACT/365 basis. Quote arrays for each asset must have matching lengths and strike ordering.

Keep market_data_private_CIR.py beside the three scripts on your computer. This file is excluded by .gitignore. When uploading through a browser, select only the public files in this package.

## Installation

Run the commands from the CIR directory. The scripts use Python 3. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The supplied NumPy signature implementation also supports execution without iisignature; tqdm controls optional progress bars.

## Calibration

Review the model parameters, Monte Carlo settings and optimizer options in the main script, then run:

```bash
python main_level3_enrichment_CIR.py
```

The model retains two OU factors and one CIR factor, the generator-closed time/CIR and slow-OU cubic enrichment, positive-part SPX variance, the absolute VIX option proxy, pointwise floors on the signed conditional VIX futures, and the supplied VXX roll and fee conventions. The constrained solver retains its active-set stages and complete sampled variance and signed-VIX checks.

Initial coefficients and loss normalizations are constructed in the main script. Automatic checkpoint loading and result archives have been removed. Calibration features use a separate run folder inside cache_CIR to avoid keeping the full feature tensor in RAM and reusing files still open in another session. The directory is ignored by Git and can be deleted after closing the running Python session.

## Out-of-sample plots

Copy ell_optimal and rho_b_optimal from the current calibration into ell_fixed and rho_b_fixed in the validation script. The included arrays are the explicit manual coefficients from the attached CIR source; they are not the result of a new calibration. Keep the factor parameters, signature basis, market inputs and pricing conventions consistent between the two scripts.

```bash
python validation_level3_enrichment_CIR.py
```

The validation script displays the three call implied-volatility smiles. total_paths and batch_size control the sample and memory usage. It simulates through the last option maturity, uses all simulated paths, and applies the original payoff formulas without forward rescaling or control variates. It writes no price files or diagnostic tables.

The configuration uses simulation_seed = None, so each execution draws fresh Monte Carlo paths. Independent simulation at the same market snapshot is distinct from testing the model on another market date. Using another snapshot requires updating the private inputs and recalibrating the coefficients.

[Back to both models](../README.md)
