# Joint SPX VIX and VXX calibration with two OU factors and one CIR

Python implementation using a generator-closed level-three signature enrichment.

## Files

- functions_level3_enrichment_CIR.py: simulation, signatures, pricing, calibration and plotting functions.
- main_level3_enrichment_CIR.py: model configuration and constrained joint calibration.
- validation_level3_enrichment_CIR.py: call-option pricing on independent Monte Carlo paths and implied-volatility plots.
- validation_level3_enrichment_CIR_synth.py: synthetic-data validation with optimal parameters already included.
- market_data_private_CIR_synth.py: bundled synthetic quotes used by the synthetic validation script.
- market_data_template_CIR.py: empty template for local market inputs.

## Synthetic-data validation

`validation_level3_enrichment_CIR_synth.py` already contains optimal parameters (`ell_fixed` and `rho_b_fixed`) calibrated to `market_data_private_CIR_synth.py`. Keep both files beside `functions_level3_enrichment_CIR.py`. After installing the dependencies, run the example directly from the CIR directory:

```bash
python validation_level3_enrichment_CIR_synth.py
```

No preliminary calibration or manual parameter copying is required for this synthetic example. Control variates are disabled (`use_price_control_variates = False`). The standard validation instructions below apply when calibrating to other data.

## Local market inputs

Copy market_data_template_CIR.py to market_data_private_CIR.py and fill in the valuation and previous-future expiry dates, option and futures maturities, spot prices, interest rate, strikes, and call and put bid-ask arrays. Maturities are year fractions on an ACT/365 basis. Quote arrays for each asset must have matching lengths and strike ordering.

Keep market_data_private_CIR.py beside the model scripts on your computer. This file is excluded by .gitignore. When uploading through a browser, select only the public files in this package.

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

The model retains two OU factors and one CIR factor, the generator-closed time/CIR and slow-OU cubic enrichment, positive-part SPX variance, the absolute VIX option proxy, first-hit stopping of the signed conditional VIX futures, and the supplied VXX roll and fee conventions. The constrained solver retains its active-set stages and complete sampled variance and signed-VIX checks, and enforces that every required future starts above its stopping level.

## VIX proxy and futures stopping

VIX option payoffs use the absolute linear signature proxy: 100 * abs(ell dot signature). The same vix_option_proxy function is used in calibration and validation.

Futures use the signed conditional linear signature proxy. Each contract is stopped at the first simulation-grid value at or below 100 * future_floor and is then frozen at that level. Its history starts at valuation time, including before it enters the VXX roll; changing its role in the roll does not reset its stopping history. No absolute value is applied to futures.

The main and validation scripts both set future_floor = 1e-2, matching OU3D_enriched. This is a decimal VIX level and corresponds to 1 VIX index point. Set the same value in both scripts when changing it. Required futures must start strictly above this level; invalid initial values are rejected instead of being clipped.

## Validation plots

Copy ell_optimal and rho_b_optimal from the current calibration into ell_fixed and rho_b_fixed in the validation script. The included arrays are the explicit manual coefficients from the attached CIR source; they are not the result of a new calibration. Keep the factor parameters, signature basis, market inputs and pricing conventions consistent between the two scripts.

```bash
python validation_level3_enrichment_CIR.py
```

The validation script displays the three call implied-volatility smiles. total_paths and batch_size control the sample and memory usage. It simulates through the last option maturity, uses all simulated paths, and applies the original payoff formulas.

[Back to both models](../README.md)
