# A joint signature framework for SPX, VIX and VXX

Python implementations for the joint pricing and calibration of SPX, VIX and VXX options using path signatures.

## Models

- [OU3D enriched](OU3D_enriched/): three Ornstein-Uhlenbeck factors with a level-three signature enrichment.
- [OU-CIR](CIR/): two Ornstein-Uhlenbeck factors and one CIR factor with a generator-closed level-three signature enrichment.

Each model has its own shared functions, calibration script, standard and synthetic-data validation scripts, market-input files and detailed README. Keep each model's files together in its folder.

## Files

The OU3D_enriched folder contains:

- functions_level3_enrichment.py
- main_level3_enrichment.py
- validation_level3_enrichment.py
- validation_level3_enrichment_synth.py: validation with optimal parameters already provided for the bundled synthetic data.
- market_data_private_synth.py: bundled synthetic quotes.
- market_data_template.py

The CIR folder contains:

- functions_level3_enrichment_CIR.py
- main_level3_enrichment_CIR.py
- validation_level3_enrichment_CIR.py
- validation_level3_enrichment_CIR_synth.py: validation with optimal parameters already provided for the bundled synthetic data.
- market_data_private_CIR_synth.py: bundled synthetic quotes.
- market_data_template_CIR.py

The root requirements.txt installs the dependencies for both implementations. Each model also retains its own requirements.txt.

## Installation

From the repository root, install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Synthetic-data examples

The validation scripts ending in `_synth.py` already contain optimal parameters (`ell_fixed` and `rho_b_fixed`) calibrated to the corresponding bundled synthetic data. No preliminary calibration or manual parameter copying is required to run these examples.

| Model folder | Validation script | Synthetic data module |
| --- | --- | --- |
| `OU3D_enriched` | `validation_level3_enrichment_synth.py` | `market_data_private_synth.py` |
| `CIR` | `validation_level3_enrichment_CIR_synth.py` | `market_data_private_CIR_synth.py` |

From `OU3D_enriched`, run:

```bash
python validation_level3_enrichment_synth.py
```

From `CIR`, run:

```bash
python validation_level3_enrichment_CIR_synth.py
```

Keep the supplied model settings and synthetic quotes together with their pre-calibrated parameters. Control variates are disabled in both examples (`use_price_control_variates = False`). Monte Carlo results vary between runs because `simulation_seed = None`.

## Private market inputs

Real market quotes are not included. Synthetic example quotes are bundled with the repository. To calibrate a model to your own market data, provide the spot prices, interest rate, strikes, and call and put bid-ask arrays using its empty template.

For OU3D_enriched:

1. Fill in the market inputs in market_data_template.py
2. Match the valuation date, option maturities and futures expiries in both the main and validation scripts to your snapshot.

For CIR:

1. Fill in the market inputs in market_data_template_CIR.py.
2. Match the valuation date, option maturities and futures expiries in both the main and validation scripts to your snapshot. Times are year fractions on an ACT/365 basis.


## OU3D enriched: calibration and evaluation

Open a terminal in OU3D_enriched and run:

```bash
python main_level3_enrichment.py
```

Copy the printed ell_optimal and rho_b_optimal into ell_fixed and rho_b_fixed in validation_level3_enrichment.py. Keep the model parameters and market settings consistent, then run:

```bash
python validation_level3_enrichment.py
```

See the [OU3D instructions](OU3D_enriched/README.md) for the configuration and simulation settings.

## OU-CIR: calibration and evaluation

Open a terminal in CIR and run:

```bash
python main_level3_enrichment_CIR.py
```

Copy the printed ell_optimal and rho_b_optimal into ell_fixed and rho_b_fixed in validation_level3_enrichment_CIR.py. Keep the model parameters and market settings consistent, then run:

```bash
python validation_level3_enrichment_CIR.py
```

See the [CIR instructions](CIR/README.md) for the configuration, constrained calibration and local feature cache.

## Numerical conventions

Both implementations use the absolute linear signature proxy for VIX option payoffs. VIX futures use the signed conditional linear signature proxy and are frozen at the floor after their first grid hit. Each contract's stopping history starts at valuation time and is shared across roll intervals, including before activation in VXX.

Both models set future_floor = 1e-2 in calibration and validation: the stopping level is 1 VIX index point. Required futures start strictly above the floor. Keep the same floor in both scripts when changing it.
