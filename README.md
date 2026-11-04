# A joint signature framework for SPX, VIX and VXX

Python implementations for the joint pricing and calibration of SPX, VIX and VXX options using path signatures.

## Models

- [OU3D enriched](OU3D_enriched/): three Ornstein-Uhlenbeck factors with a level-three signature enrichment.
- [OU-CIR](CIR/): two Ornstein-Uhlenbeck factors and one CIR factor with a generator-closed level-three signature enrichment.

Each model has its own shared functions, calibration script, out-of-sample script, empty market-input template and detailed README. Keep each model's files together in its folder.

## Files

The OU3D_enriched folder contains:

- functions_level3_enrichment.py
- main_level3_enrichment.py
- validation_level3_enrichment.py
- market_data_template.py

The CIR folder contains:

- functions_level3_enrichment_CIR.py
- main_level3_enrichment_CIR.py
- validation_level3_enrichment_CIR.py
- market_data_template_CIR.py

The root requirements.txt installs the dependencies for both implementations. Each model also retains its own requirements.txt.

## Installation

From the repository root, install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Private market inputs

Market quotes are not included. To run a model, provide your own spot prices, interest rate, strikes, and call and put bid-ask arrays using its empty template.

For OU3D_enriched:

1. Copy market_data_template.py to market_data_private.py inside OU3D_enriched.
2. Fill in the market inputs.
3. Match the valuation date, option maturities and futures expiries in both the main and validation scripts to your snapshot.

For CIR:

1. Copy market_data_template_CIR.py to market_data_private_CIR.py inside CIR.
2. Fill in the market inputs, valuation and previous-future expiry dates, option maturities and futures maturities. Times are year fractions on an ACT/365 basis.

Keep both private files on your computer. The .gitignore rules exclude these filenames from Git. When uploading through GitHub's website, select only the public package contents; do not upload a working folder containing private inputs.

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

The implementations retain their respective pricing and calibration settings. In particular, the OU3D version stops and freezes a VIX futures path at its first floor hit; the CIR version applies its floor pointwise. OU3D validation enables price control variates, while CIR validation uses the supplied payoffs without control variates or forward rescaling.

## Reproducibility

The supplied scripts draw fresh Monte Carlo paths with simulation_seed = None. Numerical results vary between runs. After calibrating, update the validation coefficients manually. The coefficient arrays already present in the validation scripts are not a new calibration to the reader's inputs.

Validation uses independent paths at the selected market snapshot. Testing another market date requires the corresponding market inputs and a deliberate choice of calibration and evaluation dates.
