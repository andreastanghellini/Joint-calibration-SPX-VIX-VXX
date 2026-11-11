# Load the pricing routines and the market snapshot.
import numpy as np
import functions_level3_enrichment_CIR as functions
from market_data_template_CIR import (
    valuation_date, previous_vix_future_expiry, futures_maturities, T_spx, T_vix, T_vxx, vxx_spot, r,
    strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx,
    strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix,
    strikes_vxx, bid_vxx, ask_vxx,
)

# === PASTE THE PARAMETERS FROM THE CURRENT CALIBRATION HERE ===
# Copy the Optimal ell and Optimal rho_b values printed by the main script.
'''
ell_fixed = np.array([
     1.78034597e-01,  3.42182044e-01,  3.38384706e-02, -3.34090536e-02,
     1.66124313e-01,  1.44033224e+01,  6.24837337e-02,  4.69427224e-02,
    -1.25331888e+00, -4.75277297e-01,  2.85416507e-02, -7.57939459e-03,
     6.89704119e-02, -5.60641533e-03, -1.33887572e-02,  7.55897376e-03,
    -2.13050781e-02, -3.07672019e-01,  7.50689457e-02, -2.12524796e-02,
     2.97081249e-01, -2.49253220e-01, -3.11540124e-01, -1.11908573e+00,
     2.20866228e+00])

rho_b_fixed = np.array([0.16198819, 0.86858957, -0.49394888])
'''

ell_fixed = np.array([1.80019427e-01,  4.61950053e-01, -1.08385853e-01, -3.02125301e-02,
     1.56072432e-01,  6.17876383e+00,  8.42840799e-01, -7.07860290e-04,
    -8.63394392e-01, -7.82715752e-01,  5.51015859e-02,  1.51765465e-02,
    -8.16255771e-02,  3.31908723e-02,  6.54638618e-03,  6.87084954e-03,
    -2.18291533e-02, -2.09107658e-01, -9.67873223e-02, -2.17794848e-02,
     2.47639880e-01,  9.08004241e-01,  3.45714490e-01, -3.94356075e-01,
     1.74820083e+00])

rho_b_fixed = np.array([0.31357466, 0.87752185, -0.4608181])

# === MODEL PARAMETERS ===
# Set the two OU factors, the CIR factor and their initial states.
d = 3
alphabet_size = d + 1
window = 30 / 365
kappa = np.array([0.1, 25])
theta = np.array([0.1, 4])
eta = np.array([0.7, 10])
rho_ou = np.array([[1, 0.213], [0.213, 1]])
x0 = np.array([1, 0.08])
kappa_cir = 4.0
theta_cir = 1.0
eta_cir = 2.0
y0_cir = 1.0
cir_letter = 3
x0_primary = np.concatenate((x0, np.array([y0_cir])))
x0_extended = np.insert(x0_primary, 0, 0.0)
rho_primary = np.eye(d)
rho_primary[:2, :2] = rho_ou


# === MATURITIES AND PRICING CONVENTIONS ===
# Use the same futures floor, VXX fee and variance convention as in calibration.
market_maturities = np.array([T_spx, T_vix, T_vxx])
previous_expiry = (previous_vix_future_expiry - valuation_date).days / 365.0
leverage_cap = 100.0
future_floor = 1e-2
vxx_annual_fee = 0.0089
variance_treatment = "positive_part"
variance_softness = 1e-8


# === MONTE CARLO SETTINGS ===
# Set the simulation size, time grid and optional variance reduction.
daily_observations = 4
total_paths = 1000000
batch_size = 10000
simulation_seed = None
use_price_control_variates = False

Tmax = float(np.max(market_maturities))
nsteps = int(round(daily_observations * Tmax * 365))
time_grid = np.linspace(0.0, Tmax, nsteps + 1)
dt = float(time_grid[1] - time_grid[0])
market_indices = np.array([np.argmin(np.abs(time_grid - maturity)) for maturity in market_maturities], dtype=int)
N_spx, N_vix, N_vxx = map(int, market_indices)
time_steps_vxx = time_grid[:N_vxx + 1]


# === GENERATOR-CLOSED LEVEL-THREE BASIS ===
# Rebuild the calibration basis and check the two pasted parameter vectors.
generator_coefficients = functions.ou_cir_generator_coefficients(kappa, theta, eta, rho_ou, kappa_cir, theta_cir, eta_cir)
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
requested_words = [[0, cir_letter, cir_letter], [slow_ou_letter] * 3]
selected_signature_indices, G_n, _, _ = functions.generator_closed_cubic_basis(generator_coefficients, x0_extended, requested_words)
dimension_ell = int(selected_signature_indices.size)
if ell_fixed.shape != (dimension_ell,) or not np.all(np.isfinite(ell_fixed)):
    raise ValueError(f"Paste the {dimension_ell} finite values from Optimal ell into ell_fixed at the top of this file.")
if rho_b_fixed.shape != (d,) or not np.all(np.isfinite(rho_b_fixed)):
    raise ValueError(f"Paste the {d} finite values from Optimal rho_b into rho_b_fixed at the top of this file.")
reduced_variance_model = functions.prepare_reduced_variance_model(
    generator_coefficients, x0_extended, selected_signature_indices, alphabet_size, window)


# === FORWARD REFERENCES AND CALL IMPLIED VOLATILITIES ===
# Build the market call smiles using the calibration forward references.
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)
forward_vix = functions.forward_from_put_call_parity(strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix, T_vix, r)
# Match the q=0 cash-and-carry reference used for VXX calibration.
forward_vxx = vxx_spot * np.exp(r * T_vxx)
call_bid_iv_spx = functions.invert_prices(bid_spx, forward_spx, strikes_spx, T_spx, r=r)
call_ask_iv_spx = functions.invert_prices(ask_spx, forward_spx, strikes_spx, T_spx, r=r)
call_bid_iv_vix = functions.invert_prices(bid_vix, forward_vix, strikes_vix, T_vix, r=r)
call_ask_iv_vix = functions.invert_prices(ask_vix, forward_vix, strikes_vix, T_vix, r=r)
call_bid_iv_vxx = functions.invert_prices(bid_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)
call_ask_iv_vxx = functions.invert_prices(ask_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === FIXED VARIANCE MATRIX AND FUTURES PROPAGATORS ===
# Prepare the variance and futures propagators for the fixed fit.
variance_matrix_fixed = functions.fixed_ell_variance_matrix(ell_fixed, reduced_variance_model, window)
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
initial_signature = np.zeros(dimension_ell)
initial_signature[0] = 1.0
model_future_vix = float(100.0 * ell_fixed @ vix_future_propagator @ initial_signature)
if not np.isfinite(model_future_vix) or model_future_vix <= functions.future_floor_in_points(future_floor):
    raise RuntimeError("The initial VIX future must exceed the stopping floor; recalibrate the fixed coefficients.")

if use_price_control_variates:
    vix_control_loadings = functions.expected_selected_signature_on_cir_grid(
        kappa, theta, eta, rho_ou, x0, kappa_cir, theta_cir, eta_cir, y0_cir, time_grid[:N_vix + 1], selected_signature_indices)
    control_means = {"SPX": forward_spx, "VIX": float(100.0 * ell_fixed @ vix_control_loadings)}
    control_deltas = {
        "SPX": np.array([functions.black_call_delta(forward_spx, strike, T_spx, r, vol)
            for strike, vol in zip(strikes_spx, 0.5 * (call_bid_iv_spx + call_ask_iv_spx))]),
        "VIX": np.array([functions.black_call_delta(forward_vix, strike, T_vix, r, vol)
            for strike, vol in zip(strikes_vix, 0.5 * (call_bid_iv_vix + call_ask_iv_vix))]),
    }


# === SIMULATION AND OUT-OF-SAMPLE PRICING ===
# Accumulate call payoffs and report progress after each batch.
price_sums = {name: np.zeros_like(strikes, dtype=float)
    for name, strikes in (("SPX", strikes_spx), ("VIX", strikes_vix), ("VXX", strikes_vxx))}
batches = int(np.ceil(total_paths / batch_size))
seeds = np.random.SeedSequence(simulation_seed).spawn(batches)
completed_paths = 0
bar_width = 28
controls = {}
print(f"Validation | Batch 0/{batches} | [{' ' * bar_width}]   0.0% | 0/{total_paths:,} paths", flush=True)
for batch in range(batches):
    npaths = min(batch_size, total_paths - batch * batch_size)
    paths, _, _, primary_brownian_increments, independent_brownian_normals = functions.simulate_time_extended_ou_cir_with_brownian(
        kappa, theta, eta, rho_ou, rho_b_fixed, x0, kappa_cir, theta_cir, eta_cir, y0_cir, Tmax, nsteps, npaths,
        seed=seeds[batch], driver_steps=N_spx, return_brownian_increments=False)
    sigX_truncated, raw_variance = functions.compute_fixed_cir_signatures_and_variance(
        paths, selected_signature_indices, variance_matrix_fixed)
    del paths
    variance_left = functions.variance_for_pricing(raw_variance[:N_spx], variance_treatment, variance_softness)
    brownian_increments = functions.brownian_increments_with_saturation(rho_b_fixed, rho_primary, variance_left,
        primary_brownian_increments, np.sqrt(dt) * independent_brownian_normals, leverage_cap)
    terminal_spx = forward_spx * functions.stock_ratio_path(variance_left, brownian_increments, dt)[-1]
    raw_terminal_vix = 100.0 * (sigX_truncated[N_vix] @ ell_fixed)
    terminal_vix = functions.vix_option_proxy(raw_terminal_vix)
    discounted_vxx = functions.vxx_path(
        vxx_spot, ell_fixed, time_steps_vxx, sigX_truncated[:N_vxx + 1], vxx_roll_data, future_floor=future_floor)[-1]
    terminal_vxx = np.exp((r - vxx_annual_fee) * T_vxx) * discounted_vxx
    if use_price_control_variates:
        controls = {"SPX": terminal_spx - control_means["SPX"], "VIX": raw_terminal_vix - control_means["VIX"]}
    for name, terminal, strikes, maturity in (
        ("SPX", terminal_spx, strikes_spx, T_spx), ("VIX", terminal_vix, strikes_vix, T_vix), ("VXX", terminal_vxx, strikes_vxx, T_vxx)):
        if not np.all(np.isfinite(terminal)):
            raise RuntimeError(f"Non-finite {name} terminal values in batch {batch + 1}.")
        payoff = np.exp(-r * maturity) * np.maximum(terminal[:, None] - strikes, 0.0)
        if name in controls:
            payoff -= controls[name][:, None] * control_deltas[name]
        price_sums[name] += payoff.sum(axis=0)
    del sigX_truncated, raw_variance, variance_left, brownian_increments, primary_brownian_increments, independent_brownian_normals
    completed_paths += npaths
    filled = bar_width * completed_paths // total_paths
    bar = "#" * filled + " " * (bar_width - filled)
    print(f"Validation | Batch {batch + 1}/{batches} | [{bar}] {completed_paths / total_paths:6.1%} | "
          f"{completed_paths:,}/{total_paths:,} paths", flush=True)


# === MODEL PRICES AND IMPLIED VOLATILITIES ===
# Average the discounted payoffs and invert the resulting call prices.
prices_spx = price_sums["SPX"] / total_paths
prices_vix = price_sums["VIX"] / total_paths
prices_vxx = price_sums["VXX"] / total_paths
model_iv_spx = functions.invert_prices(prices_spx, forward_spx, strikes_spx, T_spx, r=r)
model_iv_vix = functions.invert_prices(prices_vix, forward_vix, strikes_vix, T_vix, r=r)
model_iv_vxx = functions.invert_prices(prices_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === OUT-OF-SAMPLE IMPLIED-VOLATILITY PLOTS ===
# Plot the three out-of-sample smiles against the market bid and ask bands.
figure, axes = functions.plot_out_of_sample_smiles(model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix,
    strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx,
    call_bid_iv_spx, call_ask_iv_spx, call_bid_iv_vix, call_ask_iv_vix, call_bid_iv_vxx, call_ask_iv_vxx)
