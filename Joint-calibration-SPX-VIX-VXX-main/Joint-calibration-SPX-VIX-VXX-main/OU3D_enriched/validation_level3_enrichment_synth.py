# Load the pricing routines and the market snapshot.
import numpy as np
import functions_level3_enrichment as functions
from datetime import date

# Fill in market_data_template.py with the market inputs before running.
from market_data_private_synth import (
    market_future_vix,
    vxx_spot,
    r,
    strikes_spx,
    bid_spx,
    ask_spx,
    put_bid_spx,
    put_ask_spx,
    strikes_vix,
    bid_vix,
    ask_vix,
    strikes_vxx,
    bid_vxx,
    ask_vxx,
)

# === PASTE THE PARAMETERS FROM THE CURRENT CALIBRATION HERE ===
# Copy the Optimal ell and Optimal rho_b values printed by the main script.
'''
ell_fixed = np.array([
     1.83159828e-01, -1.96218052e-03, -2.16225440e-01, -5.52373569e-04,
     6.66435010e-02,  4.32026956e+00,  1.79949194e+00,  1.09780902e-02,
    -3.23008696e-01,  1.68384185e+00,  1.81401677e+00,  5.98552565e-03,
    -2.90483898e-03,  1.09849190e-02,  1.12389614e-02, -1.15447369e-04,
    -2.13938503e-04, -3.80179066e-01, -7.73550830e-03,  3.01248305e-04,
     3.52224971e-02, -5.40774936e+00, -1.55189355e+02])

rho_b_fixed = np.array([0.72476236, 0.08704893, -0.97668878])
'''

ell_fixed = np.array([
    1.85138349e-01, -1.92269464e-01, -2.70779625e-01, -1.98802585e-03,
    6.13864086e-02,  9.51436717e+00,  2.09537687e+00,  1.63869811e-02,
   -2.92722244e-01,  1.98824514e+00,  1.80186495e+00,  6.07585753e-03,
   -1.86482368e-02,  3.01931432e-02,  1.73585996e-02, -6.86456551e-05,
    3.35839507e-05, -4.04026409e-01, -2.93990936e-02, -7.43620046e-04,
    3.08184079e-02, -5.68814966e+00, -2.13004769e+02
], dtype=float)

rho_b_fixed = np.array([
    0.76132244, 0.16333509, -0.95742185
], dtype=float)

# === MODEL PARAMETERS ===
truncation_vola = 2
d = 3
window = 30 / 365
alphabet_size = d + 1
kappa = np.array([0.1, 25, 10])
theta = np.array([0.1, 4, 0.08])
eta = np.array([0.7, 10, 5])
rho_ou = np.array([[1, 0.213, -0.576], [0.213, 1, -0.044], [-0.576, -0.044, 1]])
x0 = np.array([1, 0.08, 2])
x0_extended = np.insert(x0, 0, 0)


# === LEVEL-THREE SIGNATURE EXTENSION ===
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
slow_cubic_word = [slow_ou_letter, slow_ou_letter, slow_ou_letter]
time_letter = 0
time_cubic_word = [time_letter, time_letter, time_letter]


# === MATURITIES AND PRICING CONVENTIONS (24/02/2026) ===
futures_maturities = np.array([22, 50, 84, 113]) / 365
valuation_date = date(2026, 2, 24)
previous_vix_future_expiry = date(2026, 2, 18)
previous_expiry = (previous_vix_future_expiry - valuation_date).days / 365.0
leverage_cap = 100.0
future_floor = 1e-2
vxx_annual_fee = 0.0089
T_spx = 24 / 365
T_vix = 22 / 365
T_vxx = 24 / 365
market_maturities = np.array([T_spx, T_vix, T_vxx])
'''
# === MATURITIES AND PRICING CONVENTIONS (26/08/2026) ===
futures_maturities = np.array([21, 56, 84, 112]) / 365
valuation_date = date(2026, 8, 26)
previous_vix_future_expiry = date(2026, 8, 19)
previous_expiry = (previous_vix_future_expiry - valuation_date).days / 365.0
leverage_cap = 100.0
future_floor = 1e-2
vxx_annual_fee = 0.0089
T_spx = 23 / 365
T_vix = 21 / 365
T_vxx = 23 / 365
market_maturities = np.array([T_spx, T_vix, T_vxx])
'''

# === FORWARD REFERENCES AND CALL IMPLIED VOLATILITIES ===
if market_future_vix is None or not np.isfinite(market_future_vix) or market_future_vix <= 0.0:
    raise ValueError("Set market_future_vix in market_data_template.py to the quoted VIX future for T_vix, in VIX points.")
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)
forward_vix = market_future_vix
# Use the Black-Scholes spot convention with q = 0 for both VXX smiles.
forward_vxx = vxx_spot * np.exp(r * T_vxx)
call_bid_iv_spx = functions.invert_prices(bid_spx, forward_spx, strikes_spx, T_spx, r=r)
call_ask_iv_spx = functions.invert_prices(ask_spx, forward_spx, strikes_spx, T_spx, r=r)
call_bid_iv_vix = functions.invert_prices(bid_vix, forward_vix, strikes_vix, T_vix, r=r)
call_ask_iv_vix = functions.invert_prices(ask_vix, forward_vix, strikes_vix, T_vix, r=r)
call_bid_iv_vxx = functions.invert_prices(bid_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)
call_ask_iv_vxx = functions.invert_prices(ask_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === MONTE CARLO SETTINGS ===
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
time_steps_vxx = time_grid[: N_vxx + 1]


# === INITIALIZATION POLYNOMIAL COEFFICIENTS TO COMPUTE THE GENERATOR===
b_const = np.zeros(alphabet_size)
b_lin = np.zeros((alphabet_size, alphabet_size))
b_const[0] = 1.0
for i in range(d):
    coordinate = i + 1
    b_const[coordinate] = kappa[i] * theta[i]
    b_lin[coordinate, coordinate] = -kappa[i]
a_const = np.zeros((alphabet_size, alphabet_size))
a_const[1:, 1:] = rho_ou * np.outer(eta, eta)
a_lin = np.zeros((alphabet_size, alphabet_size, alphabet_size))
a_quad = np.zeros((alphabet_size, alphabet_size, alphabet_size, alphabet_size))
generator_coefficients = (b_const, b_lin, a_const, a_lin, a_quad)


# === PREPARING VIX REPRESENTATION AND SPX VARIANCE ===
selected_signature_indices, G_n, selected_generator_closure_error = functions.restricted_signature_generator(coeff=generator_coefficients,x0=x0_extended,base_level=truncation_vola,extra_words=[slow_cubic_word, time_cubic_word],dim=alphabet_size)
dimension_ell = int(selected_signature_indices.size)
if ell_fixed.shape != (dimension_ell,) or not np.all(np.isfinite(ell_fixed)):
    raise ValueError(f"Paste the {dimension_ell} finite values from Optimal ell into ell_fixed at the top of this file.")
if rho_b_fixed.shape != (d,) or not np.all(np.isfinite(rho_b_fixed)):
    raise ValueError(f"Paste the {d} finite values from Optimal rho_b into rho_b_fixed at the top of this file.")
reduced_variance_model = functions.prepare_reduced_variance_model(coeff=generator_coefficients,x0=x0_extended,selected_indices=selected_signature_indices,dim=alphabet_size,window=window)


# === FIXED VARIANCE MATRIX AND FUTURES PROPAGATORS ===
variance_matrix_fixed = functions.fixed_ell_variance_matrix(ell_fixed, reduced_variance_model, window)
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
initial_truncated_signature = np.zeros(dimension_ell)
initial_truncated_signature[0] = 1.0
model_future_vix = float(100.0 * ell_fixed @ vix_future_propagator @ initial_truncated_signature)
if not np.isfinite(model_future_vix) or model_future_vix <= functions.future_floor_in_points(future_floor):
    raise RuntimeError("The initial VIX future must exceed the stopping floor; recalibrate the fixed coefficients.")

if use_price_control_variates:
    vix_control_loadings = functions.expected_selected_signature_on_ou_grid(kappa, theta, eta, rho_ou, x0, time_grid[: N_vix + 1], selected_signature_indices)
    mean_raw_vix = float(100.0 * ell_fixed @ vix_control_loadings)
    control_means = {"SPX": forward_spx, "VIX": mean_raw_vix}
    control_deltas = {
        "SPX": np.array([functions.black_call_delta(forward_spx, strike, T_spx, r, vol)
            for strike, vol in zip(strikes_spx, 0.5 * (call_bid_iv_spx + call_ask_iv_spx))]),
        "VIX": np.array([functions.black_call_delta(forward_vix, strike, T_vix, r, vol)
            for strike, vol in zip(strikes_vix, 0.5 * (call_bid_iv_vix + call_ask_iv_vix))]),
    }


# === SIMULATION AND OUT-OF-SAMPLE PRICING ===
# Accumulate call payoffs and VIX terminal levels across batches.
price_sums = {name: np.zeros_like(strikes, dtype=float)
    for name, strikes in (("SPX", strikes_spx), ("VIX", strikes_vix), ("VXX", strikes_vxx))}
vix_terminal_sum = 0.0
batches = int(np.ceil(total_paths / batch_size))
seeds = np.random.SeedSequence(simulation_seed).spawn(batches)
completed_paths = 0
bar_width = 28
controls = {}
print(f"Validation | Batch 0/{batches} | [{' ' * bar_width}]   0.0% | 0/{total_paths:,} paths", flush=True)
for batch in range(batches):
    npaths = min(batch_size, total_paths - batch * batch_size)
    paths, _, factor_brownian_increments, independent_brownian_increments = functions.simulate_time_extended_ou_3d_with_drivers(kappa, theta, eta, rho_ou, x0, Tmax, nsteps, npaths, seed=seeds[batch])
    sigX_truncated = np.empty((nsteps + 1, npaths, dimension_ell))
    raw_variance = np.empty((nsteps + 1, npaths))
    state = np.zeros((npaths, dimension_ell))
    state[:, 0] = 1.0
    for node in range(nsteps + 1):
        if node:
            state = functions.advance_selected_signature(state, paths[node] - paths[node - 1])
        sigX_truncated[node] = state
        raw_variance[node] = functions.raw_variance_from_selected_signature(state, variance_matrix_fixed)
    del paths
    variance_left = functions.variance_for_pricing(raw_variance[:N_spx])
    brownian_increments = functions.brownian_increments_with_saturation(rho_b_fixed, rho_ou, variance_left, factor_brownian_increments[:N_spx], independent_brownian_increments[:N_spx], leverage_cap)
    terminal_spx = forward_spx * functions.stock_ratio_path(variance_left, brownian_increments, dt)[-1]
    raw_terminal_vix = 100.0 * (sigX_truncated[N_vix] @ ell_fixed)
    terminal_vix = functions.vix_option_proxy(raw_terminal_vix)
    discounted_vxx = functions.vxx_path(vxx_spot, ell_fixed, time_steps_vxx, sigX_truncated[: N_vxx + 1], vxx_roll_data, future_floor=future_floor)[-1]
    terminal_vxx = np.exp((r - vxx_annual_fee) * T_vxx) * discounted_vxx
    vix_terminal_sum += terminal_vix.sum()
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
    del sigX_truncated, raw_variance, variance_left, brownian_increments, factor_brownian_increments, independent_brownian_increments
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
# Use the mean of the same terminal VIX proxy used to price the calls.
model_forward_vix = vix_terminal_sum / total_paths
model_iv_spx = functions.invert_prices(prices_spx, forward_spx, strikes_spx, T_spx, r=r)
model_iv_vix = functions.invert_prices(prices_vix, model_forward_vix, strikes_vix, T_vix, r=r)
model_iv_vxx = functions.invert_prices(prices_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === OUT-OF-SAMPLE IMPLIED-VOLATILITY PLOTS ===
# Plot the three out-of-sample smiles against the market bid and ask bands.
figure, axes = functions.plot_out_of_sample_smiles(model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix,
    strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx,
    call_bid_iv_spx, call_ask_iv_spx, call_bid_iv_vix, call_ask_iv_vix, call_bid_iv_vxx, call_ask_iv_vxx,
    model_forward_vix=model_forward_vix)
