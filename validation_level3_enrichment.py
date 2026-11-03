import importlib
import numpy as np
import functions_level3_enrichment as functions
from datetime import date

# Market inputs are loaded from an untracked local file.
from market_data_private import (
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
    put_bid_vix,
    put_ask_vix,
    strikes_vxx,
    bid_vxx,
    ask_vxx,
    put_bid_vxx,
    put_ask_vxx,
)

functions = importlib.reload(functions)


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


# === LEVEL 3 EXTENSION ===
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
slow_cubic_word = [slow_ou_letter, slow_ou_letter, slow_ou_letter]
time_letter = 0
time_cubic_word = [time_letter, time_letter, time_letter]


# === MARKET MATURITIES AND CONSTRAINTS ===
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


# === MONTE CARLO SETTINGS ===
daily_observations = 4
total_paths = 1000000
batch_size = 10000
simulation_seed = None
use_price_control_variates = True
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
reduced_variance_model = functions.prepare_reduced_variance_model(coeff=generator_coefficients,x0=x0_extended,selected_indices=selected_signature_indices,dim=alphabet_size,window=window)


# === LOCAL MARKET INPUTS ARE IMPORTED ABOVE ===


# === SPX DATA ===
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)


# == VIX DATA ===
forward_vix = functions.forward_from_put_call_parity(strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix, T_vix, r)


# === VXX DATA ===
forward_vxx = functions.forward_from_put_call_parity(strikes_vxx, bid_vxx, ask_vxx, put_bid_vxx, put_ask_vxx, T_vxx, r)
cp_spx = np.ones_like(strikes_spx, dtype=int)
cp_vix = np.ones_like(strikes_vix, dtype=int)
cp_vxx = np.ones_like(strikes_vxx, dtype=int)
bid_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r, cp) for price, strike, cp in zip(bid_spx, strikes_spx, cp_spx)])
ask_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r, cp) for price, strike, cp in zip(ask_spx, strikes_spx, cp_spx)])
bid_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r, cp) for price, strike, cp in zip(bid_vix, strikes_vix, cp_vix)])
ask_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r, cp) for price, strike, cp in zip(ask_vix, strikes_vix, cp_vix)])
bid_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r, cp) for price, strike, cp in zip(bid_vxx, strikes_vxx, cp_vxx)])
ask_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r, cp) for price, strike, cp in zip(ask_vxx, strikes_vxx, cp_vxx)])


# === FIXED CALIBRATION PARAMETERS ===
ell_fixed = np.array([
    1.82789493e-01,  3.18305010e-01, -2.56639788e-01,
   -1.04020507e-02,  5.72693101e-02,  4.44179188e+00,
    2.24191097e+00,  3.19029181e-02, -9.87127842e-02,
    1.82469142e+00,  1.84341293e+00,  9.78635773e-03,
   -1.92386619e-02,  3.37051802e-02,  7.18668722e-03,
    7.64522543e-04, -4.87624357e-03, -1.44022429e-01,
   -5.30579847e-03, -3.98358877e-03,  2.82369876e-02,
   -4.65870476e+00, -1.90741702e+02])

rho_b_fixed = np.array([0.75332517, 0.34725247, -0.92662907])


# === INITIALIZATION AND PRECOMPUTATION ===
variance_matrix_fixed = functions.fixed_ell_variance_matrix(ell_fixed, reduced_variance_model, window)
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
initial_truncated_signature = np.zeros(dimension_ell)
initial_truncated_signature[0] = 1.0
model_future_vix = float(functions.regularize_vix_futures(100.0 * ell_fixed @ vix_future_propagator @ initial_truncated_signature, future_floor))
vix_control_loadings = functions.expected_selected_signature_on_ou_grid(kappa, theta, eta, rho_ou, x0, time_grid[: N_vix + 1], selected_signature_indices)
control_means = {"SPX": forward_spx, "VIX": float(100.0 * ell_fixed @ vix_control_loadings)}
control_deltas = {}
for name, strikes, forward, maturity, bid, ask, put_bid, put_ask in (
    ("SPX", strikes_spx, forward_spx, T_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx),
    ("VIX", strikes_vix, forward_vix, T_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix),
):
    cp = np.where(strikes < forward, -1, 1)
    selected_bid = np.where(cp == -1, put_bid, bid)
    selected_ask = np.where(cp == -1, put_ask, ask)
    bid_iv = np.array([functions.implied_vol_black76(price, forward, strike, maturity, r, int(sign)) for price, strike, sign in zip(selected_bid, strikes, cp)])
    ask_iv = np.array([functions.implied_vol_black76(price, forward, strike, maturity, r, int(sign)) for price, strike, sign in zip(selected_ask, strikes, cp)])
    deltas = np.array([functions.black_option_delta(forward, strike, maturity, r, vol, int(sign)) for strike, vol, sign in zip(strikes, 0.5 * (bid_iv + ask_iv), cp)])
    control_deltas[name] = deltas + np.exp(-r * maturity) * (cp == -1)
price_sums = {"SPX": np.zeros_like(strikes_spx, dtype=float), "VIX": np.zeros_like(strikes_vix, dtype=float), "VXX": np.zeros_like(strikes_vxx, dtype=float)}
batches = int(np.ceil(total_paths / batch_size))
seeds = np.random.SeedSequence(simulation_seed).spawn(batches)


# === SIMULATION AND PRICING OUT OF SAMPLE ===
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
    controls = {"SPX": terminal_spx - control_means["SPX"], "VIX": raw_terminal_vix - control_means["VIX"]}
    for name, terminal, strikes, maturity in (("SPX", terminal_spx, strikes_spx, T_spx), ("VIX", terminal_vix, strikes_vix, T_vix), ("VXX", terminal_vxx, strikes_vxx, T_vxx)):
        payoff = np.exp(-r * maturity) * np.maximum(terminal[:, None] - strikes, 0.0)
        if use_price_control_variates and name in controls:
            payoff -= controls[name][:, None] * control_deltas[name]
        price_sums[name] += payoff.sum(axis=0)
    del sigX_truncated, raw_variance, variance_left, brownian_increments, factor_brownian_increments, independent_brownian_increments


# === MODEL PRICES AND IMPLIED VOLATILITIES ===
prices_spx = price_sums["SPX"] / total_paths
prices_vix = price_sums["VIX"] / total_paths
prices_vxx = price_sums["VXX"] / total_paths
model_iv_spx = functions.invert_prices(prices_spx, forward_spx, strikes_spx, T_spx, r=r)
model_iv_vix = functions.invert_prices(prices_vix, forward_vix, strikes_vix, T_vix, r=r)
model_iv_vxx = functions.invert_prices(prices_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === PLOTS OUT OF SAMPLE ===
figure, axes = functions.plot_out_of_sample_smiles(
    model_iv_spx,
    model_iv_vix,
    model_iv_vxx,
    model_future_vix,
    strikes_spx,
    strikes_vix,
    strikes_vxx,
    forward_spx,
    forward_vix,
    forward_vxx,
    bid_iv_spx,
    ask_iv_spx,
    bid_iv_vix,
    ask_iv_vix,
    bid_iv_vxx,
    ask_iv_vxx,
)
