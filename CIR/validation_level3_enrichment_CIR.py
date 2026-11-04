import importlib
import numpy as np
import functions_level3_enrichment_CIR as functions
from market_data_private_CIR import (
    valuation_date, previous_vix_future_expiry, futures_maturities, T_spx, T_vix, T_vxx,
    spx_spot, vix_spot, vxx_spot, r,
    strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx,
    strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix,
    strikes_vxx, bid_vxx, ask_vxx, put_bid_vxx, put_ask_vxx,
)

functions = importlib.reload(functions)


# === MODEL PARAMETERS ===
truncation_vola = 2
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
market_maturities = np.array([T_spx, T_vix, T_vxx])
previous_expiry = (previous_vix_future_expiry - valuation_date).days / 365.0
leverage_cap = 100.0
future_floor = 1e-2
vxx_annual_fee = 0.0089
variance_treatment = "positive_part"
variance_softness = 1e-8


# === MONTE CARLO SETTINGS ===
daily_observations = 4
total_paths = 1000000
batch_size = 10000
simulation_seed = None
Tmax = float(np.max(market_maturities))
nsteps = int(round(daily_observations * Tmax * 365))
time_grid = np.linspace(0.0, Tmax, nsteps + 1)
dt = float(time_grid[1] - time_grid[0])
market_indices = np.array([np.argmin(np.abs(time_grid - maturity)) for maturity in market_maturities], dtype=int)
N_spx, N_vix, N_vxx = map(int, market_indices)
time_steps_vxx = time_grid[:N_vxx + 1]


# === GENERATOR-CLOSED LEVEL-THREE BASIS ===
generator_coefficients = functions.ou_cir_generator_coefficients(kappa, theta, eta, rho_ou, kappa_cir, theta_cir, eta_cir)
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
requested_words = [[0, cir_letter, cir_letter], [slow_ou_letter] * 3]
selected_signature_indices, G_n, selected_generator_closure_error, extra_words = functions.generator_closed_cubic_basis(generator_coefficients, x0_extended, requested_words)
dimension_ell = int(selected_signature_indices.size)
reduced_variance_model = functions.prepare_reduced_variance_model(generator_coefficients, x0_extended, selected_signature_indices, alphabet_size, window)


# === MARKET FORWARDS AND CALL IMPLIED VOLATILITIES ===
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)
forward_vix = functions.forward_from_put_call_parity(strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix, T_vix, r)
forward_vxx = functions.forward_from_put_call_parity(strikes_vxx, bid_vxx, ask_vxx, put_bid_vxx, put_ask_vxx, T_vxx, r)
call_bid_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r) for price, strike in zip(bid_spx, strikes_spx)])
call_ask_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r) for price, strike in zip(ask_spx, strikes_spx)])
call_bid_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r) for price, strike in zip(bid_vix, strikes_vix)])
call_ask_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r) for price, strike in zip(ask_vix, strikes_vix)])
call_bid_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r) for price, strike in zip(bid_vxx, strikes_vxx)])
call_ask_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r) for price, strike in zip(ask_vxx, strikes_vxx)])


# === FIXED CALIBRATION PARAMETERS ===
# Replace the supplied manual coefficients with the output of the current main.
ell_fixed = np.array([
    0.182100857, 0.464353033, -0.0450299399, -0.0285110184,
    0.194550541, 6.29543307, 0.640799127, 0.0868448403,
    -1.340185, -0.594418605, -0.0771023452, 0.00708784571,
    -0.019192054, 0.000794116325, -0.00432555506, 0.00527626548,
    -0.0269361097, -0.61691656, -0.0531788295, -0.0228010235,
    0.285513551, 0.28013186, 0.242786089, -0.918218481,
    1.62404547,
], dtype=float)
rho_b_fixed = np.array([0.28144331, 0.74590192, -0.64584134], dtype=float)


# === FIXED VARIANCE MATRIX AND FUTURES PROPAGATORS ===
variance_matrix_fixed = functions.fixed_ell_variance_matrix(ell_fixed, reduced_variance_model, window)
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
initial_signature = np.zeros(dimension_ell)
initial_signature[0] = 1.0
model_future_vix = float(100.0 * ell_fixed @ vix_future_propagator @ initial_signature)
if not np.isfinite(model_future_vix) or model_future_vix <= functions.future_floor_in_points(future_floor):
    raise RuntimeError("The initial VIX future must exceed the stopping floor; recalibrate the fixed coefficients.")
price_sums = {"SPX": np.zeros_like(strikes_spx, dtype=float), "VIX": np.zeros_like(strikes_vix, dtype=float), "VXX": np.zeros_like(strikes_vxx, dtype=float)}
batches = int(np.ceil(total_paths / batch_size))
seeds = np.random.SeedSequence(simulation_seed).spawn(batches)


# === SIMULATION AND OUT-OF-SAMPLE PRICING ===
for batch in range(batches):
    npaths = min(batch_size, total_paths - batch * batch_size)
    paths, _, _, primary_brownian_increments, independent_brownian_normals = functions.simulate_time_extended_ou_cir_with_brownian(
        kappa, theta, eta, rho_ou, rho_b_fixed, x0, kappa_cir, theta_cir, eta_cir, y0_cir, Tmax, nsteps, npaths,
        seed=seeds[batch], driver_steps=N_spx, return_brownian_increments=False
    )
    sigX_truncated, raw_variance = functions.compute_fixed_cir_signatures_and_variance(paths, selected_signature_indices, variance_matrix_fixed)
    del paths
    variance_left = functions.variance_for_pricing(raw_variance[:N_spx], variance_treatment, variance_softness)
    brownian_increments = functions.brownian_increments_with_saturation(rho_b_fixed, rho_primary, variance_left,
        primary_brownian_increments, np.sqrt(dt) * independent_brownian_normals, leverage_cap)
    terminal_spx = forward_spx * functions.stock_ratio_path(variance_left, brownian_increments, dt)[-1]
    terminal_vix = functions.vix_option_proxy(100.0 * (sigX_truncated[N_vix] @ ell_fixed))
    discounted_vxx = functions.vxx_path(vxx_spot, ell_fixed, time_steps_vxx, sigX_truncated[:N_vxx + 1], vxx_roll_data, future_floor=future_floor)[-1]
    terminal_vxx = np.exp((r - vxx_annual_fee) * T_vxx) * discounted_vxx
    for name, terminal, strikes, maturity in (("SPX", terminal_spx, strikes_spx, T_spx), ("VIX", terminal_vix, strikes_vix, T_vix), ("VXX", terminal_vxx, strikes_vxx, T_vxx)):
        if not np.all(np.isfinite(terminal)):
            raise RuntimeError(f"Non-finite {name} terminal values; no trajectory has been discarded.")
        payoff = np.exp(-r * maturity) * np.maximum(terminal[:, None] - strikes, 0.0)
        price_sums[name] += payoff.sum(axis=0)
    del sigX_truncated, raw_variance, variance_left, brownian_increments, primary_brownian_increments, independent_brownian_normals


# === MODEL PRICES AND IMPLIED VOLATILITIES ===
prices_spx = price_sums["SPX"] / total_paths
prices_vix = price_sums["VIX"] / total_paths
prices_vxx = price_sums["VXX"] / total_paths
model_iv_spx = functions.invert_prices(prices_spx, forward_spx, strikes_spx, T_spx, r=r)
model_iv_vix = functions.invert_prices(prices_vix, forward_vix, strikes_vix, T_vix, r=r)
model_iv_vxx = functions.invert_prices(prices_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === OUT-OF-SAMPLE IMPLIED-VOLATILITY PLOTS ===
figure, axes = functions.plot_out_of_sample_smiles(model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix,
    strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx,
    call_bid_iv_spx, call_ask_iv_spx, call_bid_iv_vix, call_ask_iv_vix, call_bid_iv_vxx, call_ask_iv_vxx)
