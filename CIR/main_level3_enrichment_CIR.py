import importlib
from functools import partial
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
future_initial_safety_margin = 1e-8
vxx_annual_fee = 0.0089
variance_treatment = "positive_part"
variance_softness = 1e-8


# === CALIBRATION AND MONTE CARLO SETTINGS ===
rho_b = np.array([0.80656419, -0.00128551, -0.20])
npaths = 10000
daily_observations = 4
simulation_seed = None
positivity_constraint_maturity = 84 / 365
minimum_variance = 0.0
variance_safety_margin = 1e-6
minimum_loss_normalization = 1e-12
beta_spx, beta_vix, beta_vxx = (0.98, 0.9, 0.98)
lambda_spx, lambda_vix, lambda_vxx = (0.4, 0.5, 0.1)
optimizer_options = {"maxiter": 360, "stage_iterations": 40, "max_stages": 12, "ftol": 3e-5}
Tmax = float(max(np.max(market_maturities), positivity_constraint_maturity))
nsteps = int(round(daily_observations * Tmax * 365))
cache_directory = functions.create_signature_cache()


# === GENERATOR-CLOSED LEVEL-THREE BASIS ===
generator_coefficients = functions.ou_cir_generator_coefficients(kappa, theta, eta, rho_ou, kappa_cir, theta_cir, eta_cir)
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
requested_words = [[0, cir_letter, cir_letter], [slow_ou_letter] * 3]
selected_signature_indices, G_n, selected_generator_closure_error, extra_words = functions.generator_closed_cubic_basis(generator_coefficients, x0_extended, requested_words)
dimension_ell = int(selected_signature_indices.size)
reduced_variance_model = functions.prepare_reduced_variance_model(generator_coefficients, x0_extended, selected_signature_indices, alphabet_size, window)
variance_pair_positions = reduced_variance_model["pair_positions"]
variance_shuffle_coordinates = reduced_variance_model["shuffle_product_coordinates"]


# === SIMULATION AND SIGNATURE FEATURES ===
paths, time_grid, _, primary_brownian_increments, independent_brownian_normals = functions.simulate_time_extended_ou_cir_with_brownian(
    kappa, theta, eta, rho_ou, rho_b, x0, kappa_cir, theta_cir, eta_cir, y0_cir, Tmax, nsteps, npaths,
    seed=simulation_seed, driver_steps=int(round(T_spx * 365 * daily_observations)), return_brownian_increments=False
)
sigX_truncated, variance_signature_features = functions.compute_cubic_signature_features(paths, selected_signature_indices, reduced_variance_model, cache_directory)
del paths
dt = float(time_grid[1] - time_grid[0])
market_indices = np.array([np.argmin(np.abs(time_grid - maturity)) for maturity in market_maturities], dtype=int)
N_spx, N_vix, N_vxx = map(int, market_indices)
positivity_constraint_index = int(np.argmin(np.abs(time_grid - positivity_constraint_maturity)))
N_variance_constraint = positivity_constraint_index + 1
time_steps_vxx = time_grid[:N_vxx + 1]


# === MARKET FORWARDS ===
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)
forward_vix = functions.forward_from_put_call_parity(strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix, T_vix, r)
forward_vxx = functions.forward_from_put_call_parity(strikes_vxx, bid_vxx, ask_vxx, put_bid_vxx, put_ask_vxx, T_vxx, r)


# === OUT-OF-THE-MONEY CALIBRATION QUOTES ===
cp_spx = np.where(strikes_spx < forward_spx, -1, 1)
cp_vix = np.where(strikes_vix < forward_vix, -1, 1)
cp_vxx = np.where(strikes_vxx < forward_vxx, -1, 1)
bid_spx, ask_spx = np.where(cp_spx == 1, bid_spx, put_bid_spx), np.where(cp_spx == 1, ask_spx, put_ask_spx)
bid_vix, ask_vix = np.where(cp_vix == 1, bid_vix, put_bid_vix), np.where(cp_vix == 1, ask_vix, put_ask_vix)
bid_vxx, ask_vxx = np.where(cp_vxx == 1, bid_vxx, put_bid_vxx), np.where(cp_vxx == 1, ask_vxx, put_ask_vxx)
bid_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r, cp) for price, strike, cp in zip(bid_spx, strikes_spx, cp_spx)])
ask_iv_spx = np.array([functions.implied_vol_black76(price, forward_spx, strike, T_spx, r, cp) for price, strike, cp in zip(ask_spx, strikes_spx, cp_spx)])
bid_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r, cp) for price, strike, cp in zip(bid_vix, strikes_vix, cp_vix)])
ask_iv_vix = np.array([functions.implied_vol_black76(price, forward_vix, strike, T_vix, r, cp) for price, strike, cp in zip(ask_vix, strikes_vix, cp_vix)])
bid_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r, cp) for price, strike, cp in zip(bid_vxx, strikes_vxx, cp_vxx)])
ask_iv_vxx = np.array([functions.implied_vol_black76(price, forward_vxx, strike, T_vxx, r, cp) for price, strike, cp in zip(ask_vxx, strikes_vxx, cp_vxx)])


# === LOSS NORMALIZATIONS AND FUTURES PROPAGATORS ===
normalization_spx = functions.precompute_loss_normalization(forward_spx, strikes_spx, T_spx, r, bid_iv_spx, ask_iv_spx, minimum_loss_normalization)
normalization_vix = functions.precompute_loss_normalization(forward_vix, strikes_vix, T_vix, r, bid_iv_vix, ask_iv_vix, minimum_loss_normalization)
normalization_vxx = functions.precompute_loss_normalization(forward_vxx, strikes_vxx, T_vxx, r, bid_iv_vxx, ask_iv_vxx, minimum_loss_normalization)
mid_iv_vix = 0.5 * (bid_iv_vix + ask_iv_vix)
deltas_vix = np.array([functions.black_option_delta(forward_vix, strike, T_vix, r, vol, cp) for strike, vol, cp in zip(strikes_vix, mid_iv_vix, cp_vix)])
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)


# === INITIAL FUTURES CONSTRAINTS ===
required_future_maturities = np.unique(np.array([T_vix] + [maturity for segment in vxx_roll_data for maturity in segment[2:4]]))
initial_future_loadings = np.stack([functions.precompute_vix_future_propagators(np.array([0.0]), maturity, G_n)[0]
    @ sigX_truncated[0, 0] for maturity in required_future_maturities])
minimum_initial_future = future_floor + future_initial_safety_margin


# === OBJECTIVE INPUTS AND VARIANCE CONSTRAINT ===
objective_model_inputs = {
    "shuffle_tensor_nn": None, "variance_signature_features": variance_signature_features,
    "variance_pair_positions": variance_pair_positions, "variance_shuffle_coordinates": variance_shuffle_coordinates,
    "truncated_signature_path": sigX_truncated, "window": window, "dt": dt,
    "N_spx": N_spx, "N_vix": N_vix, "time_steps_vxx": time_steps_vxx,
    "vix_future_propagator": vix_future_propagator, "vxx_roll_data": vxx_roll_data,
    "forward_spx": forward_spx, "vxx_spot": vxx_spot, "leverage_cap": leverage_cap,
    "future_floor": future_floor, "vxx_annual_fee": vxx_annual_fee,
}
objective_market_inputs = {
    "strikes_spx": strikes_spx, "strikes_vix": strikes_vix, "strikes_vxx": strikes_vxx,
    "cp_spx": cp_spx, "cp_vix": cp_vix, "cp_vxx": cp_vxx,
    "maturity_spx": T_spx, "maturity_vix": T_vix, "maturity_vxx": T_vxx, "rate": r,
    "bid_spx": bid_spx, "ask_spx": ask_spx, "bid_vix": bid_vix, "ask_vix": ask_vix, "bid_vxx": bid_vxx, "ask_vxx": ask_vxx,
}
objective_loss_inputs = {
    "normalization_spx": normalization_spx, "normalization_vix": normalization_vix, "normalization_vxx": normalization_vxx,
    "deltas_vix": deltas_vix, "market_future_vix": forward_vix,
    "beta_spx": beta_spx, "beta_vix": beta_vix, "beta_vxx": beta_vxx,
    "lambda_spx": lambda_spx, "lambda_vix": lambda_vix, "lambda_vxx": lambda_vxx,
    "variance_softness": variance_softness, "variance_treatment": variance_treatment,
}
objective_inputs = {**objective_model_inputs, **objective_market_inputs, **objective_loss_inputs}
objective_with_components = partial(functions.joint_calibration_objective_with_cir_correlations, dimension_ell=dimension_ell, rho_primary=rho_primary,
    driver_increments=primary_brownian_increments, independent_normals=independent_brownian_normals, objective_inputs=objective_inputs, return_components=True)
variance_constraint_inputs = {
    "shuffle_tensor_nn": None, "N": N_variance_constraint, "variance_signature_features": variance_signature_features,
    "window": window, "minimum_variance": minimum_variance + variance_safety_margin,
    "variance_pair_positions": variance_pair_positions, "variance_shuffle_coordinates": variance_shuffle_coordinates,
}
variance_constraint = partial(functions.variance_positivity_constraint_with_correlations, dimension_ell=dimension_ell, constraint_inputs=variance_constraint_inputs)


# === INITIAL CALIBRATION PARAMETERS ===
atm_spx_index = int(np.argmin(np.abs(strikes_spx / forward_spx - 1.0)))
atm_spx_iv = 0.5 * (bid_iv_spx[atm_spx_index] + ask_iv_spx[atm_spx_index])
ell_safe = np.zeros(dimension_ell)
ell_safe[0] = atm_spx_iv
ell_safe[1] = (forward_vix / 100.0 - atm_spx_iv) / T_vix
ell_reference_level_two = np.array([
    0.18651227131172177, -1.4918260961042886, -0.4225469969758481, 0.016758021521028788,
    0.0632522317884641, 31.752316250038007, 5.561829451300603, -0.2335494061035052,
    -0.9290206475968611, 5.355090138555593, 1.6512344968996866, -0.020840977507334685,
    -0.13577056761769396, -0.2233325404497525, -0.03446850511830134, 0.0013534146836617382,
    0.004280468134303818, -0.93982423970692, -0.15757918994536588, 0.005448697619296834,
    0.01712767243058856,
], dtype=float)
ell_reference = np.zeros(dimension_ell)
for old_index, old_coefficient in enumerate(ell_reference_level_two):
    old_word = functions.from_num_to_word(old_index, d)
    if cir_letter not in old_word:
        ell_reference[functions.from_word_to_num(old_word, d)] = old_coefficient
initial_stochastic_weight = 0.20
ell0 = ell_safe + initial_stochastic_weight * (ell_reference - ell_safe)
correlation_parameters0 = functions.parameters_from_rho_b(rho_b, rho_primary)
calibration_parameters0 = np.concatenate((ell0, correlation_parameters0))


# === NORMALIZED JOINT LOSS ===
initial_raw_loss, initial_block_losses = objective_with_components(calibration_parameters0)
loss_scales = {product: max(float(initial_block_losses[product]), 1e-12) for product in ("SPX", "VIX", "VXX")}
objective_scale = 1000.0
normalized_joint_loss = partial(functions.normalized_joint_loss, lambda_spx=lambda_spx, lambda_vix=lambda_vix, lambda_vxx=lambda_vxx,
    loss_scales=loss_scales, objective_scale=objective_scale)
scaled_objective = partial(functions.normalized_cir_objective, objective_with_components=objective_with_components, normalized_joint_loss=normalized_joint_loss)


# === CONSTRAINED JOINT CALIBRATION ===
calibration_result = functions.calibrate_cubic_with_active_constraints(scaled_objective, calibration_parameters0, dimension_ell,
    variance_signature_features[:N_variance_constraint], sigX_truncated[:N_variance_constraint], variance_pair_positions,
    variance_shuffle_coordinates, window, minimum_variance=minimum_variance + variance_safety_margin,
    initial_future_loadings=initial_future_loadings, minimum_initial_future=minimum_initial_future, **optimizer_options)
calibration_parameters_optimal = np.asarray(calibration_result.x, dtype=float)
ell_optimal = calibration_parameters_optimal[:dimension_ell]
rho_b_optimal = functions.rho_b_from_parameters(calibration_parameters_optimal[dimension_ell:], rho_primary)
final_loss = float(calibration_result.fun)
final_minimum_variance = minimum_variance + variance_safety_margin + float(variance_constraint(calibration_parameters_optimal))


# === FINAL CALIBRATION RESULTS ===
print(f"Optimizer success: {calibration_result.success}")
print(f"Optimizer message: {calibration_result.message}")
print(f"Final joint loss: {final_loss:.8e}")
print(f"Final minimum raw variance: {final_minimum_variance:.8e}")
print("Optimal ell:")
print(ell_optimal)
print("Optimal rho_b:")
print(rho_b_optimal)


# === IMPLIED-VOLATILITY PLOTS ===
_, final_calibration_details = objective_with_components(calibration_parameters_optimal)
final_model_prices = final_calibration_details["model_prices"]
figure, axes = functions.plot_calibration_smiles(final_model_prices["SPX"], final_model_prices["VIX"], final_model_prices["VXX"],
    final_calibration_details["model_future_vix"], strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx,
    bid_iv_spx, ask_iv_spx, bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx, T_spx, T_vix, T_vxx, r, cp_spx, cp_vix, cp_vxx)
