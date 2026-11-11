import importlib
from functools import partial
import numpy as np
import functions_level3_enrichment_CIR as functions
from market_data_private_CIR import (
    valuation_date, previous_vix_future_expiry, futures_maturities, T_spx, T_vix, T_vxx,
    spx_spot, vix_spot, vxx_spot, r,
    strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx,
    strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix,
    strikes_vxx, bid_vxx, ask_vxx)

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
positivity_constraint_maturity = 84 / 365
minimum_variance = 0.0
variance_safety_margin = 1e-6
minimum_loss_normalization = 1e-12
beta_spx, beta_vix, beta_vxx = (0.98, 0.1, 0.5)
lambda_spx, lambda_vix, lambda_vxx = (0.25, 0.45, 0.3)
normalization_power = 0.5
Tmax = float(max(np.max(market_maturities), positivity_constraint_maturity))
nsteps = int(round(daily_observations * Tmax * 365))


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
    driver_steps=int(round(T_spx * 365 * daily_observations)), return_brownian_increments=False
)
sigX_truncated, variance_signature_features = functions.compute_cubic_signature_features(paths, selected_signature_indices, reduced_variance_model)
del paths
dt = float(time_grid[1] - time_grid[0])
market_indices = np.array([np.argmin(np.abs(time_grid - maturity)) for maturity in market_maturities], dtype=int)
N_spx, N_vix, N_vxx = map(int, market_indices)
positivity_constraint_index = int(np.argmin(np.abs(time_grid - positivity_constraint_maturity)))
N_variance_constraint = positivity_constraint_index + 1
time_steps_vxx = time_grid[:N_vxx + 1]


# === FORWARD REFERENCES ===
forward_spx = functions.forward_from_put_call_parity(strikes_spx, bid_spx, ask_spx, put_bid_spx, put_ask_spx, T_spx, r)
forward_vix = functions.forward_from_put_call_parity(strikes_vix, bid_vix, ask_vix, put_bid_vix, put_ask_vix, T_vix, r)
forward_vxx = vxx_spot * np.exp(r * T_vxx)


# === EUROPEAN CALL CALIBRATION QUOTES ===
cp_spx = np.ones_like(strikes_spx, dtype=int)
cp_vix = np.ones_like(strikes_vix, dtype=int)
cp_vxx = np.ones_like(strikes_vxx, dtype=int)
bid_iv_spx = functions.invert_prices(bid_spx, forward_spx, strikes_spx, T_spx, r=r)
ask_iv_spx = functions.invert_prices(ask_spx, forward_spx, strikes_spx, T_spx, r=r)
bid_iv_vix = functions.invert_prices(bid_vix, forward_vix, strikes_vix, T_vix, r=r)
ask_iv_vix = functions.invert_prices(ask_vix, forward_vix, strikes_vix, T_vix, r=r)
bid_iv_vxx = functions.invert_prices(bid_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)
ask_iv_vxx = functions.invert_prices(ask_vxx, forward_vxx, strikes_vxx, T_vxx, r=r)


# === LOSS NORMALIZATIONS AND FUTURES PROPAGATORS ===
normalization_spx = functions.precompute_loss_normalization(forward_spx, strikes_spx, T_spx, r, bid_iv_spx, ask_iv_spx, minimum_loss_normalization)
normalization_vix = functions.precompute_loss_normalization(forward_vix, strikes_vix, T_vix, r, bid_iv_vix, ask_iv_vix, minimum_loss_normalization)
normalization_vxx = functions.precompute_loss_normalization(forward_vxx, strikes_vxx, T_vxx, r, bid_iv_vxx, ask_iv_vxx, minimum_loss_normalization)
mid_iv_spx = 0.5 * (bid_iv_spx + ask_iv_spx)
mid_iv_vix = 0.5 * (bid_iv_vix + ask_iv_vix)
deltas_spx = np.array([functions.black_call_delta(forward_spx, strike, T_spx, r, vol) for strike, vol in zip(strikes_spx, mid_iv_spx)])
deltas_vix = np.array([functions.black_call_delta(forward_vix, strike, T_vix, r, vol) for strike, vol in zip(strikes_vix, mid_iv_vix)])
vix_control_loadings = functions.expected_selected_signature_on_cir_grid(kappa, theta, eta, rho_ou, x0,
    kappa_cir, theta_cir, eta_cir, y0_cir, time_grid[:N_vix + 1], selected_signature_indices)
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)


# === INITIAL FUTURES CONSTRAINTS ===
required_future_maturities = np.unique(np.array([T_vix] + [maturity for segment in vxx_roll_data for maturity in segment[2:4]]))
initial_future_loadings = np.stack([functions.precompute_vix_future_propagators(np.array([0.0]), maturity, G_n)[0]
    @ sigX_truncated[0, 0] for maturity in required_future_maturities])
minimum_initial_future = future_floor + future_initial_safety_margin
initial_future_constraint = partial(functions.initial_future_constraint, initial_future_loadings=initial_future_loadings,
    minimum_initial_future=minimum_initial_future)


# === VARIANCE CONSTRAINT ===
variance_constraint_inputs = {
    "shuffle_tensor_nn": None, "N": N_variance_constraint, "variance_signature_features": variance_signature_features,
    "window": window, "minimum_variance": minimum_variance + variance_safety_margin,
    "variance_pair_positions": variance_pair_positions, "variance_shuffle_coordinates": variance_shuffle_coordinates,
}
variance_constraint = partial(functions.variance_positivity_constraint_with_correlations, dimension_ell=dimension_ell,
    constraint_inputs=variance_constraint_inputs)


# === OBJECTIVE INPUTS ===
objective_model_inputs = {
    "shuffle_tensor_nn": None, "variance_signature_features": variance_signature_features,
    "variance_pair_positions": variance_pair_positions, "variance_shuffle_coordinates": variance_shuffle_coordinates,
    "truncated_signature_path": sigX_truncated, "window": window, "dt": dt,
    "N_spx": N_spx, "N_vix": N_vix, "time_steps_vxx": time_steps_vxx,
    "vix_future_propagator": vix_future_propagator, "vxx_roll_data": vxx_roll_data,
    "forward_spx": forward_spx, "vxx_spot": vxx_spot, "leverage_cap": leverage_cap,
    "future_floor": future_floor, "vxx_annual_fee": vxx_annual_fee,
    "use_price_control_variates": False, "vix_control_loadings": vix_control_loadings, "collect_gradient_inputs": True,
}
objective_market_inputs = {
    "strikes_spx": strikes_spx, "strikes_vix": strikes_vix, "strikes_vxx": strikes_vxx,
    "cp_spx": cp_spx, "cp_vix": cp_vix, "cp_vxx": cp_vxx,
    "maturity_spx": T_spx, "maturity_vix": T_vix, "maturity_vxx": T_vxx, "rate": r,
    "bid_spx": bid_spx, "ask_spx": ask_spx, "bid_vix": bid_vix, "ask_vix": ask_vix, "bid_vxx": bid_vxx, "ask_vxx": ask_vxx,
}
objective_loss_inputs = {
    "normalization_spx": normalization_spx, "normalization_vix": normalization_vix, "normalization_vxx": normalization_vxx,
    "deltas_spx": deltas_spx, "deltas_vix": deltas_vix, "market_future_vix": forward_vix,
    "beta_spx": beta_spx, "beta_vix": beta_vix, "beta_vxx": beta_vxx,
    "lambda_spx": lambda_spx, "lambda_vix": lambda_vix, "lambda_vxx": lambda_vxx,
    "variance_softness": variance_softness, "variance_treatment": variance_treatment,
}
objective_inputs = {**objective_model_inputs, **objective_market_inputs, **objective_loss_inputs}
brownian_inputs = {
    "dimension_ell": dimension_ell, "rho_primary": rho_primary, "driver_increments": primary_brownian_increments,
    "independent_normals": independent_brownian_normals, "objective_inputs": objective_inputs,
}
objective_with_components = partial(functions.joint_calibration_objective_with_cir_correlations, return_components=True, **brownian_inputs)


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
initial_variance_margin = variance_constraint(calibration_parameters0)
if initial_variance_margin < 0.0 or np.min(initial_future_constraint(calibration_parameters0)) < 0.0:
    for _ in range(30):
        initial_stochastic_weight *= 0.5
        ell0 = ell_safe + initial_stochastic_weight * (ell_reference - ell_safe)
        calibration_parameters0[:dimension_ell] = ell0
        initial_variance_margin = variance_constraint(calibration_parameters0)
        if initial_variance_margin >= 0.0 and np.min(initial_future_constraint(calibration_parameters0)) >= 0.0:
            break
    else:
        raise RuntimeError("Could not construct a feasible starting point for the CIR calibration.")


# === NORMALIZED JOINT LOSS ===
initial_raw_loss, initial_block_losses = objective_with_components(calibration_parameters0)
if not np.isfinite(initial_raw_loss) or initial_raw_loss >= 1e12 or any(not np.isfinite(initial_block_losses[product]) or initial_block_losses[product] >= 1e12 for product in ("SPX", "VIX", "VXX")):
    raise RuntimeError("Invalid initial pricing loss in the CIR calibration.")
strike_counts = {"SPX": strikes_spx.size, "VIX": strikes_vix.size, "VXX": strikes_vxx.size}
loss_weights = {"SPX": lambda_spx, "VIX": lambda_vix, "VXX": lambda_vxx}
loss_scales = {product: count * max(initial_block_losses[product] / count, 1e-12) ** normalization_power
    for product, count in strike_counts.items()}
initial_mean_loss = sum(weight * initial_block_losses[product] / loss_scales[product] for product, weight in loss_weights.items())
objective_scale = 1000.0 / max(initial_mean_loss, 1e-12)
normalized_joint_loss = partial(functions.normalized_joint_loss, lambda_spx=lambda_spx, lambda_vix=lambda_vix, lambda_vxx=lambda_vxx,
    loss_scales=loss_scales, objective_scale=objective_scale)
reference_loss, _ = normalized_joint_loss(initial_block_losses)


# === RECOVERED WARM START ===
# Recovered CIR fit for the February 24 snapshot.
ell_warm = np.array([
     0.1820341935794468,    0.45757757113627046,  -0.0450364276223573,
    -0.028551866457687626,  0.19271133785129402,   6.4702720554813995,
     0.5633287806852676,    0.09108857206229698,  -1.3051986088713505,
    -0.5688964205854604,   -0.041991842810198396,  0.008401788736925946,
     0.0022105029867746373, -0.0016763345893431388, -0.0035355337154674912,
     0.005227263212118898, -0.026346155468862015, -0.5857533394753457,
    -0.028889823466329432, -0.02332659066528454,   0.2866086094020981,
     0.2439957729169196,    0.18731552206162885,  -0.959527588969014,
     1.6182306997686515,
], dtype=float)
rho_b_warm = np.array([0.25257050232725603, 0.8039805944306185, -0.5868334268691128], dtype=float)

def sample_feasible(parameters):
    return np.all(initial_future_constraint(parameters) >= 0.0) and variance_constraint(parameters) >= 0.0

if ell_warm.size == dimension_ell and rho_b_warm @ np.linalg.solve(rho_primary, rho_b_warm) < 1.0:
    warm_correlations = functions.parameters_from_rho_b(rho_b_warm, rho_primary)
    recovered_parameters = np.r_[ell_warm, warm_correlations]
    warm_anchor = np.r_[calibration_parameters0[:dimension_ell], warm_correlations]
    warm_parameters = functions.feasible_point_toward_candidate(warm_anchor, recovered_parameters, sample_feasible)
    warm_raw_loss, warm_block_losses = objective_with_components(warm_parameters)
    if np.isfinite(warm_raw_loss) and warm_raw_loss < 1e12 and all(np.isfinite(warm_block_losses[p]) and warm_block_losses[p] < 1e12 for p in loss_weights):
        warm_loss, _ = normalized_joint_loss(warm_block_losses)
        if warm_loss < reference_loss:
            calibration_parameters0 = warm_parameters
            initial_block_losses = warm_block_losses

loss_factors = {product: objective_scale * weight / loss_scales[product] for product, weight in loss_weights.items()}
objective_gradient = partial(functions.joint_calibration_gradient_with_cir_correlations, loss_factors=loss_factors, **brownian_inputs)


# === CONSTRAINED JOINT CALIBRATION ===
scaling_path_stride = max(1, npaths // 256)
scaling_signatures = sigX_truncated[market_indices][:, ::scaling_path_stride, :]
signature_rms = np.sqrt(np.mean(scaling_signatures**2, axis=(0, 1)))
ell_parameter_scales = 0.02 / np.maximum(signature_rms, 1e-6)
correlation_parameter_scales = np.maximum(np.abs(calibration_parameters0[dimension_ell:]), 1.0)
parameter_scales = np.concatenate((ell_parameter_scales, correlation_parameter_scales))
constraint_scale = max(float(atm_spx_iv**2), 1e-4)
optimizer_runs = (
    {"maxiter": 360, "ftol": 3e-8, "eps": 2e-5, "disp": False},
    {"maxiter": 160, "ftol": 3e-8, "eps": 2e-5, "disp": False},
    {"maxiter": 160, "ftol": 3e-9, "eps": 2e-6, "disp": False},
)
# All signature coefficients, including ell[0], are free optimization variables.
_, optimization_progress = functions.calibrate_cir(initial_parameters=calibration_parameters0,
    initial_block_losses=initial_block_losses, objective_with_components=objective_with_components, objective_gradient=objective_gradient,
    normalized_joint_loss=normalized_joint_loss, variance_constraint_inputs=variance_constraint_inputs, constraint_scale=constraint_scale,
    initial_future_loadings=initial_future_loadings, minimum_initial_future=minimum_initial_future, parameter_scales=parameter_scales,
    rho_primary=rho_primary, optimizer_runs=optimizer_runs)
calibration_parameters_optimal = optimization_progress["best_parameters"].copy()
ell_optimal = calibration_parameters_optimal[:dimension_ell]
rho_b_optimal = functions.rho_b_from_parameters(calibration_parameters_optimal[dimension_ell:], rho_primary)


# === FINAL CALIBRATION RESULTS ===
final_variance_margin = variance_constraint(calibration_parameters_optimal)
final_future_margin = float(np.min(initial_future_constraint(calibration_parameters_optimal)))
if not (np.isfinite(final_variance_margin) and final_variance_margin >= 0.0 and np.isfinite(final_future_margin) and final_future_margin >= 0.0):
    raise RuntimeError("Selected fit violates the calibration-sample constraints.")
print("Optimal ell:", ell_optimal, flush=True)
print("Optimal rho_b:", rho_b_optimal, flush=True)
