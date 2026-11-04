import importlib
import numpy as np
import functions_level3_enrichment as functions
from functools import partial
from datetime import date

# Market inputs are loaded from an untracked local file.
from market_data_private import (
    spx_spot,
    vix_spot,
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
rho_b = np.array([0.80656419, -0.00128551, -0.91915343], dtype=float)
x0 = np.array([1, 0.08, 2])
x0_extended = np.insert(x0, 0, 0)


# === LEVEL 3 EXTENSION ===
slow_ou_component = int(np.argmin(kappa))
slow_ou_letter = slow_ou_component + 1
slow_cubic_word = [slow_ou_letter, slow_ou_letter, slow_ou_letter]
slow_cubic_index = functions.from_word_to_num(slow_cubic_word, d)
time_letter = 0
time_cubic_word = [time_letter, time_letter, time_letter]
time_cubic_index = functions.from_word_to_num(time_cubic_word, d)


# === MARKET MATURITIES AND CONSTRAINTS ===
futures_maturities = np.array([22, 50, 84, 113]) / 365
valuation_date = date(2026, 2, 24)
previous_vix_future_expiry = date(2026, 2, 18)
previous_expiry = (previous_vix_future_expiry - valuation_date).days / 365.0
leverage_cap = 100.0
future_floor = 1e-2
future_initial_safety_margin = 1e-8
vxx_annual_fee = 0.0089
positivity_constraint_maturity = 84.0 / 365.0
T_spx = 24 / 365
T_vix = 22 / 365
T_vxx = 24 / 365
market_maturities = np.array([T_spx, T_vix, T_vxx])


# === MONTE CARLO SETTINGS ===
daily_observations = 4
Tmax = float(max(np.max(market_maturities), positivity_constraint_maturity))
nsteps = int(round(daily_observations * Tmax * 365))
npaths = 10000
simulation_seed = None


# === SIMULATION OF PATHS AND BROWNIAN INCREMENTS ===
paths, time_grid, factor_brownian_increments, independent_brownian_increments = (functions.simulate_time_extended_ou_3d_with_drivers(kappa, theta, eta, rho_ou, x0, Tmax, nsteps, npaths, seed=simulation_seed))


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
variance_pair_positions = reduced_variance_model["pair_positions"]
variance_shuffle_coordinates = reduced_variance_model["shuffle_product_coordinates"]
sigX_truncated, variance_signature_features = functions.compute_selected_signature_and_variance_features(X=paths,level=3,selected_indices=selected_signature_indices,pair_positions=variance_pair_positions,pair_to_variance_features=reduced_variance_model["pair_to_variance_features"])
del paths


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


# === CALIBRATION SETTINGS - LOSS FUNCTION PARAMETERS ===
beta_spx, beta_vix, beta_vxx = (0.98, 0.9, 0.98)
lambda_spx, lambda_vix, lambda_vxx = (0.25, 0.5, 0.25)
lization_power = 0.5


# === FLAG OPTIONS ===
match_vix_future_exactly = False
use_analytic_objective_gradient = True
use_price_control_variates = True
variance_treatment = "positive_part"


# === INITIALIZATION AND PRECOMPUTATION ===
minimum_lization = 1e-12
dt = float(time_grid[1] - time_grid[0])
market_indices = np.array([np.argmin(np.abs(time_grid - maturity)) for maturity in market_maturities], dtype=int)
positivity_constraint_index = int(np.argmin(np.abs(time_grid - positivity_constraint_maturity)))
N_spx, N_vix, N_vxx = map(int, market_indices)
sigX_truncated = sigX_truncated[: int(np.max(market_indices)) + 1].copy()
factor_brownian_increments = factor_brownian_increments[:N_spx].copy()
independent_brownian_increments = independent_brownian_increments[:N_spx].copy()
N_variance_constraint = positivity_constraint_index + 1
time_steps_vxx = time_grid[: N_vxx + 1]


# == PRECOMPUTATION OF LOSS NORMALIZATION AND DELTAS ===
normalization_spx = functions.precompute_loss_normalization(forward_spx, strikes_spx, T_spx, r, bid_iv_spx, ask_iv_spx, minimum_lization)
normalization_vix = functions.precompute_loss_normalization(forward_vix, strikes_vix, T_vix, r, bid_iv_vix, ask_iv_vix, minimum_lization)
normalization_vxx = functions.precompute_loss_normalization(forward_vxx, strikes_vxx, T_vxx, r, bid_iv_vxx, ask_iv_vxx, minimum_lization)
mid_iv_vix = 0.5 * (bid_iv_vix + ask_iv_vix)
deltas_vix = np.array([functions.black_option_delta(forward_vix, strike, T_vix, r, vol, cp) for strike, vol, cp in zip(strikes_vix, mid_iv_vix, cp_vix)])
mid_iv_spx = 0.5 * (bid_iv_spx + ask_iv_spx)
deltas_spx = np.array([functions.black_option_delta(forward_spx, strike, T_spx, r, vol, cp) for strike, vol, cp in zip(strikes_spx, mid_iv_spx, cp_spx)])
vix_future_propagator = functions.precompute_vix_future_propagators(np.array([0.0]), T_vix, G_n)[0]
vix_future_loadings = vix_future_propagator @ sigX_truncated[0, 0]
vix_control_loadings = functions.expected_selected_signature_on_ou_grid(kappa, theta, eta, rho_ou, x0, time_grid[: N_vix + 1], selected_signature_indices)
vxx_roll_data = functions.precompute_vxx_roll_data(time_steps_vxx, futures_maturities, G_n, previous_expiry=previous_expiry)
required_future_maturities = np.unique(np.array([T_vix] + [maturity for segment in vxx_roll_data for maturity in segment[2:4]]))
minimum_initial_future = future_floor + future_initial_safety_margin


# === INITIALIZATION FUTURES ===
initial_future_loadings = np.stack([functions.precompute_vix_future_propagators(np.array([0.0]), maturity, G_n)[0] @ sigX_truncated[0, 0] for maturity in required_future_maturities])
initial_future_constraint = partial(functions.initial_future_constraint,dimension_ell=dimension_ell,initial_future_loadings=initial_future_loadings,minimum_initial_future=minimum_initial_future,)
initial_future_constraint_jacobian = np.zeros((required_future_maturities.size, dimension_ell + d))
initial_future_constraint_jacobian[:, :dimension_ell] = initial_future_loadings
initial_roll_length = float(futures_maturities[0] - previous_expiry)
initial_front_weight = float(futures_maturities[0] / initial_roll_length)
initial_second_weight = float(-previous_expiry / initial_roll_length)


# === PRINT INFO (optional) ===
print(f"Leverage saturation cap L: {leverage_cap:.4f} ({100.0 * leverage_cap:.1f}% annual volatility)",flush=True)
print(f"Historical previous VIX-future expiry: {previous_vix_future_expiry} | Model T0 = {previous_expiry:.8f} years ({365.0 * previous_expiry:.0f} days)",flush=True)
print(f"Initial roll weights | front = {100.0 * initial_front_weight:.4f}% | second = {100.0 * initial_second_weight:.4f}%",flush=True)
print(f"Futures: primo hit sulla griglia, poi congelati a m | m={future_floor:.2e} decimale ({100.0 * future_floor:.2e} punti)",flush=True)
print("Opzioni VIX: valore assoluto della proxy lineare, senza floor | Def.3.24 e Cor.3.28", flush=True)
print(f"VXX annual fee: {100.0 * vxx_annual_fee:.4f}%", flush=True)


# == DICTONARY OF INPUTS FOR OBJECTIVE FUNCTION AND CONSTRAINTS ===
objective_inputs = {
    "variance_signature_features": variance_signature_features,
    "variance_pair_positions": variance_pair_positions,
    "variance_shuffle_coordinates": variance_shuffle_coordinates,
    "truncated_signature_path": sigX_truncated,
    "window": window,
    "dt": dt,
    "N_spx": N_spx,
    "N_vix": N_vix,
    "time_steps_vxx": time_steps_vxx,
    "vix_future_propagator": vix_future_propagator,
    "vxx_roll_data": vxx_roll_data,
    "forward_spx": forward_spx,
    "vxx_spot": vxx_spot,
    "future_floor": future_floor,
    "vxx_annual_fee": vxx_annual_fee,
    "strikes_spx": strikes_spx,
    "strikes_vix": strikes_vix,
    "strikes_vxx": strikes_vxx,
    "cp_spx": cp_spx,
    "cp_vix": cp_vix,
    "cp_vxx": cp_vxx,
    "maturity_spx": T_spx,
    "maturity_vix": T_vix,
    "maturity_vxx": T_vxx,
    "rate": r,
    "bid_spx": bid_spx,
    "ask_spx": ask_spx,
    "bid_vix": bid_vix,
    "ask_vix": ask_vix,
    "bid_vxx": bid_vxx,
    "ask_vxx": ask_vxx,
    "normalization_spx": normalization_spx,
    "normalization_vix": normalization_vix,
    "normalization_vxx": normalization_vxx,
    "deltas_vix": deltas_vix,
    "deltas_spx": deltas_spx,
    "use_price_control_variates": use_price_control_variates,
    "vix_control_loadings": vix_control_loadings,
    "market_future_vix": forward_vix,
    "beta_spx": beta_spx,
    "beta_vix": beta_vix,
    "beta_vxx": beta_vxx,
    "lambda_spx": lambda_spx,
    "lambda_vix": lambda_vix,
    "lambda_vxx": lambda_vxx,
}
brownian_inputs = {
    "dimension_ell": dimension_ell,
    "rho_ou": rho_ou,
    "factor_brownian_increments": factor_brownian_increments,
    "independent_brownian_increments": independent_brownian_increments,
    "leverage_cap": leverage_cap,
    "objective_inputs": objective_inputs,
}
objective_with_components = partial(functions.joint_calibration_objective_with_saturated_correlations,return_components=True,**brownian_inputs)


# === VARIANCE CONSTRAINT ===
minimum_variance = 0.0
variance_safety_margin = 1e-6
variance_constraint_time_blocks = 12
variance_constraint_inputs = {
    "N": N_variance_constraint,
    "variance_signature_features": variance_signature_features,
    "window": window,
    "minimum_variance": minimum_variance + variance_safety_margin,
    "variance_pair_positions": variance_pair_positions,
    "variance_shuffle_coordinates": variance_shuffle_coordinates,
}

variance_constraint = partial(functions.variance_positivity_constraint_with_correlations,dimension_ell=dimension_ell,constraint_inputs=variance_constraint_inputs)
variance_constraint_with_jacobian = partial(functions.variance_positivity_constraint_with_correlations,dimension_ell=dimension_ell,constraint_inputs=variance_constraint_inputs,return_jacobian=True,number_of_time_blocks=variance_constraint_time_blocks)


# === INITIALIZATION OF ELL PARAMETERS ===
atm_spx_index = int(np.argmin(np.abs(strikes_spx / forward_spx - 1.0)))
atm_spx_iv = 0.5 * (bid_iv_spx[atm_spx_index] + ask_iv_spx[atm_spx_index])
ell_safe = np.zeros(dimension_ell, dtype=float)
ell_safe[0] = atm_spx_iv
ell_safe[1] = (forward_vix / 100.0 - atm_spx_iv) / T_vix
ell_reference_level_two = np.array([
                            0.18651227131172177, -1.4918260961042886, -0.4225469969758481, 0.016758021521028788,
                            0.0632522317884641, 31.752316250038007, 5.561829451300603, -0.2335494061035052,
                            -0.9290206475968611, 5.355090138555593, 1.6512344968996866, -0.020840977507334685,
                            -0.13577056761769396, -0.2233325404497525, -0.03446850511830134, 0.0013534146836617382,
                            0.004280468134303818, -0.93982423970692, -0.15757918994536588, 0.005448697619296834,
                            0.01712767243058856],dtype=float)
ell_reference = np.concatenate((ell_reference_level_two, np.array([0.0, 0.0])))
initial_stochastic_weight = 0.2
ell0 = ell_safe + initial_stochastic_weight * (ell_reference - ell_safe)

if match_vix_future_exactly:
    ell0[0] = (forward_vix / 100.0 - vix_future_loadings[1:] @ ell0[1:]) / vix_future_loadings[0]

correlation_parameters0 = functions.parameters_from_rho_b(rho_b, rho_ou)
calibration_parameters0 = np.concatenate((ell0, correlation_parameters0))
initial_variance_constraint = float(variance_constraint(calibration_parameters0))
initial_minimum_variance = minimum_variance + variance_safety_margin + initial_variance_constraint
print(f"Initial minimum raw variance: {initial_minimum_variance:.8e}", flush=True)


# === CHECK FEASIBILITY OF INITIAL PARAMETERS ===
if initial_variance_constraint < -1e-12 or np.min(initial_future_constraint(calibration_parameters0)) < 0.0:
    for feasibility_attempt in range(30):
        initial_stochastic_weight *= 0.5
        ell0 = ell_safe + initial_stochastic_weight * (ell_reference - ell_safe)
        if match_vix_future_exactly:
            ell0[0] = (forward_vix / 100.0 - vix_future_loadings[1:] @ ell0[1:]) / vix_future_loadings[0]
        calibration_parameters0[:dimension_ell] = ell0
        initial_variance_constraint = float(variance_constraint(calibration_parameters0))
        if (initial_variance_constraint >= 0.0 and np.min(initial_future_constraint(calibration_parameters0)) >= 0.0):
            break
    initial_minimum_variance = minimum_variance + variance_safety_margin + initial_variance_constraint
    print(f"Feasible starting weight: {initial_stochastic_weight:.8f} | minimum raw variance = {initial_minimum_variance:.8e}",flush=True)


# === INITIAL LOSS AND CONTRIBUTIONS ===
initial_raw_loss, initial_block_losses = objective_with_components(calibration_parameters0)
loss_scales = {
    "SPX": float(strikes_spx.size)
    * max(float(initial_block_losses["SPX"]) / strikes_spx.size, 1e-12) ** lization_power,
    "VIX": float(strikes_vix.size)
    * max(float(initial_block_losses["VIX"]) / strikes_vix.size, 1e-12) ** lization_power,
    "VXX": float(strikes_vxx.size)
    * max(float(initial_block_losses["VXX"]) / strikes_vxx.size, 1e-12) ** lization_power,
}

initial_mean_loss = (lambda_spx * initial_block_losses["SPX"] / loss_scales["SPX"] + lambda_vix * initial_block_losses["VIX"] / loss_scales["VIX"] + lambda_vxx * initial_block_losses["VXX"] / loss_scales["VXX"])
objective_scale = 1000.0 / max(float(initial_mean_loss), 1e-12)
normalized_joint_loss = partial(functions.normalized_joint_loss,lambda_spx=lambda_spx,lambda_vix=lambda_vix,lambda_vxx=lambda_vxx,loss_scales=loss_scales,objective_scale=objective_scale)
initial_loss, initial_contributions = normalized_joint_loss(initial_block_losses)


# === PRINT INITIAL LOSS AND CONTRIBUTIONS (optional) ===
print(f"Initial normalized loss | joint = {initial_loss:.8e} | SPX = {initial_contributions['SPX']:.8e} | VIX = {initial_contributions['VIX']:.8e} | VXX = {initial_contributions['VXX']:.8e}",flush=True,)
print(f"Initial raw block losses | SPX = {initial_block_losses['SPX']:.8e} | VIX = {initial_block_losses['VIX']:.8e} | VXX = {initial_block_losses['VXX']:.8e}",flush=True)
print(f"Loss normalization power: {lization_power:.2f} | positivity horizon: {365.0 * positivity_constraint_maturity:g} days",flush=True)
print(f"Initial rho_b: {initial_block_losses['rho_b']}", flush=True)
print(f"Initial saturated SPX points: {100.0 * initial_block_losses['leverage_saturated_fraction']:.6f}%",flush=True)
_, initial_floor_diagnostics = functions.vxx_path(vxx_spot,ell0,time_steps_vxx,sigX_truncated[: time_steps_vxx.size],vxx_roll_data,future_floor=future_floor,return_diagnostics=True)
print(f"Initial stopped roll-future observations: {initial_floor_diagnostics['floored_future_values']} / {initial_floor_diagnostics['evaluated_future_values']} ({100.0 * initial_floor_diagnostics['floored_future_fraction']:.6f}%)",flush=True)


# == SCALING OF PARAMETERS FOR OPTIMIZATION === 
scaling_path_stride = max(1, npaths // 256)
scaling_signatures = sigX_truncated[market_indices][:, ::scaling_path_stride, :]
signature_rms = np.sqrt(np.mean(scaling_signatures**2, axis=(0, 1)))
ell_parameter_scales = 0.02 / np.maximum(signature_rms, 1e-6)
correlation_parameter_scales = np.maximum(np.abs(correlation_parameters0), 1.0)
parameter_scales = np.concatenate((ell_parameter_scales, correlation_parameter_scales))


# === PARAMETERS IN CASE WE OPTIMIZE L0 OR NOT ===
free_parameter_indices = np.arange(1 if match_vix_future_exactly else 0, calibration_parameters0.size)
parameter_directions = np.diag(parameter_scales)[:, free_parameter_indices]
if match_vix_future_exactly:
    parameter_directions[0, : dimension_ell - 1] = (-vix_future_loadings[1:] * parameter_scales[1:dimension_ell] / vix_future_loadings[0])
constraint_scale = max(float(atm_spx_iv**2), 0.0001)
feasibility_tolerance = 1e-09

# SLSQP settings
optimizer_options = {"maxiter": 360, "ftol": 3e-08, "eps": 2e-05, "disp": False}
optimizer_restart_options = {"maxiter": 160, "ftol": 3e-08, "eps": 2e-05, "disp": False}
optimizer_refinement_options = {"maxiter": 160, "ftol": 3e-09, "eps": 2e-06, "disp": False}
optimizer_runs = (optimizer_options, optimizer_restart_options, optimizer_refinement_options)
optimizer_run_names = ("initial descent", "correlation restart", "best-point refinement")
correlation_restart_shrink = 0.9
calibration_result, optimization_progress = functions.calibrate(
    brownian_inputs=brownian_inputs,
    calibration_parameters0=calibration_parameters0,
    constraint_scale=constraint_scale,
    correlation_restart_shrink=correlation_restart_shrink,
    dimension_ell=dimension_ell,
    feasibility_tolerance=feasibility_tolerance,
    free_parameter_indices=free_parameter_indices,
    initial_block_losses=initial_block_losses,
    initial_future_constraint=initial_future_constraint,
    initial_future_constraint_jacobian=initial_future_constraint_jacobian,
    initial_loss=initial_loss,
    lambda_spx=lambda_spx,
    lambda_vix=lambda_vix,
    lambda_vxx=lambda_vxx,
    loss_scales=loss_scales,
    normalized_joint_loss=normalized_joint_loss,
    objective_scale=objective_scale,
    objective_with_components=objective_with_components,
    optimizer_run_names=optimizer_run_names,
    optimizer_runs=optimizer_runs,
    parameter_directions=parameter_directions,
    parameter_scales=parameter_scales,
    rho_ou=rho_ou,
    use_analytic_objective_gradient=use_analytic_objective_gradient,
    variance_constraint_with_jacobian=variance_constraint_with_jacobian,
)


# === FINAL RESULTS ===
calibration_parameters_optimal = optimization_progress["best_parameters"].copy()
ell_optimal = calibration_parameters_optimal[:dimension_ell]
rho_b_optimal = functions.rho_b_from_parameters(calibration_parameters_optimal[dimension_ell:], rho_ou)
final_loss = float(optimization_progress["best_loss"])
final_variance_constraint = float(variance_constraint(calibration_parameters_optimal))
final_minimum_variance = minimum_variance + variance_safety_margin + final_variance_constraint


# === PRINT FINAL RESULTS (optional) ===
print(f"Last joint optimizer success: {calibration_result.success}", flush=True)
print(f"Last joint optimizer message: {calibration_result.message}", flush=True)
print("Selected result: best feasible accepted iterate across all SLSQP runs", flush=True)
print(f"Final joint loss on the original Monte Carlo sample: {final_loss:.8e}", flush=True)
print(f"Final minimum raw variance: {final_minimum_variance:.8e}", flush=True)
print("Optimal ell:", flush=True)
print(ell_optimal, flush=True)
print("Optimal rho_b:", flush=True)
print(rho_b_optimal, flush=True)


# === CHECK FINAL VARIANCE POSITIVITY ===
final_ell_shuffle = functions.reduced_shuffle_product_ell(ell_optimal, variance_pair_positions, variance_shuffle_coordinates)
final_raw_variance = functions.variance_path(final_ell_shuffle, N_variance_constraint, variance_signature_features, window)
negative_variance_mask = final_raw_variance < 0.0
negative_variance_count = int(np.count_nonzero(negative_variance_mask))
negative_variance_fraction = float(np.mean(negative_variance_mask))
print(f"Negative raw-variance points: {negative_variance_count} / {final_raw_variance.size}", flush=True)
print(f"Negative raw-variance fraction: {100.0 * negative_variance_fraction:.6f}%", flush=True)