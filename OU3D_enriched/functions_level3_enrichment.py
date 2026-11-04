import math

import iisignature
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import ndtr
from tqdm import tqdm

PAPER_PRICING_VERSION = "ou_fit_abs_vix_stopped_futures_v1"
VIX_PROXY = "absolute_linear_signature"
VIX_FUTURE_PROXY = "stopped_signed_conditional_signature"
FUTURE_STOPPING_SCHEME = "first_grid_hit_freeze_at_m"

# Encode a word in level-first lexicographic order; the empty word is zero.
def from_word_to_num(word, d_minus_one):

    if len(word) == 0:
        return 0
    word = np.asarray(word, dtype=int)
    powers = np.flip(np.array([(d_minus_one + 1) ** i for i in range(len(word))], dtype=float))
    all0but1 = np.ones(len(word))
    all0but1[-1] = 0

    return int(np.dot(word + all0but1, powers)) + 1

# Decode a level-first signature coordinate into its word.
def from_num_to_word(n, d_minus_one):

    if n == 0:
        return []

    return from_num_to_word((n - 1) // (d_minus_one + 1), d_minus_one) + [(n - 1) % (d_minus_one + 1)]

# Return the last coordinate index at the specified level.
def from_level_to_num(level, d_minus_one):

    return from_word_to_num(d_minus_one * np.ones(level, dtype=int), d_minus_one)

# Enumerate all shuffles, retaining multiplicities of repeated words.
def shuffle(a, b):

    if len(a) == 0:
        return [b]
    if len(b) == 0:
        return [a]
    out = []
    out.extend((p + [a[-1]] for p in shuffle(a[:-1], b)))
    out.extend((p + [b[-1]] for p in shuffle(a, b[:-1])))

    return out

# Build the dense shuffle multiplication tensor for two truncation levels.
def shuffle_tensor(level_1, level_2, d_minus_one):

    dim1 = from_level_to_num(max(level_1, level_2), d_minus_one) + 1
    dim2 = from_level_to_num(min(level_1, level_2), d_minus_one) + 1
    dim_out = from_level_to_num(level_1 + level_2, d_minus_one) + 1
    tensor = np.zeros((dim1, dim2, dim_out))
    tensor[0, 0, 0] = 1
    for i in range(1, dim1):
        tensor[i, 0, i] = 1
        if i < dim2:
            tensor[0, i, i] = 1
        for j in range(1, min(i + 1, dim2)):
            for word in shuffle(from_num_to_word(i, d_minus_one), from_num_to_word(j, d_minus_one)):
                tensor[i, j, from_word_to_num(word, d_minus_one)] += 1
                if i != j and i < dim2:
                    tensor[j, i, from_word_to_num(word, d_minus_one)] += 1
    if level_2 > level_1:
        tensor = np.swapaxes(tensor, 0, 1)

    return tensor

# Restrict the signature generator and return the projection residual.
def restricted_signature_generator(coeff, x0, base_level, extra_words, dim):
    normalized_extra_words = []
    for word in extra_words:
        word = np.asarray(word, dtype=int)
        normalized_extra_words.append(word.tolist())
    base_dimension = from_level_to_num(base_level, dim - 1) + 1
    base_indices = np.arange(base_dimension, dtype=int)
    extra_indices = np.array([from_word_to_num(word, dim - 1) for word in normalized_extra_words], dtype=int)
    selected_indices = np.concatenate((base_indices, extra_indices))
    maximum_level = max([base_level] + [len(word) for word in normalized_extra_words])
    full_generator = matrix_L(coeff, x0, maximum_level, dim)
    excluded_indices = np.setdiff1d(np.arange(full_generator.shape[0]), selected_indices, assume_unique=True)
    closure_block = full_generator[np.ix_(selected_indices, excluded_indices)]
    denominator = max(float(np.linalg.norm(full_generator[selected_indices], ord="fro")), 1.0)
    relative_closure_error = float(np.linalg.norm(closure_block, ord="fro") / denominator)
    restricted_generator = full_generator[np.ix_(selected_indices, selected_indices)]

    return (selected_indices, restricted_generator, relative_closure_error)

# Express a quadratic polynomial in increment-signature coordinates around x0.
def poly_to_sig(c0, c1, c2, x0):
    c2sym = (c2 + c2.T) / 2
    csig = c0 + np.dot(c1, x0) + np.tensordot(np.tensordot(x0, x0, axes=0), c2sym, axes=([0, 1], [0, 1]))
    csig = np.concatenate((np.atleast_1d(csig), c1 + 2 * np.dot(x0, c2sym)), axis=None)
    csig = np.concatenate((csig, 2 * np.squeeze(np.asarray(c2sym))), axis=None)
    return csig

# Convert polynomial drift and covariance coefficients to signature coordinates.
def poly_to_sig_vec(coeff, x0):
    b_const, b_lin, a_const, a_lin, a_quad = coeff
    dim = len(b_const)
    bsig = np.zeros((dim, 1 + dim + dim**2))
    asig = np.zeros((dim, dim, 1 + dim + dim**2))
    for i in range(dim):
        bsig[i] = poly_to_sig(b_const[i], b_lin[i, :], np.zeros((dim, dim)), x0)
        for j in range(dim):
            asig[i, j] = poly_to_sig(a_const[i, j], a_lin[i, j, :], a_quad[i, j, :], x0)
    return (bsig, asig)

# Build the generator acting on expected signature columns: dE[S]/dt = G E[S].
def matrix_L(coeff, x0, level, dim):
    d_minus_one = dim - 1
    dim_l = from_level_to_num(level, d_minus_one) + 1
    sigcoeff = poly_to_sig_vec(coeff, x0)
    shuffletens = shuffle_tensor(2, level - 1, d_minus_one)[:, :, :dim_l]
    out = np.zeros((dim_l, dim_l))
    for n in range(1, dim_l):
        word = from_num_to_word(n, d_minus_one)
        numprime = from_word_to_num(word[:-1], d_minus_one)
        row = np.tensordot(sigcoeff[0][word[-1]], shuffletens, axes=([-1], [0]))[numprime, :]
        if len(word) > 1:
            numsecond = from_word_to_num(word[:-2], d_minus_one)
            row += (0.5 * np.tensordot(sigcoeff[1][tuple(word[-2:])], shuffletens, axes=([-1], [0]))[numsecond, :])
        out[n, :] = row
    return out

# Compute A = integral_0^window exp(s G) ds and its inverse via a block exponential.
def integrated_signature_operator(G, window):
    n = G.shape[0]
    z = np.zeros_like(G)
    block_matrix = np.block([[G, np.eye(n)], [z, z]])
    block_exponential = expm(window * block_matrix)
    A = block_exponential[:n, n:]
    inverse_A = np.linalg.solve(A, np.eye(n))
    return (A, inverse_A)

# Build the signature generator sparsely, retaining shuffle multiplicities.
def sparse_matrix_L(coeff, x0, level, dim):
    d_minus_one = dim - 1
    generator_dimension = from_level_to_num(level, d_minus_one) + 1
    drift_signature, diffusion_signature = poly_to_sig_vec(coeff, x0)
    row_indices, column_indices, values = ([], [], [])
    for row_index in range(1, generator_dimension):
        word = from_num_to_word(row_index, d_minus_one)
        row_entries = {}

        # Add shuffle contributions to the current sparse generator row.
        def accumulate_shuffle(coefficients, prefix, scale):
            for coefficient_index in np.flatnonzero(coefficients):
                coefficient_word = from_num_to_word(int(coefficient_index), d_minus_one)
                coefficient_value = scale * coefficients[coefficient_index]
                for shuffled_word in shuffle(coefficient_word, prefix):
                    column_index = from_word_to_num(shuffled_word, d_minus_one)
                    if column_index < generator_dimension:
                        row_entries[column_index] = row_entries.get(column_index, 0.0) + coefficient_value

        accumulate_shuffle(drift_signature[word[-1]], word[:-1], 1.0)
        if len(word) > 1:
            accumulate_shuffle(diffusion_signature[tuple(word[-2:])], word[:-2], 0.5)
        for column_index, value in row_entries.items():
            if value != 0.0:
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(value)
    generator = csr_matrix((values, (row_indices, column_indices)), shape=(generator_dimension, generator_dimension), dtype=float)
    generator.sum_duplicates()
    generator.eliminate_zeros()
    return generator

# Build the sparse products of each unordered pair of selected words.
def selected_shuffle_product_matrix(selected_indices, d_minus_one):
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_words = [from_num_to_word(int(index), d_minus_one) for index in selected_indices]
    maximum_level = 2 * max((len(word) for word in selected_words))
    output_dimension = from_level_to_num(maximum_level, d_minus_one) + 1
    row_indices, column_indices, values, pair_positions = ([], [], [], [])
    product_row = 0
    for left_position, left_word in enumerate(selected_words):
        for right_position in range(left_position, len(selected_words)):
            pair_positions.append((left_position, right_position))
            right_word = selected_words[right_position]
            for shuffled_word in shuffle(left_word, right_word):
                row_indices.append(product_row)
                column_indices.append(from_word_to_num(shuffled_word, d_minus_one))
                values.append(1.0)
            product_row += 1
    shuffle_products = csr_matrix((values, (row_indices, column_indices)), shape=(product_row, output_dimension), dtype=float)
    shuffle_products.sum_duplicates()
    shuffle_products.eliminate_zeros()
    return (shuffle_products, np.asarray(pair_positions, dtype=int))

# Enlarge a row span until it is invariant under right multiplication by G.
def invariant_row_closure(initial_rows, generator, tolerance=1e-11, max_iterations=20):

    initial_dense = np.asarray(initial_rows.toarray(), dtype=float)
    _, singular_values, basis_rows = np.linalg.svd(initial_dense, full_matrices=False)
    initial_threshold = tolerance * max(float(singular_values[0]), 1.0)
    initial_rank = int(np.count_nonzero(singular_values > initial_threshold))
    basis_rows = basis_rows[:initial_rank]
    rank_history = [initial_rank]
    for _ in range(max_iterations):
        generator_images = np.asarray(generator.T.dot(basis_rows.T).T, dtype=float)
        residual = generator_images - generator_images @ basis_rows.T @ basis_rows
        residual -= residual @ basis_rows.T @ basis_rows
        _, new_singular_values, new_directions = np.linalg.svd(residual, full_matrices=False)
        residual_threshold = tolerance * max(float(np.linalg.norm(generator_images, ord="fro")), 1.0)
        new_rank = int(np.count_nonzero(new_singular_values > residual_threshold))
        if new_rank == 0:
            break
        basis_rows = np.vstack((basis_rows, new_directions[:new_rank]))
        basis_rows = np.linalg.qr(basis_rows.T, mode="reduced")[0].T
        rank_history.append(basis_rows.shape[0])
    generator_images = np.asarray(generator.T.dot(basis_rows.T).T, dtype=float)
    reduced_generator = generator_images @ basis_rows.T
    closure_residual = generator_images - reduced_generator @ basis_rows
    closure_error = float(np.linalg.norm(closure_residual, ord="fro") / max(float(np.linalg.norm(generator_images, ord="fro")), 1.0))

    return (basis_rows, reduced_generator, closure_error, rank_history)

# Build the invariant shuffle-product space and inverse-window feature maps.
def prepare_reduced_variance_model(coeff, x0, selected_indices, dim, window, tolerance=1e-11, max_iterations=20):

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_words = [from_num_to_word(int(index), dim - 1) for index in selected_indices]
    maximum_level = 2 * max((len(word) for word in selected_words))
    full_generator = sparse_matrix_L(coeff, x0, maximum_level, dim)
    shuffle_products, pair_positions = selected_shuffle_product_matrix(selected_indices, dim - 1)
    basis_rows, reduced_generator, closure_error, rank_history = invariant_row_closure(shuffle_products, full_generator, tolerance, max_iterations)
    integrated_operator, inverse_integrated_operator = integrated_signature_operator(reduced_generator, window)
    shuffle_product_coordinates = np.asarray(shuffle_products @ basis_rows.T, dtype=float)
    pair_to_reduced_map = np.linalg.pinv(shuffle_product_coordinates, rcond=tolerance)
    pair_map_condition_number = float(np.linalg.cond(shuffle_product_coordinates))
    pair_map_inverse_residual = float(np.linalg.norm(pair_to_reduced_map @ shuffle_product_coordinates - np.eye(reduced_generator.shape[0]), ord=np.inf))
    pair_to_variance_features = inverse_integrated_operator @ pair_to_reduced_map

    return {
        "pair_positions": pair_positions,
        "shuffle_product_coordinates": shuffle_product_coordinates,
        "pair_to_variance_features": pair_to_variance_features,
        "generator": reduced_generator,
        "integrated_operator": integrated_operator,
        "inverse_integrated_operator": inverse_integrated_operator,
        "dimension": int(reduced_generator.shape[0]),
        "full_level": maximum_level,
        "closure_error": closure_error,
        "rank_history": rank_history,
        "pair_map_condition_number": pair_map_condition_number,
        "pair_map_inverse_residual": pair_map_inverse_residual
    }

# Project ell shuffle ell into the reduced basis, doubling off-diagonal pairs.
def reduced_shuffle_product_ell(ell, pair_positions, shuffle_product_coordinates):
    ell = np.asarray(ell, dtype=float)
    pair_positions = np.asarray(pair_positions, dtype=int)
    shuffle_product_coordinates = np.asarray(shuffle_product_coordinates, dtype=float)
    pair_weights = ell[pair_positions[:, 0]] * ell[pair_positions[:, 1]]
    pair_weights[pair_positions[:, 0] != pair_positions[:, 1]] *= 2.0
    return pair_weights @ shuffle_product_coordinates


# Simulate exact OU transitions jointly with factor and independent Brownian increments.
def simulate_time_extended_ou_3d_with_drivers(kappa, theta, eta, rho_ou, y0, Tmax, nsteps, npaths, seed=None):
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    eta = np.asarray(eta, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    dt = Tmax / nsteps
    time_grid = np.linspace(0.0, Tmax, nsteps + 1)
    decay = np.exp(-kappa * dt)
    kappa_sum = kappa[:, None] + kappa[None, :]
    ou_covariance = rho_ou * np.outer(eta, eta) * -np.expm1(-kappa_sum * dt) / kappa_sum
    cross_covariance = (eta * -np.expm1(-kappa * dt) / kappa)[:, None] * rho_ou
    joint_covariance = np.block([[ou_covariance, cross_covariance], [cross_covariance.T, dt * rho_ou]])
    joint_cholesky = np.linalg.cholesky(0.5 * (joint_covariance + joint_covariance.T))
    rng = np.random.default_rng(seed)
    paths = np.empty((nsteps + 1, npaths, 4), dtype=float)
    paths[:, :, 0] = time_grid[:, None]
    paths[0, :, 1:] = y0
    factor_increments = np.empty((nsteps, npaths, 3), dtype=float)
    independent_increments = np.empty((nsteps, npaths), dtype=float)
    for step in range(nsteps):
        normals = rng.standard_normal((npaths, 7))
        joint_innovations = normals[:, :6] @ joint_cholesky.T
        paths[step + 1, :, 1:] = theta + (paths[step, :, 1:] - theta) * decay + joint_innovations[:, :3]
        factor_increments[step] = joint_innovations[:, 3:]
        independent_increments[step] = np.sqrt(dt) * normals[:, 6]
    return (paths, time_grid, factor_increments, independent_increments)


# Build price Brownian increments using volatility at the left node and capped leverage.
def brownian_increments_with_saturation(rho_b, rho_ou, variance_left, factor_increments, independent_increments, leverage_cap):
    rho_b = np.asarray(rho_b, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    variance_left = np.asarray(variance_left, dtype=float)
    factor_increments = np.asarray(factor_increments, dtype=float)
    independent_increments = np.asarray(independent_increments, dtype=float)
    if (not np.all(np.isfinite(variance_left)) or np.any(variance_left < 0.0) or (not np.all(np.isfinite(factor_increments))) or (not np.all(np.isfinite(independent_increments)))):
        return np.full(variance_left.shape, np.nan)
    weights = np.linalg.solve(rho_ou, rho_b)
    q_rho = float(rho_b @ weights)
    q_rho = float(np.clip(q_rho, 0.0, 1.0))
    sigma = np.sqrt(variance_left)
    chi = np.ones_like(sigma)
    np.divide(leverage_cap, sigma, out=chi, where=sigma > leverage_cap)
    correlated_increments = np.einsum("tpi,i->tp", factor_increments, weights, optimize=True)
    independent_weight = np.sqrt(np.maximum(1.0 - chi**2 * q_rho, 0.0))
    return chi * correlated_increments + independent_weight * independent_increments


# Evaluate the joint loss for signature coefficients and admissible price correlations.
def joint_calibration_objective_with_saturated_correlations(calibration_parameters, dimension_ell, rho_ou, factor_brownian_increments, independent_brownian_increments, leverage_cap, objective_inputs, return_components=False):
    calibration_parameters = np.asarray(calibration_parameters, dtype=float)
    if not np.all(np.isfinite(calibration_parameters)):
        invalid_details = {"SPX": 1e12, "VIX": 1e12, "VXX": 1e12, "rho_b": np.full(np.asarray(rho_ou).shape[0], np.nan)}
        return (1e12, invalid_details) if return_components else 1e12
    ell = calibration_parameters[:dimension_ell]
    rho_b = rho_b_from_parameters(calibration_parameters[dimension_ell:], rho_ou)
    result = joint_calibration_objective(
        ell, factor_brownian_increments=factor_brownian_increments, independent_brownian_increments=independent_brownian_increments, rho_b=rho_b, rho_ou=rho_ou,
        leverage_cap=leverage_cap, return_components=return_components, **objective_inputs
    )
    if return_components:
        joint_loss, details = result
        details["rho_b"] = rho_b
        return (joint_loss, details)
    return result


# Map unconstrained parameters into the admissible joint-correlation region.
def rho_b_from_parameters(parameters, rho_ou):
    parameters = np.asarray(parameters, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    cholesky_ou = np.linalg.cholesky(0.5 * (rho_ou + rho_ou.T))
    unit_ball_parameters = parameters / np.sqrt(1.0 + parameters @ parameters)
    rho_b = cholesky_ou @ unit_ball_parameters
    return rho_b


# Invert the interior joint-correlation parametrization.
def parameters_from_rho_b(rho_b, rho_ou):
    rho_b = np.asarray(rho_b, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    cholesky_ou = np.linalg.cholesky(0.5 * (rho_ou + rho_ou.T))
    unit_ball_parameters = np.linalg.solve(cholesky_ou, rho_b)
    residual = 1.0 - unit_ball_parameters @ unit_ball_parameters
    return unit_ball_parameters / np.sqrt(residual)

# Compute selected prefix signatures and compact reduced variance features.
def compute_selected_signature_and_variance_features(X, level, selected_indices, pair_positions, pair_to_variance_features, show_progress=True):
    X = np.asarray(X, dtype=float)
    selected_indices = np.asarray(selected_indices, dtype=int)
    pair_positions = np.asarray(pair_positions, dtype=int)
    pair_to_variance_features = np.asarray(pair_to_variance_features, dtype=float)
    n_times, npaths, dim_x = X.shape
    selected_signatures = np.zeros((n_times, npaths, selected_indices.size), dtype=float)
    selected_signatures[:, :, selected_indices == 0] = 1.0
    variance_signature_features = {"signature_path": selected_signatures,"pair_positions": pair_positions,"pair_to_variance_features": pair_to_variance_features}
    path_iterator = tqdm(range(npaths), disable=not show_progress)
    nonempty_mask = selected_indices > 0
    nonempty_columns = selected_indices[nonempty_mask] - 1
    for path_index in path_iterator:
        signature_prefixes = np.asarray(iisignature.sig(X[:, path_index, :], level, 2), dtype=float)
        selected_signatures[1:, path_index, nonempty_mask] = signature_prefixes[:, nonempty_columns]
    return (selected_signatures, variance_signature_features)


# Evaluate the standard normal cumulative distribution function.
def std_norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))


# Evaluate the standard normal density.
def std_norm_pdf(x):
    x = np.asarray(x)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# Estimate the forward from near-ATM call and put midpoint pairs.
def forward_from_put_call_parity(strikes, call_bid, call_ask, put_bid, put_ask, maturity, rate, n_atm_pairs=3):
    strikes = np.asarray(strikes, dtype=float)
    call_bid = np.asarray(call_bid, dtype=float)
    call_ask = np.asarray(call_ask, dtype=float)
    put_bid = np.asarray(put_bid, dtype=float)
    put_ask = np.asarray(put_ask, dtype=float)
    valid = (np.isfinite(strikes) & np.isfinite(call_bid) & np.isfinite(call_ask) & np.isfinite(put_bid) & np.isfinite(put_ask) & (strikes > 0.0) & (call_bid >= 0.0) & (put_bid >= 0.0) & (call_ask >= call_bid) & (put_ask >= put_bid))
    strikes = strikes[valid]
    call_mid = 0.5 * (call_bid[valid] + call_ask[valid])
    put_mid = 0.5 * (put_bid[valid] + put_ask[valid])
    count = min(int(n_atm_pairs), strikes.size)
    atm = np.argsort(np.abs(call_mid - put_mid), kind="stable")[:count]
    discount = math.exp(-rate * maturity)
    forwards = strikes[atm] + (call_mid[atm] - put_mid[atm]) / discount
    forwards = forwards[np.isfinite(forwards) & (forwards > 0.0)]
    return float(np.median(forwards))

# Price a discounted European call under Black-76.
def black76_call(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    d2 = d1 - vsqrt
    return float(discount * (forward * std_norm_cdf(d1) - strike * std_norm_cdf(d2)))

# Price a discounted European put under Black-76.
def black76_put(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    d2 = d1 - vsqrt
    return float(discount * (strike * std_norm_cdf(-d2) - forward * std_norm_cdf(-d1)))

# Compute Black-76 vega for the supplied discount factor.
def black_vega(discount, forward, strike, maturity, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    return float(discount * forward * std_norm_pdf(d1) * math.sqrt(maturity))

# Compute the discounted Black-76 call delta with respect to the forward.
def black_call_delta(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    return float(math.exp(-rate * maturity) * std_norm_cdf(d1))

# Compute discounted forward delta for a call (+1) or put (-1).
def black_option_delta(forward, strike, maturity, rate, vol, cp=1):
    call_delta = black_call_delta(forward, strike, maturity, rate, vol)
    if not np.isfinite(call_delta):
        return np.nan
    return call_delta if cp == 1 else call_delta - math.exp(-rate * maturity)


# Invert Black-76 by bisection; invalid/unattainable prices produce NaN.
def implied_vol_black76(price, forward, strike, maturity, rate, cp=1, low=1e-8, high=8.0, tol=1e-8, max_iter=100):
    if not np.isfinite(price) or price <= 0 or forward <= 0 or (strike <= 0) or (maturity <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    intrinsic = discount * max(cp * (forward - strike), 0.0)
    upper = discount * (forward if cp == 1 else strike)
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan
    pricing_function = black76_call if cp == 1 else black76_put
    flo = pricing_function(forward, strike, maturity, rate, low)
    fhi = pricing_function(forward, strike, maturity, rate, high)
    if not np.isfinite(flo) or not np.isfinite(fhi):
        return np.nan
    if abs(price - flo) < tol:
        return low
    if price > fhi:
        return np.nan
    lo, hi = (low, high)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = pricing_function(forward, strike, maturity, rate, mid)
        if not np.isfinite(fmid):
            return np.nan
        if abs(fmid - price) < tol:
            return mid
        if fmid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Expected selected signature of the SAME piecewise-linear OU paths.
def expected_selected_signature_on_ou_grid(kappa, theta, eta, rho_ou, y0, time_grid, selected_indices):
    kappa, theta, eta, y0 = (np.asarray(value, dtype=float) for value in (kappa, theta, eta, y0))
    time_grid = np.asarray(time_grid, dtype=float)
    dim = kappa.size + 1
    mean_ou = y0.copy()
    covariance = np.zeros((dim, dim))
    second_signature = np.zeros((dim, dim))
    for previous_time, next_time in zip(time_grid[:-1], time_grid[1:]):
        dt = float(next_time - previous_time)
        decay = np.exp(-kappa * dt)
        drift_increment = np.concatenate(([0.0], decay - 1.0))
        mean_signature = np.concatenate(([previous_time - time_grid[0]], mean_ou - y0))
        mean_increment = np.concatenate(([dt], (mean_ou - theta) * (decay - 1.0)))
        kappa_sum = kappa[:, None] + kappa[None, :]
        innovation_covariance = np.zeros_like(covariance)
        innovation_covariance[1:, 1:] = rho_ou * np.outer(eta, eta) * -np.expm1(-kappa_sum * dt) / kappa_sum
        increment_covariance = covariance * np.outer(drift_increment, drift_increment) + innovation_covariance
        second_signature += np.outer(mean_signature, mean_increment) + covariance * drift_increment[None, :]
        second_signature += 0.5 * (np.outer(mean_increment, mean_increment) + increment_covariance)
        mean_ou = theta + (mean_ou - theta) * decay
        extended_decay = np.concatenate(([1.0], decay))
        covariance = covariance * np.outer(extended_decay, extended_decay) + innovation_covariance
    mean_signature = np.concatenate(([time_grid[-1] - time_grid[0]], mean_ou - y0))
    expected = []
    for index in selected_indices:
        word = from_num_to_word(int(index), dim - 1)
        if len(word) == 0:
            expected.append(1.0)
        elif len(word) == 1:
            expected.append(mean_signature[word[0]])
        elif len(word) == 2:
            expected.append(second_signature[tuple(word)])
        elif len(word) == 3 and len(set(word)) == 1:
            letter = word[0]
            expected.append((mean_signature[letter] ** 3 + 3.0 * mean_signature[letter] * covariance[letter, letter]) / 6.0)
    return np.asarray(expected, dtype=float)


# Evaluate raw variance at the first N time points from compact signature features.
def variance_path(ell_shuffle, N, variance_signature_features, window):
    ell_shuffle = np.asarray(ell_shuffle, dtype=float)
    signature_path = variance_signature_features["signature_path"]
    pair_positions = variance_signature_features["pair_positions"]
    pair_map = variance_signature_features["pair_to_variance_features"]
    pair_coefficients = pair_map.T @ ell_shuffle
    left, right = (pair_positions[:, 0], pair_positions[:, 1])
    quadratic_matrix = np.zeros((signature_path.shape[-1], signature_path.shape[-1]))
    quadratic_matrix[left, right] = pair_coefficients / np.where(left == right, 1.0, 2.0)
    quadratic_matrix[right, left] = quadratic_matrix[left, right]
    selected_signature = signature_path[:N]
    linear_part = np.einsum("tpi,ij->tpj", selected_signature, quadratic_matrix, optimize=True)
    return window * np.einsum("tpi,tpi->tp", selected_signature, linear_part, optimize=True)


# Evaluate the signed VIX proxy through terminal time index N, in index points.
def vix_path(ell, N, truncated_signature_path):
    ell = np.asarray(ell, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    return 100.0 * np.einsum("tpi,i->tp", truncated_signature_path[: N + 1], ell, optimize=True)

# Compute exp((maturity-t) G) on the requested grid.
def precompute_vix_future_propagators(time_steps, maturity, generator):
    time_steps = np.asarray(time_steps, dtype=float)
    generator = np.asarray(generator, dtype=float)
    return np.stack([expm((maturity - time) * generator) for time in time_steps])

# Precompute the roll AND each contract's complete pre-activation history.
def precompute_vxx_roll_data(time_steps, futures_maturities, generator, previous_expiry=None):
    time_steps = np.asarray(time_steps, dtype=float)
    futures_maturities = np.asarray(futures_maturities, dtype=float)
    generator = np.asarray(generator, dtype=float)
    histories = {}

    def history(maturity):
        if maturity not in histories:
            nodes = time_steps[time_steps <= maturity + 1e-12]
            histories[maturity] = precompute_vix_future_propagators(nodes, maturity, generator)
        return histories[maturity]

    roll_data = []
    horizon = time_steps[-1]
    for k in range(futures_maturities.size - 1):
        previous = float(previous_expiry if k == 0 else futures_maturities[k - 1])
        front, second = map(float, futures_maturities[k : k + 2])
        if k > 0 and horizon <= previous + 1e-12:
            break
        end = min(front, horizon)
        left = 0 if k == 0 else int(np.argmin(np.abs(time_steps - previous)))
        right = int(np.argmin(np.abs(time_steps - end)))
        indices = np.arange(left, right + 1)
        front_history, second_history = (history(front), history(second))
        roll_data.append((indices,previous,front,second,front_history[indices],second_history[indices],front_history,second_history))
        if horizon <= front + 1e-12:
            break
    return tuple(roll_data)


# Evaluate the raw conditional VIX proxy in VIX points, before stopping.
def raw_vix_future_path(ell, propagators, truncated_signature_path):
    ell = np.asarray(ell, dtype=float)
    propagators = np.asarray(propagators, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    propagated_ell = np.einsum("i,tij->tj", ell, propagators, optimize=True)
    return 100.0 * np.einsum("tj,tpj->tp", propagated_ell, truncated_signature_path, optimize=True)


# Convert the decimal future stopping level into VIX points.
def future_floor_in_points(future_floor=1e-8):
    floor_points = 100.0 * float(future_floor)
    return floor_points


# Stop each future at its first grid hit of 100*m and keep it frozen thereafter.
def regularize_vix_futures(raw_future_points, future_floor=1e-8, return_diagnostics=False):
    floor_points = future_floor_in_points(future_floor)
    raw = np.asarray(raw_future_points, dtype=float)
    if raw.ndim == 0:
        return raw.copy()
    crossed = raw <= floor_points
    stopped_mask = np.maximum.accumulate(crossed, axis=0)
    stopped = np.where(stopped_mask, floor_points, raw)
    if not return_diagnostics:
        return stopped
    hit = np.any(crossed, axis=0)
    first = np.where(hit, np.argmax(crossed, axis=0), -1)
    first_values = np.take_along_axis(raw, np.maximum(first, 0)[None, ...], axis=0)[0]
    overshoot = np.where(hit, np.maximum(floor_points - first_values, 0.0), 0.0)
    return (
        stopped,
        {
            "stopped_mask": stopped_mask,
            "first_hit_indices": first,
            "hit_paths": int(np.count_nonzero(hit)),
            "stopped_values": int(np.count_nonzero(stopped_mask)),
            "first_hit_overshoot_points": overshoot,
            "maximum_first_hit_overshoot_points": float(np.max(overshoot))
        }
    )


# Return the absolute linear VIX proxy in VIX points.
def vix_option_proxy(raw_vix_points):
    raw = np.asarray(raw_vix_points, dtype=float)
    return np.abs(raw)


# Evaluate one VXX roll segment from globally stopped futures in VIX points.
def vxx_roll_segment_path(initial_vxx, time_steps, previous_expiry, front_expiry, front_future_path, second_future_path):
    time_steps = np.asarray(time_steps, dtype=float)
    front_future_path = np.asarray(front_future_path, dtype=float)
    second_future_path = np.asarray(second_future_path, dtype=float)
    n_times, n_paths = front_future_path.shape
    invalid_output = np.full((n_times, n_paths), np.nan)
    if not np.all(np.isfinite(front_future_path)) or not np.all(np.isfinite(second_future_path)):
        return invalid_output
    front_weight = front_expiry - time_steps
    second_weight = time_steps - previous_expiry
    roll_basket = front_weight[:, None] * front_future_path + second_weight[:, None] * second_future_path
    if np.any(roll_basket <= 0.0) or not np.all(np.isfinite(roll_basket)):
        return invalid_output
    integrand = (front_future_path - second_future_path) / roll_basket
    integral_increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(time_steps)[:, None]
    cumulative_integral = np.vstack((np.zeros((1, n_paths)), np.cumsum(integral_increments, axis=0)))
    if not np.all(np.isfinite(cumulative_integral)):
        return invalid_output
    log_vxx = np.log(initial_vxx) + np.log(roll_basket) - np.log(roll_basket[0]) + cumulative_integral
    if (not np.all(np.isfinite(log_vxx)) or np.any(log_vxx < np.log(np.finfo(float).tiny)) or np.any(log_vxx > np.log(np.finfo(float).max))):
        return invalid_output
    vxx_path = np.exp(log_vxx)
    vxx_path[0] = initial_vxx
    if not np.all(np.isfinite(vxx_path)) or np.any(vxx_path <= 0.0):
        return invalid_output
    return vxx_path


# Return one global stopped history per contract, shared across roll dates.
def stopped_vxx_future_paths(ell, signature_path, roll_data, future_floor=1e-8):
    signature_path = np.asarray(signature_path, dtype=float)
    paths = {}
    for segment in roll_data:
        for maturity, propagators in ((segment[2], segment[6]), (segment[3], segment[7])):
            if maturity in paths:
                continue
            raw = raw_vix_future_path(ell, propagators, signature_path[: len(propagators)])
            stopped, diagnostics = regularize_vix_futures(raw, future_floor, return_diagnostics=True)
            paths[maturity] = {"raw": raw, "stopped": stopped, "alive": ~diagnostics["stopped_mask"], **diagnostics}
    return paths


# Build the discounted, fee-free VXX path from globally stopped futures.
def vxx_path(initial_vxx, ell, time_steps, truncated_signature_path, roll_data, future_floor=1e-8, return_diagnostics=False):
    floor_points = future_floor_in_points(future_floor)
    times = np.asarray(time_steps, dtype=float)
    signatures = np.asarray(truncated_signature_path, dtype=float)
    ell = np.asarray(ell, dtype=float)
    n_times, n_paths = signatures.shape[:2]
    invalid = np.full((n_times, n_paths), np.nan)
    diagnostics = {
        "future_floor_decimal": float(future_floor),
        "future_floor_points": floor_points,
        "future_stopping_scheme": FUTURE_STOPPING_SCHEME,
        "evaluated_future_values": 0,
        "stopped_future_values": 0,
        "floored_future_values": 0,
        "negative_raw_future_values": 0,
        "raw_future_minimum_points": np.inf,
        "regularized_future_minimum_points": np.inf,
        "maximum_floor_adjustment_points": 0.0,
        "maximum_first_hit_overshoot_points": 0.0,
        "stopped_contract_paths": 0,
        "evaluated_contract_paths": 0,
        "minimum_roll_basket": np.inf,
        "floored_future_fraction": 0.0,
        "initial_future_admissible": True,
        "valid_path": False
    }

    # Return the VXX path with its diagnostics when requested.
    def finish(path, valid):
        diagnostics["valid_path"] = bool(valid)
        if diagnostics["evaluated_future_values"]:
            diagnostics["floored_future_fraction"] = (diagnostics["stopped_future_values"] / diagnostics["evaluated_future_values"])
        return (path, diagnostics) if return_diagnostics else path

    future_paths = stopped_vxx_future_paths(ell, signatures, roll_data, future_floor)
    if any((np.any(item["raw"][0] <= floor_points) for item in future_paths.values())):
        diagnostics["initial_future_admissible"] = False
        return finish(invalid, False)
    for item in future_paths.values() if return_diagnostics else ():
        raw, stopped = (item["raw"], item["stopped"])
        diagnostics["evaluated_future_values"] += raw.size
        diagnostics["stopped_future_values"] += item["stopped_values"]
        diagnostics["floored_future_values"] += item["stopped_values"]
        diagnostics["stopped_contract_paths"] += item["hit_paths"]
        diagnostics["evaluated_contract_paths"] += n_paths
        diagnostics["negative_raw_future_values"] += int(np.count_nonzero(raw < 0.0))
        diagnostics["raw_future_minimum_points"] = min(diagnostics["raw_future_minimum_points"], float(raw.min()))
        diagnostics["regularized_future_minimum_points"] = min(diagnostics["regularized_future_minimum_points"], float(stopped.min()))
        diagnostics["maximum_floor_adjustment_points"] = max(diagnostics["maximum_floor_adjustment_points"], float(np.max(np.abs(stopped - raw))))
        diagnostics["maximum_first_hit_overshoot_points"] = max(diagnostics["maximum_first_hit_overshoot_points"], item["maximum_first_hit_overshoot_points"])
    output = np.empty((n_times, n_paths))
    current = np.full(n_paths, initial_vxx)
    last = None
    for segment in roll_data:
        indices, previous, front, second, _, _, _, _ = segment
        f = future_paths[front]["stopped"][indices]
        g = future_paths[second]["stopped"][indices]
        factor = vxx_roll_segment_path(1.0, times[indices], previous, front, f, g)
        if not np.all(np.isfinite(factor)):
            return finish(invalid, False)
        values = current[None, :] * factor
        output[indices if last is None else indices[1:]] = values if last is None else values[1:]
        if return_diagnostics:
            basket = (front - times[indices, None]) * f + (times[indices, None] - previous) * g
            diagnostics["minimum_roll_basket"] = min(diagnostics["minimum_roll_basket"], float(basket.min()))
        current = values[-1]
        last = int(indices[-1])
    if last != n_times - 1 or not np.all(np.isfinite(output)):
        return finish(invalid, False)
    return finish(output, True)


# Transform raw variance for SPX pricing without altering raw diagnostics.
def variance_for_pricing(raw_variance):
    return np.maximum(np.asarray(raw_variance, dtype=float), 0.0)


# Compute the drift-compensated stochastic exponential using left-point variance.
def stock_ratio_path(variance_left, brownian_increments, dt):
    variance_left = np.asarray(variance_left, dtype=float)
    brownian_increments = np.asarray(brownian_increments, dtype=float)
    n_steps, n_paths = variance_left.shape
    invalid_output = np.full((n_steps + 1, n_paths), np.nan)
    if (n_steps == 0 or np.any(variance_left < 0.0) or (not np.all(np.isfinite(variance_left))) or (not np.all(np.isfinite(brownian_increments)))):
        return invalid_output
    log_increments = np.sqrt(variance_left) * brownian_increments - 0.5 * variance_left * dt
    log_ratio = np.vstack((np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)))
    if not np.all(np.isfinite(log_ratio)) or np.max(log_ratio) > 700.0:
        return invalid_output
    return np.exp(log_ratio)


# Average discounted call/put payoffs; cp is +1 for calls and -1 for puts.
def monte_carlo_option_prices(terminal_values, strikes, maturity, rate, cp):
    terminal_values = np.asarray(terminal_values, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    cp = np.asarray(cp, dtype=int)
    if terminal_values.size == 0 or not np.all(np.isfinite(terminal_values)):
        return np.full(strikes.shape, np.nan)
    payoffs = np.maximum(cp[None, :] * (terminal_values[:, None] - strikes[None, :]), 0.0)
    return np.exp(-rate * maturity) * payoffs.mean(axis=0)


# Compute Black vega times the IV bid-ask spread with a positive floor.
def precompute_loss_normalization(forward, strikes, maturity, rate, bid_iv, ask_iv, minimum=1e-12):
    strikes = np.asarray(strikes, dtype=float)
    bid_iv = np.asarray(bid_iv, dtype=float)
    ask_iv = np.asarray(ask_iv, dtype=float)
    discount = np.exp(-rate * maturity)
    mid_iv = 0.5 * (bid_iv + ask_iv)
    vegas = np.array([black_vega(discount, forward, strike, maturity, vol) for strike, vol in zip(strikes, mid_iv)])
    normalization = np.maximum(vegas * (ask_iv - bid_iv), minimum)
    return normalization


# Sum squared normalized midpoint/band residuals, optionally including future errors.
def calibration_block_loss(model_prices, bid_prices, ask_prices, normalization, beta=0.98, model_future=None, market_future=None, deltas=None, weights=None):
    model_prices = np.asarray(model_prices, dtype=float)
    bid_prices = np.asarray(bid_prices, dtype=float)
    ask_prices = np.asarray(ask_prices, dtype=float)
    normalization = np.asarray(normalization, dtype=float)
    if not np.all(np.isfinite(model_prices)):
        return 1e12
    use_future = model_future is not None or market_future is not None or deltas is not None
    if use_future:
        deltas = np.asarray(deltas, dtype=float)
        if not np.isfinite(model_future) or not np.isfinite(market_future):
            return 1e12
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
    mid_prices = 0.5 * (bid_prices + ask_prices)
    mid_error = np.abs(model_prices - mid_prices)
    outside_band = np.maximum(bid_prices - model_prices, 0.0) + np.maximum(model_prices - ask_prices, 0.0)
    numerator = (1.0 - beta) * mid_error + beta * outside_band
    if use_future:
        numerator = numerator + np.abs(deltas * (model_future - market_future))
    scaled_residual = np.clip(numerator / normalization, 0.0, 1e150)
    residual = scaled_residual**2
    if weights is not None:
        residual = weights * residual
    loss = np.sum(residual)
    return float(loss) if np.isfinite(loss) else 1e12


# Differentiate one product loss with respect to option prices and the future.
def calibration_block_loss_gradient(model_prices, bid_prices, ask_prices, normalization, beta, model_future=None, market_future=None, deltas=None):
    mid_prices = 0.5 * (bid_prices + ask_prices)
    outside_band = np.maximum(bid_prices - model_prices, 0.0) + np.maximum(model_prices - ask_prices, 0.0)
    numerator = (1.0 - beta) * np.abs(model_prices - mid_prices) + beta * outside_band
    if model_future is not None:
        numerator = numerator + np.abs(deltas * (model_future - market_future))
    price_slope = (1.0 - beta) * np.sign(model_prices - mid_prices) + beta * ((model_prices > ask_prices).astype(float) - (model_prices < bid_prices).astype(float))
    residual_slope = 2.0 * numerator / normalization**2
    price_gradient = residual_slope * price_slope
    future_gradient = (float(np.sum(residual_slope * np.abs(deltas)) * np.sign(model_future - market_future)) if model_future is not None else 0.0)
    return (price_gradient, future_gradient)


# Differentiate the joint loss on the same simulated paths used for pricing.
def joint_calibration_gradient_with_saturated_correlations(calibration_parameters, dimension_ell, rho_ou, factor_brownian_increments, independent_brownian_increments, leverage_cap, objective_inputs, calibration_details, loss_factors):
    inputs = objective_inputs
    ell = np.asarray(calibration_parameters[:dimension_ell], dtype=float)
    correlation_parameters = np.asarray(calibration_parameters[dimension_ell:], dtype=float)
    rho_b = rho_b_from_parameters(correlation_parameters, rho_ou)
    weights = np.linalg.solve(rho_ou, rho_b)
    q_rho = float(rho_b @ weights)
    signature_path = inputs["truncated_signature_path"]
    n_paths = signature_path.shape[1]
    gradient = np.zeros_like(calibration_parameters, dtype=float)
    price_gradients = {}
    future_gradient = 0.0
    for product in ("SPX", "VIX", "VXX"):
        key = product.lower()
        future_inputs = ({"model_future": calibration_details["model_future_vix"], "market_future": inputs["market_future_vix"], "deltas": inputs["deltas_vix"]} if product == "VIX" else {})
        price_gradient, current_future_gradient = calibration_block_loss_gradient(
            calibration_details["model_prices"][product], inputs["bid_" + key], inputs["ask_" + key], inputs["normalization_" + key], inputs["beta_" + key],
            **future_inputs
        )
        price_gradients[product] = loss_factors[product] * price_gradient
        future_gradient += loss_factors[product] * current_future_gradient

    # Differentiate the product loss with respect to each terminal underlying value.
    def terminal_loss_sensitivity(terminal, product):
        key = product.lower()
        cp = inputs["cp_" + key]
        payoff_slopes = cp[None, :] * (cp[None, :] * (terminal[:, None] - inputs["strikes_" + key][None, :]) > 0.0)
        discount = np.exp(-inputs["rate"] * inputs["maturity_" + key])
        result = discount * (payoff_slopes @ price_gradients[product]) / n_paths
        if product == "SPX" and inputs.get("use_price_control_variates", False):
            result = result - price_gradients[product] @ inputs["deltas_spx"] / n_paths
        return result

    pair_positions = inputs["variance_pair_positions"]
    shuffle_coordinates = inputs["variance_shuffle_coordinates"]
    ell_shuffle = reduced_shuffle_product_ell(ell, pair_positions, shuffle_coordinates)
    n_spx = inputs["N_spx"]
    raw_variance = variance_path(ell_shuffle, n_spx, inputs["variance_signature_features"], inputs["window"])
    variance = np.maximum(raw_variance, 0.0)
    sigma = np.sqrt(variance)
    a = np.minimum(sigma, leverage_cap)
    b = np.sqrt(np.maximum(variance - q_rho * a**2, 0.0))
    factor_drivers = factor_brownian_increments[:n_spx]
    independent_drivers = independent_brownian_increments[:n_spx]
    correlated_driver = np.einsum("tpi,i->tp", factor_drivers, weights, optimize=True)
    brownian = brownian_increments_with_saturation(rho_b, rho_ou, variance, factor_drivers, independent_drivers, leverage_cap)
    terminal_spx = inputs["forward_spx"] * stock_ratio_path(variance, brownian, inputs["dt"])[-1]
    stock_log_sensitivity = terminal_loss_sensitivity(terminal_spx, "SPX") * terminal_spx
    da_dv = np.zeros_like(variance)
    np.divide(0.5, sigma, out=da_dv, where=(sigma > 0.0) & (sigma < leverage_cap))
    db_dv = np.zeros_like(variance)
    np.divide(1.0 - q_rho * (sigma < leverage_cap), 2.0 * b, out=db_dv, where=b > 0.0)
    step_sensitivity = (da_dv * correlated_driver + db_dv * independent_drivers - 0.5 * inputs["dt"]) * (raw_variance > 0.0)
    weighted_step_sensitivity = step_sensitivity * stock_log_sensitivity[None, :]
    variance_features = inputs["variance_signature_features"]
    selected_signature = variance_features["signature_path"][:n_spx]
    flat_signature = selected_signature.reshape(-1, dimension_ell)
    weighted_signature = flat_signature * weighted_step_sensitivity.reshape(-1, 1)
    signature_products = flat_signature.T @ weighted_signature
    left, right = (pair_positions[:, 0], pair_positions[:, 1])
    pair_moments = signature_products[left, right]
    feature_sensitivity = inputs["window"] * (variance_features["pair_to_variance_features"] @ pair_moments)
    pair_sensitivity = shuffle_coordinates @ feature_sensitivity
    left, right = (pair_positions[:, 0], pair_positions[:, 1])
    np.add.at(gradient, left, 2.0 * pair_sensitivity * ell[right])
    off_diagonal = left != right
    np.add.at(gradient, right[off_diagonal], 2.0 * pair_sensitivity[off_diagonal] * ell[left[off_diagonal]])
    radius_squared = 1.0 + correlation_parameters @ correlation_parameters
    correlation_jacobian = np.linalg.cholesky(rho_ou) @ (np.eye(correlation_parameters.size) / np.sqrt(radius_squared) - np.outer(correlation_parameters, correlation_parameters) / radius_squared**1.5)
    weights_jacobian = np.linalg.solve(rho_ou, correlation_jacobian)
    q_gradient = 2.0 * weights @ correlation_jacobian
    factor_sensitivity = np.einsum("p,tp,tpi->i", stock_log_sensitivity, a, factor_drivers, optimize=True)
    independent_coefficient = np.zeros_like(b)
    np.divide(a**2, 2.0 * b, out=independent_coefficient, where=b > 0.0)
    independent_sensitivity = np.einsum("p,tp,tp->", stock_log_sensitivity, independent_coefficient, independent_drivers, optimize=True)
    gradient[dimension_ell:] += factor_sensitivity @ weights_jacobian - independent_sensitivity * q_gradient
    floor_points = 100.0 * inputs["future_floor"]
    terminal_signature_vix = signature_path[inputs["N_vix"]]
    raw_vix = 100.0 * (terminal_signature_vix @ ell)
    terminal_vix = vix_option_proxy(raw_vix)
    terminal_sensitivity = terminal_loss_sensitivity(terminal_vix, "VIX") * np.sign(raw_vix)
    vix_control_sensitivity = (float(price_gradients["VIX"] @ inputs["deltas_vix"]) if inputs.get("use_price_control_variates", False) else 0.0)
    terminal_sensitivity = terminal_sensitivity - vix_control_sensitivity / n_paths
    gradient[:dimension_ell] += 100.0 * terminal_sensitivity @ terminal_signature_vix
    future_loadings = inputs["vix_future_propagator"] @ signature_path[0, 0]
    if inputs.get("use_price_control_variates", False):
        gradient[:dimension_ell] += 100.0 * vix_control_sensitivity * inputs["vix_control_loadings"]
    if calibration_details["raw_model_future_vix"] > floor_points:
        gradient[:dimension_ell] += 100.0 * future_gradient * future_loadings
    terminal_vxx = np.full(n_paths, inputs["vxx_spot"], dtype=float)
    segment_sensitivities = []
    global_futures = stopped_vxx_future_paths(ell, signature_path, inputs["vxx_roll_data"], inputs["future_floor"])
    for segment in inputs["vxx_roll_data"]:
        indices, previous_expiry, front_expiry, second_expiry, front_propagators, second_propagators = (segment[:6])
        segment_times = inputs["time_steps_vxx"][indices]
        segment_signature = signature_path[indices]
        front = global_futures[front_expiry]["stopped"][indices]
        second = global_futures[second_expiry]["stopped"][indices]
        front_weight = (front_expiry - segment_times)[:, None]
        second_weight = (segment_times - previous_expiry)[:, None]
        basket = front_weight * front + second_weight * second
        dt_segment = np.diff(segment_times)
        quadrature_weights = np.zeros(segment_times.size)
        quadrature_weights[:-1] += 0.5 * dt_segment
        quadrature_weights[1:] += 0.5 * dt_segment
        integrand = (front - second) / basket
        integral = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * dt_segment[:, None], axis=0)
        terminal_vxx *= np.exp(np.log(basket[-1]) - np.log(basket[0]) + integral)
        tau = front_expiry - previous_expiry
        front_sensitivity = quadrature_weights[:, None] * tau * second / basket**2
        second_sensitivity = -quadrature_weights[:, None] * tau * front / basket**2
        front_sensitivity[-1] += front_weight[-1] / basket[-1]
        front_sensitivity[0] -= front_weight[0] / basket[0]
        second_sensitivity[-1] += second_weight[-1] / basket[-1]
        second_sensitivity[0] -= second_weight[0] / basket[0]
        front_sensitivity *= global_futures[front_expiry]["alive"][indices]
        second_sensitivity *= global_futures[second_expiry]["alive"][indices]
        segment_sensitivities.append((segment_signature, front_propagators, second_propagators, front_sensitivity, second_sensitivity))
    terminal_vxx *= np.exp((inputs["rate"] - inputs["vxx_annual_fee"]) * inputs["maturity_vxx"])
    vxx_log_sensitivity = terminal_loss_sensitivity(terminal_vxx, "VXX") * terminal_vxx
    for (segment_signature, front_propagators, second_propagators, front_sensitivity, second_sensitivity) in segment_sensitivities:
        for propagators, sensitivity in ((front_propagators, front_sensitivity), (second_propagators, second_sensitivity)):
            weighted_sensitivity = sensitivity * vxx_log_sensitivity[None, :]
            signature_sensitivity = np.einsum("tp,tpj->tj", weighted_sensitivity, segment_signature, optimize=True)
            gradient[:dimension_ell] += 100.0 * np.einsum("tij,tj->i", propagators, signature_sensitivity, optimize=True)
    return gradient


# Return the sampled minimum variance margin and its active-point gradient.
def variance_positivity_constraint(ell, N, variance_signature_features, window, minimum_variance=0.0, variance_pair_positions=None, variance_shuffle_coordinates=None, return_jacobian=False):
    ell = np.asarray(ell, dtype=float)
    if not np.all(np.isfinite(ell)):
        invalid_value = -1e12
        return (invalid_value, np.zeros_like(ell)) if return_jacobian else invalid_value
    with np.errstate(over="ignore", invalid="ignore"):
        ell_shuffle = reduced_shuffle_product_ell(ell, variance_pair_positions, variance_shuffle_coordinates)
        raw_variance = variance_path(ell_shuffle, N, variance_signature_features, window)
    if not np.all(np.isfinite(raw_variance)):
        invalid_value = -1e12
        return (invalid_value, np.zeros_like(ell)) if return_jacobian else invalid_value
    constraint_value = float(np.min(raw_variance) - minimum_variance)
    if not return_jacobian:
        return constraint_value
    active_index = np.unravel_index(int(np.argmin(raw_variance)), raw_variance.shape)
    signature_value = variance_signature_features["signature_path"][active_index]
    feature_pairs = variance_signature_features["pair_positions"]
    active_features = variance_signature_features["pair_to_variance_features"] @ (signature_value[feature_pairs[:, 0]] * signature_value[feature_pairs[:, 1]])
    variance_pair_positions = np.asarray(variance_pair_positions, dtype=int)
    variance_shuffle_coordinates = np.asarray(variance_shuffle_coordinates, dtype=float)
    pair_left = variance_pair_positions[:, 0]
    pair_right = variance_pair_positions[:, 1]
    pair_coefficients = window * (variance_shuffle_coordinates @ active_features)
    gradient = np.zeros_like(ell)
    np.add.at(gradient, pair_left, 2.0 * pair_coefficients * ell[pair_right])
    off_diagonal = pair_left != pair_right
    np.add.at(gradient, pair_right[off_diagonal], 2.0 * pair_coefficients[off_diagonal] * ell[pair_left[off_diagonal]])
    return (constraint_value, gradient)


# Price SPX, VIX and VXX and evaluate their weighted calibration losses.
def joint_calibration_objective(
    ell, variance_signature_features, truncated_signature_path, window, dt, N_spx, N_vix, time_steps_vxx, vix_future_propagator, vxx_roll_data, forward_spx, vxx_spot,
    strikes_spx, strikes_vix, strikes_vxx, maturity_spx, maturity_vix, maturity_vxx, rate, bid_spx, ask_spx, bid_vix, ask_vix, bid_vxx, ask_vxx, normalization_spx,
    normalization_vix, normalization_vxx, deltas_vix, market_future_vix, beta_spx=0.98, beta_vix=0.98, beta_vxx=0.98, lambda_spx=0.25, lambda_vix=0.5,
    lambda_vxx=0.25, return_components=False, cp_spx=None, cp_vix=None, cp_vxx=None, variance_pair_positions=None, variance_shuffle_coordinates=None,
    future_floor=1e-8, collect_future_floor_diagnostics=False, factor_brownian_increments=None, independent_brownian_increments=None, rho_b=None, rho_ou=None,
    leverage_cap=100.0, vxx_annual_fee=0.0089, use_price_control_variates=False, deltas_spx=None, vix_control_loadings=None
):
    ell = np.asarray(ell, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    time_steps_vxx = np.asarray(time_steps_vxx, dtype=float)
    vix_future_propagator = np.asarray(vix_future_propagator, dtype=float)
    cp_spx = (np.ones(np.asarray(strikes_spx).shape, dtype=int) if cp_spx is None else np.asarray(cp_spx, dtype=int))
    cp_vix = (np.ones(np.asarray(strikes_vix).shape, dtype=int) if cp_vix is None else np.asarray(cp_vix, dtype=int))
    cp_vxx = (np.ones(np.asarray(strikes_vxx).shape, dtype=int) if cp_vxx is None else np.asarray(cp_vxx, dtype=int))
    if not np.all(np.isfinite(ell)):
        invalid_losses = {"SPX": 1e12, "VIX": 1e12, "VXX": 1e12}
        return (1e12, invalid_losses) if return_components else 1e12
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ell_shuffle = reduced_shuffle_product_ell(ell, variance_pair_positions, variance_shuffle_coordinates)
        variance_spx = variance_path(ell_shuffle, N_spx, variance_signature_features, window)
        effective_variance_spx = variance_for_pricing(variance_spx)
        brownian_increments = brownian_increments_with_saturation(rho_b, rho_ou, effective_variance_spx, factor_brownian_increments[:N_spx], independent_brownian_increments[:N_spx], leverage_cap)
        stock_ratio_spx = stock_ratio_path(effective_variance_spx, brownian_increments[:N_spx], dt)
        terminal_spx = forward_spx * stock_ratio_spx[-1]
        model_prices_spx = monte_carlo_option_prices(terminal_spx, strikes_spx, maturity_spx, rate, cp_spx)
        raw_terminal_vix = vix_path(ell, N_vix, truncated_signature_path)[-1]
        terminal_vix = vix_option_proxy(raw_terminal_vix)
        model_prices_vix = monte_carlo_option_prices(terminal_vix, strikes_vix, maturity_vix, rate, cp_vix)
        raw_model_future_vix = 100.0 * ell @ vix_future_propagator @ truncated_signature_path[0, 0]
        model_future_vix = float(raw_model_future_vix)
        if use_price_control_variates:
            model_prices_spx = model_prices_spx - deltas_spx * (np.mean(terminal_spx) - forward_spx)
            control_mean_vix = 100.0 * ell @ vix_control_loadings
            model_prices_vix = model_prices_vix - deltas_vix * (np.mean(raw_terminal_vix) - control_mean_vix)
        vxx_result = vxx_path(vxx_spot, ell, time_steps_vxx, truncated_signature_path[: time_steps_vxx.size], vxx_roll_data, future_floor=future_floor, return_diagnostics=collect_future_floor_diagnostics)
        if collect_future_floor_diagnostics:
            discounted_vxx_path, future_floor_diagnostics = vxx_result
        else:
            discounted_vxx_path = vxx_result
        terminal_vxx = np.exp((rate - vxx_annual_fee) * maturity_vxx) * discounted_vxx_path[-1]
        model_prices_vxx = monte_carlo_option_prices(terminal_vxx, strikes_vxx, maturity_vxx, rate, cp_vxx)
        loss_spx = calibration_block_loss(model_prices_spx, bid_spx, ask_spx, normalization_spx, beta_spx)
        loss_vix = calibration_block_loss(model_prices_vix, bid_vix, ask_vix, normalization_vix, beta_vix, model_future_vix, market_future_vix, deltas_vix)
        loss_vxx = calibration_block_loss(model_prices_vxx, bid_vxx, ask_vxx, normalization_vxx, beta_vxx)
        joint_loss = lambda_spx * loss_spx + lambda_vix * loss_vix + lambda_vxx * loss_vxx
    if raw_model_future_vix <= future_floor_in_points(future_floor):
        joint_loss = 1e12
    joint_loss = float(joint_loss) if np.isfinite(joint_loss) else 1e12
    calibration_details = {
        "SPX": float(loss_spx),
        "VIX": float(loss_vix),
        "VXX": float(loss_vxx),
        "model_prices": {"SPX": model_prices_spx, "VIX": model_prices_vix, "VXX": model_prices_vxx},
        "model_future_vix": float(model_future_vix),
        "raw_model_future_vix": float(raw_model_future_vix),
        "future_floor_decimal": float(future_floor),
        "terminal_vix_negative_values": int(np.count_nonzero(raw_terminal_vix < 0.0)),
        "vix_proxy": VIX_PROXY,
        "vix_future_proxy": VIX_FUTURE_PROXY,
        "raw_terminal_vix_minimum_points": float(np.min(raw_terminal_vix))
    }
    if collect_future_floor_diagnostics:
        calibration_details["future_floor_diagnostics"] = future_floor_diagnostics
    if return_components:
        calibration_details["leverage_saturated_fraction"] = float(np.mean(np.sqrt(effective_variance_spx) > leverage_cap))
        calibration_details["q_rho"] = float(rho_b @ np.linalg.solve(rho_ou, rho_b))
        return (joint_loss, calibration_details)
    return joint_loss


# Evaluate variance constraints and gradients over all paths, optionally in time blocks.
def variance_positivity_constraint_with_correlations(calibration_parameters, dimension_ell, constraint_inputs, return_jacobian=False, number_of_time_blocks=1):
    calibration_parameters = np.asarray(calibration_parameters, dtype=float)
    if number_of_time_blocks == 1:
        result = variance_positivity_constraint(calibration_parameters[:dimension_ell], return_jacobian=return_jacobian, **constraint_inputs)
    else:
        number_of_times = int(constraint_inputs["N"])
        block_edges = np.linspace(0, number_of_times, min(number_of_time_blocks, number_of_times) + 1, dtype=int)
        block_results = []
        for block_start, block_end in zip(block_edges[:-1], block_edges[1:]):
            features = constraint_inputs["variance_signature_features"]
            block_features = {**features, "signature_path": features["signature_path"][block_start:block_end]}
            block_inputs = {**constraint_inputs, "N": int(block_end - block_start), "variance_signature_features": block_features}
            block_results.append(variance_positivity_constraint(calibration_parameters[:dimension_ell], return_jacobian=return_jacobian, **block_inputs))
        result = ((np.array([value for value, _ in block_results]), np.stack([gradient for _, gradient in block_results])) if return_jacobian else np.asarray(block_results, dtype=float))
    if not return_jacobian:
        return result
    constraint_value, ell_gradient = result
    full_gradient = np.zeros(ell_gradient.shape[:-1] + calibration_parameters.shape, dtype=float)
    full_gradient[..., :dimension_ell] = ell_gradient
    return (constraint_value, full_gradient)


# Build the quadratic variance matrix for fixed signature coefficients.
def fixed_ell_variance_matrix(ell, reduced_model, window):
    ell = np.asarray(ell, dtype=float)
    pairs = np.asarray(reduced_model["pair_positions"], dtype=int)
    ell_shuffle = reduced_shuffle_product_ell(ell, pairs, reduced_model["shuffle_product_coordinates"])
    coefficients = window * (np.asarray(reduced_model["pair_to_variance_features"]).T @ ell_shuffle)
    matrix = np.zeros((ell.size, ell.size), dtype=float)
    for (i, j), coefficient in zip(pairs, coefficients):
        if i == j:
            matrix[i, j] = coefficient
        else:
            matrix[i, j] = matrix[j, i] = 0.5 * coefficient
    return matrix


# Update level-two signatures and the slow-OU and time cubics by Chen's identity.
def advance_selected_signature(state, increment):
    state = np.asarray(state, dtype=float)
    increment = np.asarray(increment, dtype=float)
    updated = state.copy()
    updated[..., 5:21] += (state[..., 1:5, None] * increment[..., None, :] + 0.5 * increment[..., :, None] * increment[..., None, :]).reshape(state.shape[:-1] + (16,))
    updated[..., 1:5] += increment
    updated[..., 21] = updated[..., 2] ** 3 / 6.0
    updated[..., 22] = updated[..., 1] ** 3 / 6.0
    return updated


# Evaluate raw variance from selected signatures and a fixed quadratic matrix.
def raw_variance_from_selected_signature(state, variance_matrix):
    state = np.asarray(state, dtype=float)
    return np.sum(state @ variance_matrix * state, axis=-1)


# Plot the three out-of-sample implied-volatility smiles against market bands.
def plot_out_of_sample_smiles(
    model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix, strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx, bid_iv_spx, ask_iv_spx,
    bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx, *, model_label="calibrated iv", show=True
):
    import matplotlib.pyplot as plt

    model_iv_spx = np.asarray(model_iv_spx, dtype=float)
    model_iv_vix = np.asarray(model_iv_vix, dtype=float)
    model_iv_vxx = np.asarray(model_iv_vxx, dtype=float)
    strikes_spx = np.asarray(strikes_spx, dtype=float)
    strikes_vix = np.asarray(strikes_vix, dtype=float)
    strikes_vxx = np.asarray(strikes_vxx, dtype=float)
    bid_iv_spx = np.asarray(bid_iv_spx, dtype=float)
    ask_iv_spx = np.asarray(ask_iv_spx, dtype=float)
    bid_iv_vix = np.asarray(bid_iv_vix, dtype=float)
    ask_iv_vix = np.asarray(ask_iv_vix, dtype=float)
    bid_iv_vxx = np.asarray(bid_iv_vxx, dtype=float)
    ask_iv_vxx = np.asarray(ask_iv_vxx, dtype=float)
    moneyness_spx = strikes_spx / forward_spx
    moneyness_vxx = strikes_vxx / forward_vxx
    model_color = "#315efb"
    market_color = "#e45756"
    band_color = "#f3a6a5"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120, constrained_layout=True)

    # Draw one implied-volatility smile and its bid-ask band.
    def draw_smile(axis, x_values, model_iv, bid_iv, ask_iv, xlabel):
        axis.fill_between(x_values, bid_iv, ask_iv, color=band_color, alpha=0.3, label="Call bid-ask band")
        axis.plot(x_values, bid_iv, "+", color=market_color, markersize=6)
        axis.plot(x_values, ask_iv, "+", color=market_color, markersize=6)
        axis.plot(x_values, model_iv, "o", color=model_color, linestyle="None", markersize=4.5, label=model_label)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Implied volatility")
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False)

    draw_smile(axes[0], moneyness_spx, model_iv_spx, bid_iv_spx, ask_iv_spx, "Moneyness K/F")
    draw_smile(axes[1], strikes_vix, model_iv_vix, bid_iv_vix, ask_iv_vix, "Strike")
    draw_smile(axes[2], moneyness_vxx, model_iv_vxx, bid_iv_vxx, ask_iv_vxx, "Moneyness K/F")
    axes[1].axvline(forward_vix, color=market_color, linestyle="-.", linewidth=1.4, label="Market future")
    axes[1].axvline(model_future_vix, color=model_color, linestyle="--", linewidth=1.4, label="Model future")
    axes[1].legend(frameon=False)
    if show:
        plt.show()
    return (fig, axes)


# Return the margin between each initial future and its required minimum.
def initial_future_constraint(parameters, *, dimension_ell, initial_future_loadings, minimum_initial_future):
    return initial_future_loadings @ np.asarray(parameters)[:dimension_ell] - minimum_initial_future


# Normalize product losses and return their contributions to the joint loss.
def normalized_joint_loss(block_losses, *, lambda_spx, lambda_vix, lambda_vxx, loss_scales, objective_scale):
    contributions = {
        "SPX": objective_scale * lambda_spx * float(block_losses["SPX"]) / loss_scales["SPX"],
        "VIX": objective_scale * lambda_vix * float(block_losses["VIX"]) / loss_scales["VIX"],
        "VXX": objective_scale * lambda_vxx * float(block_losses["VXX"]) / loss_scales["VXX"]
    }
    joint_loss = float(sum(contributions.values()))
    return (joint_loss, contributions)


# Run the three SLSQP passes and retain the best feasible accepted parameters.
def calibrate(
    *, brownian_inputs, calibration_parameters0, constraint_scale, correlation_restart_shrink, dimension_ell, feasibility_tolerance, free_parameter_indices,
    initial_block_losses, initial_future_constraint, initial_future_constraint_jacobian, initial_loss, lambda_spx, lambda_vix, lambda_vxx, loss_scales,
    normalized_joint_loss, objective_scale, objective_with_components, optimizer_run_names, optimizer_runs, parameter_directions, parameter_scales, rho_ou,
    use_analytic_objective_gradient, variance_constraint_with_jacobian
):
    constraint_cache = {"parameters": None, "value": None, "jacobian": None}

    # Reuse the variance constraint and Jacobian at the current parameters.
    def cached_variance_constraint(current_parameters):
        if constraint_cache["parameters"] is None or not np.array_equal(current_parameters, constraint_cache["parameters"]):
            current_value, current_jacobian = variance_constraint_with_jacobian(current_parameters)
            constraint_cache["parameters"] = np.asarray(current_parameters, dtype=float).copy()
            constraint_cache["value"] = current_value
            constraint_cache["jacobian"] = current_jacobian
        return (constraint_cache["value"], constraint_cache["jacobian"])

    # Convert scaled optimization coordinates into model parameters.
    def parameters_from_scaled(scaled_parameters):
        return calibration_parameters0 + parameter_directions @ scaled_parameters

    # Evaluate variance constraints in scaled optimization coordinates.
    def scaled_variance_constraint(scaled_parameters):
        current_parameters = parameters_from_scaled(scaled_parameters)
        current_value, _ = cached_variance_constraint(current_parameters)
        return current_value / constraint_scale

    # Map the variance-constraint Jacobian to scaled optimization coordinates.
    def scaled_variance_constraint_jacobian(scaled_parameters):
        current_parameters = parameters_from_scaled(scaled_parameters)
        _, current_jacobian = cached_variance_constraint(current_parameters)
        return current_jacobian @ parameter_directions / constraint_scale

    # Evaluate initial-future constraints at the scaled parameters.
    def scaled_initial_future_constraint(scaled_parameters):
        return initial_future_constraint(parameters_from_scaled(scaled_parameters))

    # Map the initial-future Jacobian to scaled optimization coordinates.
    def scaled_initial_future_constraint_jacobian(scaled_parameters):
        return initial_future_constraint_jacobian @ parameter_directions

    slsqp_constraints = (
        {"type": "ineq", "fun": scaled_variance_constraint, "jac": scaled_variance_constraint_jacobian},
        {"type": "ineq", "fun": scaled_initial_future_constraint, "jac": scaled_initial_future_constraint_jacobian}
    )
    optimization_progress = {
        "evaluations": 0,
        "best_seen_loss": initial_loss,
        "best_loss": initial_loss,
        "best_parameters": calibration_parameters0.copy(),
        "parameters": calibration_parameters0.copy(),
        "loss": initial_loss,
        "block_losses": initial_block_losses
    }

    # Evaluate the normalized loss, cache its details and print new minima.
    def monitored_objective(current_parameters):
        if np.array_equal(current_parameters, optimization_progress["parameters"]):
            return optimization_progress["loss"]
        current_raw_loss, current_block_losses = objective_with_components(current_parameters)
        optimization_progress["evaluations"] += 1
        if (not np.isfinite(current_raw_loss) or current_raw_loss >= 1e12 or any((not np.isfinite(current_block_losses[product]) or current_block_losses[product] >= 1e12 for product in ("SPX", "VIX", "VXX")))):
            return 1e12
        current_loss, current_contributions = normalized_joint_loss(current_block_losses)
        optimization_progress["parameters"] = np.asarray(current_parameters, dtype=float).copy()
        optimization_progress["loss"] = current_loss
        optimization_progress["block_losses"] = current_block_losses
        if current_loss < optimization_progress["best_seen_loss"]:
            optimization_progress["best_seen_loss"] = current_loss
            print(
                f"Evaluation {optimization_progress['evaluations']:5d} | joint = {current_loss:.8e} | SPX = {current_contributions['SPX']:.8e} | VIX = {current_contributions['VIX']:.8e} | VXX = {current_contributions['VXX']:.8e} | rho_b = {np.array2string(current_block_losses['rho_b'], precision=4)}",
                flush=True
            )
        return current_loss

    # Evaluate the monitored loss in scaled optimization coordinates.
    def scaled_monitored_objective(scaled_parameters):
        current_parameters = parameters_from_scaled(scaled_parameters)
        return monitored_objective(current_parameters) / max(initial_loss, 1e-12)

    # Map the joint-loss gradient to scaled optimization coordinates.
    def scaled_monitored_objective_jacobian(scaled_parameters):
        current_parameters = parameters_from_scaled(scaled_parameters)
        current_loss = monitored_objective(current_parameters)
        if current_loss >= 1e12:
            return np.zeros_like(scaled_parameters)
        loss_factors = {"SPX": objective_scale * lambda_spx / loss_scales["SPX"], "VIX": objective_scale * lambda_vix / loss_scales["VIX"], "VXX": objective_scale * lambda_vxx / loss_scales["VXX"]}
        current_gradient = joint_calibration_gradient_with_saturated_correlations(current_parameters, calibration_details=optimization_progress["block_losses"], loss_factors=loss_factors, **brownian_inputs)
        return current_gradient @ parameter_directions / max(initial_loss, 1e-12)

    # Retain an accepted iterate when it improves the best feasible loss.
    def monitor_accepted_parameters(scaled_parameters):
        current_parameters = parameters_from_scaled(scaled_parameters)
        current_loss = monitored_objective(current_parameters)
        if current_loss < optimization_progress["best_loss"]:
            current_variance_constraint, _ = cached_variance_constraint(current_parameters)
            if (np.min(current_variance_constraint) < -feasibility_tolerance or np.min(initial_future_constraint(current_parameters)) < -1e-12):
                return
            optimization_progress["best_loss"] = current_loss
            optimization_progress["best_parameters"] = current_parameters.copy()

    for optimizer_run, current_optimizer_options in enumerate(optimizer_runs, start=1):
        starting_parameters = optimization_progress["best_parameters"].copy()
        if optimizer_run == 2:
            starting_rho_b = rho_b_from_parameters(starting_parameters[dimension_ell:], rho_ou)
            starting_parameters[dimension_ell:] = parameters_from_rho_b(correlation_restart_shrink * starting_rho_b, rho_ou)
        scaled_parameters0 = (starting_parameters[free_parameter_indices] - calibration_parameters0[free_parameter_indices]) / parameter_scales[free_parameter_indices]
        starting_loss = monitored_objective(starting_parameters)
        optimization_progress["best_seen_loss"] = starting_loss
        gradient_description = ("analytic gradient" if use_analytic_objective_gradient else f"finite-difference step = {current_optimizer_options['eps']:.1e}")
        print(f"SLSQP run {optimizer_run}/{len(optimizer_runs)} | {optimizer_run_names[optimizer_run - 1]} | starting joint loss = {starting_loss:.8e} | {gradient_description}", flush=True)
        monitor_accepted_parameters(scaled_parameters0)
        calibration_result = minimize(
            scaled_monitored_objective, scaled_parameters0, method="SLSQP", jac=scaled_monitored_objective_jacobian if use_analytic_objective_gradient else None,
            bounds=None, constraints=slsqp_constraints, callback=monitor_accepted_parameters, options=current_optimizer_options
        )
        monitor_accepted_parameters(calibration_result.x)
        print(f"End of run {optimizer_run} | best feasible joint loss = {optimization_progress['best_loss']:.8e} | {calibration_result.message}", flush=True)
    return (calibration_result, optimization_progress)


# Convert call prices to Black implied volatilities, including intrinsic values.
def invert_prices(prices, forward, strikes, maturity, *, r):
    intrinsic = np.exp(-r * maturity) * np.maximum(forward - strikes, 0)
    return np.array([(0.0 if abs(price - lower) <= 1e-8 else implied_vol_black76(price, forward, strike, maturity, r, 1)) for price, lower, strike in zip(prices, intrinsic, strikes)])
