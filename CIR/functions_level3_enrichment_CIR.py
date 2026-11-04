import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.special import ndtr

try:
    import iisignature
except ImportError:
    iisignature = None

try:
    from tqdm import tqdm
except ImportError:
    # Return the iterator when progress bars are unavailable.
    def tqdm(iterable, **kwargs):
        return iterable


PAPER_PRICING_VERSION = "cir_fit_abs_vix_stopped_futures_v1"
VIX_PROXY = "absolute_linear_signature"
VIX_FUTURE_PROXY = "stopped_signed_conditional_signature"
FUTURE_STOPPING_SCHEME = "first_grid_hit_freeze_at_m"


# Encode a signature word in level-first lexicographic order.
def from_word_to_num(word, d_minus_one):
    if len(word) == 0:
        return 0
    word = np.asarray(word, dtype=int)
    powers = np.flip(np.array([(d_minus_one + 1) ** i for i in range(len(word))], dtype=float))
    all0but1 = np.ones(len(word))
    all0but1[-1] = 0
    return int(np.dot(word + all0but1, powers)) + 1

# Decode a signature coordinate into its word.
def from_num_to_word(n, d_minus_one):
    if n == 0:
        return []
    return from_num_to_word((n - 1) // (d_minus_one + 1), d_minus_one) + [(n - 1) % (d_minus_one + 1)]

# Return the final coordinate index at the requested level.
def from_level_to_num(level, d_minus_one):
    return from_word_to_num(d_minus_one * np.ones(level, dtype=int), d_minus_one)

# Enumerate shuffle words while retaining their multiplicities.
def shuffle(a, b):
    if len(a) == 0:
        return [b]
    if len(b) == 0:
        return [a]
    out = []
    out.extend((p + [a[-1]] for p in shuffle(a[:-1], b)))
    out.extend((p + [b[-1]] for p in shuffle(a, b[:-1])))
    return out

# Compute the shuffle square of the coefficient vector.
def shuffle_product_ell(ell, tensor):
    n = len(ell)
    return np.einsum('i,j,ijr->r', ell, ell, tensor[:n, :n, :], optimize=True)

# Express a quadratic polynomial in signature coordinates around the initial state.
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
    bsig = np.zeros((dim, 1 + dim + dim ** 2))
    asig = np.zeros((dim, dim, 1 + dim + dim ** 2))
    for i in range(dim):
        bsig[i] = poly_to_sig(b_const[i], b_lin[i, :], np.zeros((dim, dim)), x0)
        for j in range(dim):
            asig[i, j] = poly_to_sig(a_const[i, j], a_lin[i, j, :], a_quad[i, j, :], x0)
    return (bsig, asig)

# Integrate the signature semigroup over the VIX averaging window.
def integrated_signature_operator(G, window):
    n = G.shape[0]
    z = np.zeros_like(G)
    block_matrix = np.block([[G, np.eye(n)], [z, z]])
    block_exponential = expm(window * block_matrix)
    A = block_exponential[:n, n:]
    condition_number = float(np.linalg.cond(A))
    if not np.isfinite(condition_number) or condition_number > 1000000000000.0:
        raise np.linalg.LinAlgError(f'Integrated signature operator is singular/ill-conditioned: cond(A)={condition_number:.6g}')
    inverse_A = np.linalg.solve(A, np.eye(n))
    return (A, inverse_A)

# Build the signature generator in sparse matrix form.
def sparse_matrix_L(coeff, x0, level, dim):
    if not isinstance(level, int) or level < 1:
        raise ValueError('level must be a positive integer.')
    if not isinstance(dim, int) or dim < 2:
        raise ValueError('dim must be an integer greater than one.')
    d_minus_one = dim - 1
    generator_dimension = from_level_to_num(level, d_minus_one) + 1
    drift_signature, diffusion_signature = poly_to_sig_vec(coeff, x0)
    row_indices, column_indices, values = ([], [], [])
    for row_index in range(1, generator_dimension):
        word = from_num_to_word(row_index, d_minus_one)
        row_entries = {}

        # Accumulate shuffle multiplicities for the selected pair of words.
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

# Construct the shuffle map for the selected signature coordinates.
def selected_shuffle_product_matrix(selected_indices, d_minus_one):
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.ndim != 1 or selected_indices.size == 0 or np.unique(selected_indices).size != selected_indices.size:
        raise ValueError('selected_indices must be a non-empty vector of distinct indices.')
    if not isinstance(d_minus_one, int) or d_minus_one < 1 or np.any(selected_indices < 0):
        raise ValueError('d_minus_one and selected_indices are invalid.')
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

# Expand the row space until it is invariant under the generator.
def invariant_row_closure(initial_rows, generator, tolerance=1e-11, max_iterations=20):
    if initial_rows.ndim != 2 or generator.ndim != 2 or generator.shape[0] != generator.shape[1] or (initial_rows.shape[1] != generator.shape[0]):
        raise ValueError('initial_rows and generator have incompatible dimensions.')
    if not np.isfinite(tolerance) or tolerance <= 0.0 or (not isinstance(max_iterations, int)) or (max_iterations <= 0):
        raise ValueError('tolerance and max_iterations are invalid.')
    initial_dense = np.asarray(initial_rows.toarray(), dtype=float)
    _, singular_values, basis_rows = np.linalg.svd(initial_dense, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise np.linalg.LinAlgError('The initial row space is empty.')
    initial_threshold = tolerance * max(float(singular_values[0]), 1.0)
    initial_rank = int(np.count_nonzero(singular_values > initial_threshold))
    basis_rows = basis_rows[:initial_rank]
    rank_history = [initial_rank]
    for _ in range(max_iterations):
        generator_images = np.asarray(generator.T.dot(basis_rows.T).T, dtype=float)
        residual = generator_images - generator_images @ basis_rows.T @ basis_rows
        residual -= residual @ basis_rows.T @ basis_rows
        _, new_singular_values, new_directions = np.linalg.svd(residual, full_matrices=False)
        residual_threshold = tolerance * max(float(np.linalg.norm(generator_images, ord='fro')), 1.0)
        new_rank = int(np.count_nonzero(new_singular_values > residual_threshold))
        if new_rank == 0:
            break
        basis_rows = np.vstack((basis_rows, new_directions[:new_rank]))
        basis_rows = np.linalg.qr(basis_rows.T, mode='reduced')[0].T
        rank_history.append(basis_rows.shape[0])
    else:
        raise RuntimeError('Invariant closure did not converge.')
    generator_images = np.asarray(generator.T.dot(basis_rows.T).T, dtype=float)
    reduced_generator = generator_images @ basis_rows.T
    closure_residual = generator_images - reduced_generator @ basis_rows
    closure_error = float(np.linalg.norm(closure_residual, ord='fro') / max(float(np.linalg.norm(generator_images, ord='fro')), 1.0))
    if not np.isfinite(closure_error) or closure_error > 10.0 * tolerance:
        raise np.linalg.LinAlgError(f'The reduced space is not invariant: relative error={closure_error:.6g}')
    return (basis_rows, reduced_generator, closure_error, rank_history)

# Construct the invariant reduced representation of the variance.
def prepare_reduced_variance_model(coeff, x0, selected_indices, dim, window, tolerance=1e-11, max_iterations=20):
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError('selected_indices must be a non-empty vector.')
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
    if not np.isfinite(pair_map_condition_number) or pair_map_condition_number > 1000000000000.0:
        raise np.linalg.LinAlgError(f'The pair-coordinate map is singular/ill-conditioned: cond={pair_map_condition_number:.6g}')
    if not np.isfinite(pair_map_inverse_residual) or pair_map_inverse_residual > 100.0 * tolerance:
        raise np.linalg.LinAlgError(f'The pair-coordinate inverse is inaccurate: residual={pair_map_inverse_residual:.6g}')
    if not np.all(np.isfinite(shuffle_product_coordinates)) or not np.all(np.isfinite(pair_to_variance_features)):
        raise ValueError('The reduced variance maps contain non-finite values.')
    return {'pair_positions': pair_positions, 'shuffle_product_coordinates': shuffle_product_coordinates, 'pair_to_variance_features': pair_to_variance_features, 'generator': reduced_generator, 'integrated_operator': integrated_operator, 'inverse_integrated_operator': inverse_integrated_operator, 'dimension': int(reduced_generator.shape[0]), 'full_level': maximum_level, 'closure_error': closure_error, 'rank_history': rank_history, 'pair_map_condition_number': pair_map_condition_number, 'pair_map_inverse_residual': pair_map_inverse_residual}

# Map coefficient products into the reduced variance coordinates.
def reduced_shuffle_product_ell(ell, pair_positions, shuffle_product_coordinates):
    ell = np.asarray(ell, dtype=float)
    pair_positions = np.asarray(pair_positions, dtype=int)
    shuffle_product_coordinates = np.asarray(shuffle_product_coordinates, dtype=float)
    if ell.ndim != 1 or pair_positions.ndim != 2 or pair_positions.shape[1] != 2:
        raise ValueError('ell and pair_positions have incompatible dimensions.')
    if pair_positions.size == 0 or np.any(pair_positions < 0) or np.any(pair_positions >= ell.size):
        raise ValueError('pair_positions contain invalid ell indices.')
    if shuffle_product_coordinates.ndim != 2 or shuffle_product_coordinates.shape[0] != pair_positions.shape[0]:
        raise ValueError('shuffle_product_coordinates has incompatible dimensions.')
    pair_weights = ell[pair_positions[:, 0]] * ell[pair_positions[:, 1]]
    pair_weights[pair_positions[:, 0] != pair_positions[:, 1]] *= 2.0
    return pair_weights @ shuffle_product_coordinates

# Map unconstrained parameters into the admissible correlation ellipsoid.
def rho_b_from_parameters(parameters, rho_ou):
    parameters = np.asarray(parameters, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    if parameters.ndim != 1 or rho_ou.shape != (parameters.size, parameters.size) or (not np.all(np.isfinite(parameters))) or (not np.all(np.isfinite(rho_ou))):
        raise ValueError('parameters and rho_ou have incompatible or invalid dimensions.')
    cholesky_ou = np.linalg.cholesky(0.5 * (rho_ou + rho_ou.T))
    unit_ball_parameters = parameters / np.sqrt(1.0 + parameters @ parameters)
    rho_b = cholesky_ou @ unit_ball_parameters
    return rho_b

# Convert admissible correlations into unconstrained parameters.
def parameters_from_rho_b(rho_b, rho_ou):
    rho_b = np.asarray(rho_b, dtype=float)
    rho_ou = np.asarray(rho_ou, dtype=float)
    if rho_b.ndim != 1 or rho_ou.shape != (rho_b.size, rho_b.size) or (not np.all(np.isfinite(rho_b))) or (not np.all(np.isfinite(rho_ou))):
        raise ValueError('rho_b and rho_ou have incompatible or invalid dimensions.')
    cholesky_ou = np.linalg.cholesky(0.5 * (rho_ou + rho_ou.T))
    unit_ball_parameters = np.linalg.solve(cholesky_ou, rho_b)
    residual = 1.0 - unit_ball_parameters @ unit_ball_parameters
    if not np.isfinite(residual) or residual <= 0.0:
        raise ValueError('rho_b lies outside the admissible joint-correlation region.')
    return unit_ball_parameters / np.sqrt(residual)

# Evaluate the standard normal cumulative distribution function.
def std_norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))

# Evaluate the standard normal density.
def std_norm_pdf(x):
    x = np.asarray(x)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# Estimate the market forward using nearby call-put pairs.
def forward_from_put_call_parity(strikes, call_bid, call_ask, put_bid, put_ask, maturity, rate, n_atm_pairs=3):
    strikes = np.asarray(strikes, dtype=float)
    call_bid = np.asarray(call_bid, dtype=float)
    call_ask = np.asarray(call_ask, dtype=float)
    put_bid = np.asarray(put_bid, dtype=float)
    put_ask = np.asarray(put_ask, dtype=float)
    arrays = (strikes, call_bid, call_ask, put_bid, put_ask)
    if any((array.ndim != 1 for array in arrays)):
        raise ValueError('Parity inputs must be one-dimensional arrays.')
    if len({array.size for array in arrays}) != 1:
        raise ValueError('Parity inputs must have the same length.')
    if maturity <= 0.0 or not np.isfinite(maturity):
        raise ValueError('maturity must be finite and strictly positive.')
    if not np.isfinite(rate):
        raise ValueError('rate must be finite.')
    if n_atm_pairs <= 0:
        raise ValueError('n_atm_pairs must be a positive integer.')
    valid = np.isfinite(strikes) & np.isfinite(call_bid) & np.isfinite(call_ask) & np.isfinite(put_bid) & np.isfinite(put_ask) & (strikes > 0.0) & (call_bid >= 0.0) & (put_bid >= 0.0) & (call_ask >= call_bid) & (put_ask >= put_bid)
    if not np.any(valid):
        raise ValueError('No valid call-put pair is available for parity.')
    strikes = strikes[valid]
    call_mid = 0.5 * (call_bid[valid] + call_ask[valid])
    put_mid = 0.5 * (put_bid[valid] + put_ask[valid])
    count = min(int(n_atm_pairs), strikes.size)
    atm = np.argsort(np.abs(call_mid - put_mid), kind='stable')[:count]
    discount = math.exp(-rate * maturity)
    forwards = strikes[atm] + (call_mid[atm] - put_mid[atm]) / discount
    forwards = forwards[np.isfinite(forwards) & (forwards > 0.0)]
    if forwards.size == 0:
        raise ValueError('Put-call parity produced no finite positive forward.')
    return float(np.median(forwards))

# Evaluate a Black-76 call price.
def black76_call(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    d2 = d1 - vsqrt
    return float(discount * (forward * std_norm_cdf(d1) - strike * std_norm_cdf(d2)))

# Evaluate a Black-76 put price.
def black76_put(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    d2 = d1 - vsqrt
    return float(discount * (strike * std_norm_cdf(-d2) - forward * std_norm_cdf(-d1)))

# Evaluate Black-76 vega.
def black_vega(discount, forward, strike, maturity, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    return float(discount * forward * std_norm_pdf(d1) * math.sqrt(maturity))

# Evaluate the Black-76 forward delta of a call.
def black_call_delta(forward, strike, maturity, rate, vol):
    if forward <= 0 or strike <= 0 or maturity <= 0 or (vol <= 0):
        return np.nan
    vsqrt = vol * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * vsqrt * vsqrt) / vsqrt
    return float(math.exp(-rate * maturity) * std_norm_cdf(d1))

# Evaluate the Black-76 forward delta of the selected option type.
def black_option_delta(forward, strike, maturity, rate, vol, cp=1):
    if cp not in (-1, 1):
        raise ValueError('cp must be +1 for a call or -1 for a put.')
    call_delta = black_call_delta(forward, strike, maturity, rate, vol)
    if not np.isfinite(call_delta):
        return np.nan
    return call_delta if cp == 1 else call_delta - math.exp(-rate * maturity)

# Invert the Black-76 price to obtain implied volatility.
def implied_vol_black76(price, forward, strike, maturity, rate, cp=1, low=1e-08, high=8.0, tol=1e-08, max_iter=100):
    if cp not in (-1, 1):
        raise ValueError('cp must be +1 for a call or -1 for a put.')
    if not np.isfinite(price) or price <= 0 or forward <= 0 or (strike <= 0) or (maturity <= 0):
        return np.nan
    discount = math.exp(-rate * maturity)
    intrinsic = discount * max(cp * (forward - strike), 0.0)
    upper = discount * (forward if cp == 1 else strike)
    if price < intrinsic - 1e-08 or price > upper + 1e-08:
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

# Evaluate raw variance from the reduced signature features.
def variance_path(ell_shuffle, N, variance_signature_features, window):
    ell_shuffle = np.asarray(ell_shuffle, dtype=float)
    variance_signature_features = np.asarray(variance_signature_features, dtype=float)
    if variance_signature_features.ndim != 3:
        raise ValueError('variance_signature_features must have shape (n_times, n_paths, signature_dimension).')
    if ell_shuffle.ndim != 1 or ell_shuffle.size != variance_signature_features.shape[-1]:
        raise ValueError('ell_shuffle has incompatible dimensions.')
    if N <= 0 or N > variance_signature_features.shape[0]:
        raise ValueError('N must be between 1 and the number of available time points.')
    return window * np.einsum('tpi,i->tp', variance_signature_features[:N], ell_shuffle, optimize=True)

# Evaluate the signed linear VIX proxy in index points.
def vix_path(ell, N, truncated_signature_path):
    ell = np.asarray(ell, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    if truncated_signature_path.ndim != 3:
        raise ValueError('truncated_signature_path must have shape (n_times, n_paths, dimension_ell).')
    if ell.ndim != 1 or ell.size != truncated_signature_path.shape[-1]:
        raise ValueError('ell has incompatible dimensions.')
    if N < 0 or N >= truncated_signature_path.shape[0]:
        raise ValueError('N must be a valid time-grid index.')
    return 100.0 * np.einsum('tpi,i->tp', truncated_signature_path[:N + 1], ell, optimize=True)

# Return the absolute linear VIX proxy in VIX points.
def vix_option_proxy(raw_vix_points):
    raw = np.asarray(raw_vix_points, dtype=float)
    return np.abs(raw)


# Precompute conditional-signature propagators for a VIX future.
def precompute_vix_future_propagators(time_steps, maturity, generator):
    time_steps = np.asarray(time_steps, dtype=float)
    generator = np.asarray(generator, dtype=float)
    if time_steps.ndim != 1 or time_steps.size == 0:
        raise ValueError('time_steps must be a non-empty one-dimensional array.')
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError('generator must be a square matrix.')
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError('maturity must be finite and nonnegative.')
    if not np.all(np.isfinite(time_steps)) or np.any(time_steps < 0.0) or np.any(time_steps > maturity + 1e-12):
        raise ValueError('time_steps must lie between zero and maturity.')
    return np.stack([expm((maturity - time) * generator) for time in time_steps])

# Prepare the roll intervals and each contract's history from valuation time.
def precompute_vxx_roll_data(time_steps, futures_maturities, generator, previous_expiry=None):
    time_steps = np.asarray(time_steps, dtype=float)
    futures_maturities = np.asarray(futures_maturities, dtype=float)
    generator = np.asarray(generator, dtype=float)
    if time_steps.ndim != 1 or time_steps.size == 0:
        raise ValueError('time_steps must be a non-empty one-dimensional array.')
    if futures_maturities.ndim != 1 or futures_maturities.size < 2:
        raise ValueError('At least two VIX-future maturities are required.')
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError('generator must be a square matrix.')
    if not np.all(np.isfinite(time_steps)) or np.any(np.diff(time_steps) <= 0.0) or (not np.isclose(time_steps[0], 0.0)):
        raise ValueError('time_steps must be finite, strictly increasing and start from zero.')
    if not np.all(np.isfinite(futures_maturities)) or np.any(futures_maturities <= 0.0) or np.any(np.diff(futures_maturities) <= 0.0):
        raise ValueError('futures_maturities must be finite, positive and strictly increasing.')
    first_previous_expiry = 0.0 if previous_expiry is None else float(previous_expiry)
    if not np.isfinite(first_previous_expiry) or first_previous_expiry > time_steps[0] + 1e-12:
        raise ValueError('previous_expiry must be finite and no later than valuation time.')
    if first_previous_expiry >= futures_maturities[0]:
        raise ValueError('previous_expiry must precede the first live VIX-future maturity.')
    if time_steps[-1] > futures_maturities[-2] + 1e-12:
        raise ValueError('A second live VIX future is required over the whole time horizon.')
    histories = {}

    # Cache the complete pre-activation history of one futures contract.
    def history(maturity):
        if maturity not in histories:
            nodes = time_steps[time_steps <= maturity + 1e-12]
            histories[maturity] = precompute_vix_future_propagators(nodes, maturity, generator)
        return histories[maturity]

    roll_data = []
    horizon = time_steps[-1]
    for maturity_index in range(futures_maturities.size - 1):
        segment_previous_expiry = first_previous_expiry if maturity_index == 0 else futures_maturities[maturity_index - 1]
        front_expiry = futures_maturities[maturity_index]
        second_expiry = futures_maturities[maturity_index + 1]
        if maturity_index > 0 and horizon <= segment_previous_expiry + 1e-12:
            break
        segment_end = min(front_expiry, horizon)
        left_index = 0 if maturity_index == 0 else int(np.argmin(np.abs(time_steps - segment_previous_expiry)))
        right_index = int(np.argmin(np.abs(time_steps - segment_end)))
        if maturity_index > 0 and (not np.isclose(time_steps[left_index], segment_previous_expiry, atol=1e-12, rtol=1e-10)):
            raise ValueError('Every crossed roll date must belong to time_steps.')
        if not np.isclose(time_steps[right_index], segment_end, atol=1e-12, rtol=1e-10):
            raise ValueError('The end of every roll segment must belong to time_steps.')
        time_indices = np.arange(left_index, right_index + 1, dtype=int)
        front_history, second_history = history(front_expiry), history(second_expiry)
        roll_data.append((time_indices, segment_previous_expiry, front_expiry, second_expiry,
            front_history[time_indices], second_history[time_indices], front_history, second_history))
        if horizon <= front_expiry + 1e-12:
            break
    return tuple(roll_data)

# Evaluate the conditional linear VIX-future proxy.
def vix_future_path(ell, propagators, truncated_signature_path):
    ell = np.asarray(ell, dtype=float)
    propagators = np.asarray(propagators, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    if propagators.ndim != 3:
        raise ValueError('propagators must have shape (n_times, dimension_ell, dimension_ell).')
    if truncated_signature_path.ndim != 3:
        raise ValueError('truncated_signature_path must have shape (n_times, n_paths, dimension_ell).')
    dimension = propagators.shape[-1]
    if propagators.shape[1:] != (dimension, dimension):
        raise ValueError('propagator matrices must be square.')
    if ell.ndim != 1 or ell.size != dimension:
        raise ValueError('ell has incompatible dimensions.')
    if truncated_signature_path.shape[0] != propagators.shape[0] or truncated_signature_path.shape[-1] != dimension:
        raise ValueError('signature path and propagators have incompatible dimensions.')
    propagated_ell = np.einsum('i,tij->tj', ell, propagators, optimize=True)
    return 100.0 * np.einsum('tj,tpj->tp', propagated_ell, truncated_signature_path, optimize=True)

# Convert the decimal future stopping level into VIX points.
def future_floor_in_points(future_floor=1e-8):
    floor_points = 100.0 * float(future_floor)
    if not np.isfinite(floor_points) or floor_points <= 0.0:
        raise ValueError('future_floor must be finite and strictly positive in decimal VIX units.')
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

# Evaluate one VXX roll interval from globally stopped futures in VIX points.
def vxx_roll_segment_path(initial_vxx, time_steps, previous_expiry, front_expiry, front_future_path, second_future_path):
    time_steps = np.asarray(time_steps, dtype=float)
    front_future_path = np.asarray(front_future_path, dtype=float)
    second_future_path = np.asarray(second_future_path, dtype=float)
    if time_steps.ndim != 1 or time_steps.size == 0:
        raise ValueError('time_steps must be a non-empty one-dimensional array.')
    if front_future_path.ndim != 2 or second_future_path.ndim != 2:
        raise ValueError('front_future_path and second_future_path must be two-dimensional arrays.')
    if front_future_path.shape != second_future_path.shape or front_future_path.shape[0] != time_steps.size:
        raise ValueError('future paths and time_steps have incompatible dimensions.')
    if not np.isfinite(initial_vxx) or initial_vxx <= 0.0:
        raise ValueError('initial_vxx must be finite and strictly positive.')
    if not np.isfinite(previous_expiry) or not np.isfinite(front_expiry) or front_expiry <= previous_expiry:
        raise ValueError('The roll expiries must be finite and strictly increasing.')
    if not np.all(np.isfinite(time_steps)) or np.any(np.diff(time_steps) <= 0.0):
        raise ValueError('time_steps must be finite and strictly increasing.')
    if time_steps[0] < previous_expiry - 1e-12 or time_steps[-1] > front_expiry + 1e-12:
        raise ValueError('time_steps must lie inside the selected roll interval.')
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
    if not np.all(np.isfinite(log_vxx)) or np.any(log_vxx < np.log(np.finfo(float).tiny)) or np.any(log_vxx > np.log(np.finfo(float).max)):
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
            raw = vix_future_path(ell, propagators, signature_path[: len(propagators)])
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

# Evaluate the optional smooth positive variance transform.
def smooth_positive_floor(values, softness, floor=1e-12):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(softness) or softness <= 0.0:
        raise ValueError('softness must be finite and strictly positive.')
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError('floor must be finite and nonnegative.')
    shifted = values - floor
    root = np.hypot(shifted, 2.0 * softness)
    positive_branch = 0.5 * (shifted + root)
    negative_denominator = np.where(shifted < 0.0, root - shifted, 1.0)
    negative_branch = 2.0 * softness ** 2 / negative_denominator
    return floor + np.where(shifted >= 0.0, positive_branch, negative_branch)

# Transform raw variance according to the selected pricing convention.
def variance_for_pricing(raw_variance, variance_treatment='positive_part', variance_softness=1e-08):
    if variance_treatment == 'positive_part':
        return np.maximum(np.asarray(raw_variance, dtype=float), 0.0)
    if variance_treatment == 'smooth':
        return smooth_positive_floor(raw_variance, variance_softness)
    raise ValueError("variance_treatment must be 'positive_part' or 'smooth'.")

# Build the SPX stochastic exponential from variance and Brownian increments.
def stock_ratio_path(variance_left, brownian_increments, dt):
    variance_left = np.asarray(variance_left, dtype=float)
    brownian_increments = np.asarray(brownian_increments, dtype=float)
    if variance_left.ndim != 2 or brownian_increments.ndim != 2:
        raise ValueError('variance_left and brownian_increments must be two-dimensional arrays.')
    if variance_left.shape != brownian_increments.shape:
        raise ValueError('variance_left and brownian_increments must have the same shape.')
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError('dt must be finite and strictly positive.')
    n_steps, n_paths = variance_left.shape
    invalid_output = np.full((n_steps + 1, n_paths), np.nan)
    if n_steps == 0 or np.any(variance_left < 0.0) or (not np.all(np.isfinite(variance_left))) or (not np.all(np.isfinite(brownian_increments))):
        return invalid_output
    log_increments = np.sqrt(variance_left) * brownian_increments - 0.5 * variance_left * dt
    log_ratio = np.vstack((np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)))
    if not np.all(np.isfinite(log_ratio)) or np.max(log_ratio) > 700.0:
        return invalid_output
    return np.exp(log_ratio)

# Average discounted call or put payoffs over the simulated paths.
def monte_carlo_option_prices(terminal_values, strikes, maturity, rate, cp):
    terminal_values = np.asarray(terminal_values, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    cp = np.asarray(cp, dtype=int)
    if terminal_values.ndim != 1:
        raise ValueError('terminal_values must be one-dimensional.')
    if strikes.ndim != 1 or np.any(strikes <= 0.0):
        raise ValueError('strikes must be a one-dimensional positive array.')
    if cp.ndim != 1 or cp.shape != strikes.shape or np.any((cp != 1) & (cp != -1)):
        raise ValueError('cp must have the same shape as strikes and contain only +1 or -1.')
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError('maturity must be finite and strictly positive.')
    if not np.isfinite(rate):
        raise ValueError('rate must be finite.')
    if terminal_values.size == 0 or not np.all(np.isfinite(terminal_values)):
        return np.full(strikes.shape, np.nan)
    payoffs = np.maximum(cp[None, :] * (terminal_values[:, None] - strikes[None, :]), 0.0)
    return np.exp(-rate * maturity) * payoffs.mean(axis=0)

# Compute vega and bid-ask spread normalizations for the loss.
def precompute_loss_normalization(forward, strikes, maturity, rate, bid_iv, ask_iv, minimum=1e-12):
    strikes = np.asarray(strikes, dtype=float)
    bid_iv = np.asarray(bid_iv, dtype=float)
    ask_iv = np.asarray(ask_iv, dtype=float)
    if strikes.ndim != 1 or bid_iv.ndim != 1 or ask_iv.ndim != 1 or (strikes.shape != bid_iv.shape) or (strikes.shape != ask_iv.shape):
        raise ValueError('strikes, bid_iv and ask_iv must be one-dimensional arrays with the same shape.')
    if not np.isfinite(forward) or forward <= 0.0 or (not np.isfinite(maturity)) or (maturity <= 0.0) or (not np.isfinite(rate)):
        raise ValueError('forward, maturity and rate must be valid finite values.')
    if not np.all(np.isfinite(strikes)) or np.any(strikes <= 0.0):
        raise ValueError('strikes must contain finite and strictly positive values.')
    if not np.all(np.isfinite(bid_iv)) or not np.all(np.isfinite(ask_iv)) or np.any(bid_iv <= 0.0) or np.any(ask_iv < bid_iv):
        raise ValueError('Implied volatilities must be finite and satisfy 0 < bid_iv <= ask_iv.')
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError('minimum must be finite and strictly positive.')
    discount = np.exp(-rate * maturity)
    mid_iv = 0.5 * (bid_iv + ask_iv)
    vegas = np.array([black_vega(discount, forward, strike, maturity, vol) for strike, vol in zip(strikes, mid_iv)])
    normalization = np.maximum(vegas * (ask_iv - bid_iv), minimum)
    if not np.all(np.isfinite(normalization)):
        raise ValueError('The loss normalization contains non-finite values.')
    return normalization

# Evaluate the bid-ask calibration loss for one product.
def calibration_block_loss(model_prices, bid_prices, ask_prices, normalization, beta=0.98, model_future=None, market_future=None, deltas=None, weights=None):
    model_prices = np.asarray(model_prices, dtype=float)
    bid_prices = np.asarray(bid_prices, dtype=float)
    ask_prices = np.asarray(ask_prices, dtype=float)
    normalization = np.asarray(normalization, dtype=float)
    arrays = (model_prices, bid_prices, ask_prices, normalization)
    if any((array.ndim != 1 for array in arrays)) or len({array.shape for array in arrays}) != 1:
        raise ValueError('Model prices, market prices and normalization must be one-dimensional arrays with the same shape.')
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError('beta must belong to [0,1].')
    if not np.all(np.isfinite(bid_prices)) or not np.all(np.isfinite(ask_prices)) or np.any(bid_prices < 0.0) or np.any(ask_prices < bid_prices):
        raise ValueError('Bid and ask prices must be finite and satisfy 0 <= bid <= ask.')
    if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError('normalization must contain finite and strictly positive values.')
    if not np.all(np.isfinite(model_prices)):
        return 1000000000000.0
    use_future = model_future is not None or market_future is not None or deltas is not None
    if use_future:
        if model_future is None or market_future is None or deltas is None:
            raise ValueError('model_future, market_future and deltas must be supplied together.')
        deltas = np.asarray(deltas, dtype=float)
        if deltas.shape != model_prices.shape or not np.all(np.isfinite(deltas)):
            raise ValueError('deltas must be finite and have the same shape as model_prices.')
        if not np.isfinite(model_future) or not np.isfinite(market_future):
            return 1000000000000.0
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != model_prices.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError('weights must be finite, nonnegative and have the same shape as model_prices.')
    mid_prices = 0.5 * (bid_prices + ask_prices)
    mid_error = np.abs(model_prices - mid_prices)
    outside_band = np.maximum(bid_prices - model_prices, 0.0) + np.maximum(model_prices - ask_prices, 0.0)
    numerator = (1.0 - beta) * mid_error + beta * outside_band
    if use_future:
        numerator = numerator + np.abs(deltas * (model_future - market_future))
    scaled_residual = np.clip(numerator / normalization, 0.0, 1e+150)
    residual = scaled_residual ** 2
    if weights is not None:
        residual = weights * residual
    loss = np.sum(residual)
    return float(loss) if np.isfinite(loss) else 1000000000000.0

# Return the sampled minimum raw-variance margin.
def variance_positivity_constraint(
    ell, shuffle_tensor_nn, N, variance_signature_features, window, minimum_variance=0.0, variance_pair_positions=None, variance_shuffle_coordinates=None,
):
    ell = np.asarray(ell, dtype=float)
    use_reduced_variance = variance_pair_positions is not None or variance_shuffle_coordinates is not None
    if (variance_pair_positions is None) != (variance_shuffle_coordinates is None):
        raise ValueError('Both reduced-variance maps must be supplied together.')
    if not use_reduced_variance:
        shuffle_tensor_nn = np.asarray(shuffle_tensor_nn, dtype=float)
        if ell.ndim != 1 or shuffle_tensor_nn.ndim != 3 or shuffle_tensor_nn.shape[:2] != (ell.size, ell.size):
            raise ValueError('ell and shuffle_tensor_nn have incompatible dimensions.')
    if not np.isfinite(minimum_variance) or minimum_variance < 0.0:
        raise ValueError('minimum_variance must be finite and nonnegative.')
    if not np.all(np.isfinite(ell)):
        return -1000000000000.0
    with np.errstate(over='ignore', invalid='ignore'):
        ell_shuffle = reduced_shuffle_product_ell(ell, variance_pair_positions, variance_shuffle_coordinates) if use_reduced_variance else shuffle_product_ell(ell, shuffle_tensor_nn)
        raw_variance = variance_path(ell_shuffle, N, variance_signature_features, window)
    if not np.all(np.isfinite(raw_variance)):
        return -1000000000000.0
    return float(np.min(raw_variance) - minimum_variance)

# Price the three products and evaluate their joint calibration loss.
def joint_calibration_objective(
    ell, shuffle_tensor_nn, variance_signature_features, brownian_increments, truncated_signature_path, window, dt, N_spx, N_vix, time_steps_vxx,
    vix_future_propagator, vxx_roll_data, forward_spx, vxx_spot, strikes_spx, strikes_vix, strikes_vxx, maturity_spx, maturity_vix, maturity_vxx, rate,
    bid_spx, ask_spx, bid_vix, ask_vix, bid_vxx, ask_vxx, normalization_spx, normalization_vix, normalization_vxx, deltas_vix, market_future_vix,
    beta_spx=0.98, beta_vix=0.98, beta_vxx=0.98, lambda_spx=0.25, lambda_vix=0.5, lambda_vxx=0.25, variance_softness=1e-06, return_components=False,
    cp_spx=None, cp_vix=None, cp_vxx=None, variance_pair_positions=None, variance_shuffle_coordinates=None, variance_treatment='positive_part',
    future_floor=1e-08, vxx_annual_fee=0.0089, leverage_cap=100.0, price_rho_b=None, price_correlation_matrix=None, primary_brownian_increments=None,
    independent_brownian_normals=None,
):
    if not np.isfinite(vxx_annual_fee) or vxx_annual_fee < 0.0:
        raise ValueError('vxx_annual_fee must be a finite nonnegative annual decimal rate.')
    if not np.isfinite(leverage_cap) or leverage_cap <= 0.0:
        raise ValueError('The paper requires a finite, strictly positive leverage_cap.')
    future_floor_in_points(future_floor)
    ell = np.asarray(ell, dtype=float)
    truncated_signature_path = np.asarray(truncated_signature_path, dtype=float)
    time_steps_vxx = np.asarray(time_steps_vxx, dtype=float)
    vix_future_propagator = np.asarray(vix_future_propagator, dtype=float)
    lambdas = np.array([lambda_spx, lambda_vix, lambda_vxx], dtype=float)
    cp_spx = np.ones(np.asarray(strikes_spx).shape, dtype=int) if cp_spx is None else np.asarray(cp_spx, dtype=int)
    cp_vix = np.ones(np.asarray(strikes_vix).shape, dtype=int) if cp_vix is None else np.asarray(cp_vix, dtype=int)
    cp_vxx = np.ones(np.asarray(strikes_vxx).shape, dtype=int) if cp_vxx is None else np.asarray(cp_vxx, dtype=int)
    use_reduced_variance = variance_pair_positions is not None or variance_shuffle_coordinates is not None
    if ell.ndim != 1 or ell.size != truncated_signature_path.shape[-1]:
        raise ValueError('ell has incompatible dimensions.')
    if (variance_pair_positions is None) != (variance_shuffle_coordinates is None):
        raise ValueError('Both reduced-variance maps must be supplied together.')
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0.0) or np.sum(lambdas) <= 0.0:
        raise ValueError('Calibration weights must be finite, nonnegative and not all zero.')
    if cp_spx.shape != np.asarray(strikes_spx).shape or cp_vix.shape != np.asarray(strikes_vix).shape or cp_vxx.shape != np.asarray(strikes_vxx).shape or np.any((cp_spx != 1) & (cp_spx != -1)) or np.any((cp_vix != 1) & (cp_vix != -1)) or np.any((cp_vxx != 1) & (cp_vxx != -1)):
        raise ValueError('Every cp array must match its strike array and contain only +1 or -1.')
    if not np.all(np.isfinite(ell)):
        invalid_losses = {'SPX': 1000000000000.0, 'VIX': 1000000000000.0, 'VXX': 1000000000000.0}
        return (1000000000000.0, invalid_losses) if return_components else 1000000000000.0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ell_shuffle = reduced_shuffle_product_ell(ell, variance_pair_positions, variance_shuffle_coordinates) if use_reduced_variance else shuffle_product_ell(ell, shuffle_tensor_nn)
        variance_spx = variance_path(ell_shuffle, N_spx, variance_signature_features, window)
        effective_variance_spx = variance_for_pricing(variance_spx, variance_treatment, variance_softness)
        if price_rho_b is not None:
            brownian_increments = brownian_increments_with_saturation(price_rho_b, price_correlation_matrix, effective_variance_spx, primary_brownian_increments[:N_spx], np.sqrt(dt) * independent_brownian_normals[:N_spx], leverage_cap)
        if brownian_increments is None:
            raise ValueError('Supply primary Brownian drivers or explicit price increments.')
        stock_ratio_spx = stock_ratio_path(effective_variance_spx, brownian_increments[:N_spx], dt)
        terminal_spx = forward_spx * stock_ratio_spx[-1]
        model_prices_spx = monte_carlo_option_prices(terminal_spx, strikes_spx, maturity_spx, rate, cp_spx)
        terminal_vix = vix_option_proxy(vix_path(ell, N_vix, truncated_signature_path)[-1])
        model_prices_vix = monte_carlo_option_prices(terminal_vix, strikes_vix, maturity_vix, rate, cp_vix)
        model_future_vix_raw = 100.0 * ell @ vix_future_propagator @ truncated_signature_path[0, 0]
        model_future_vix = float(model_future_vix_raw)
        discounted_vxx_path = vxx_path(vxx_spot, ell, time_steps_vxx, truncated_signature_path[:time_steps_vxx.size], vxx_roll_data, future_floor=future_floor)
        terminal_vxx = np.exp((rate - vxx_annual_fee) * maturity_vxx) * discounted_vxx_path[-1]
        model_prices_vxx = monte_carlo_option_prices(terminal_vxx, strikes_vxx, maturity_vxx, rate, cp_vxx)
        loss_spx = calibration_block_loss(model_prices_spx, bid_spx, ask_spx, normalization_spx, beta_spx)
        loss_vix = calibration_block_loss(model_prices_vix, bid_vix, ask_vix, normalization_vix, beta_vix, model_future_vix, market_future_vix, deltas_vix)
        loss_vxx = calibration_block_loss(model_prices_vxx, bid_vxx, ask_vxx, normalization_vxx, beta_vxx)
        joint_loss = lambda_spx * loss_spx + lambda_vix * loss_vix + lambda_vxx * loss_vxx
    if model_future_vix_raw <= future_floor_in_points(future_floor):
        joint_loss = 1000000000000.0
    joint_loss = float(joint_loss) if np.isfinite(joint_loss) else 1000000000000.0
    calibration_details = {'SPX': float(loss_spx), 'VIX': float(loss_vix), 'VXX': float(loss_vxx), 'model_prices': {'SPX': model_prices_spx, 'VIX': model_prices_vix, 'VXX': model_prices_vxx}, 'model_future_vix': float(model_future_vix), 'model_future_vix_raw': float(model_future_vix_raw), 'leverage_saturated_fraction': float(np.mean(effective_variance_spx > leverage_cap ** 2)), 'stock_ratio_mean': float(np.mean(stock_ratio_spx[-1]))}
    if return_components:
        return (joint_loss, calibration_details)
    return joint_loss

# Apply the variance constraint to the full parameter vector.
def variance_positivity_constraint_with_correlations(calibration_parameters, dimension_ell, constraint_inputs):
    calibration_parameters = np.asarray(calibration_parameters, dtype=float)
    if calibration_parameters.ndim != 1 or calibration_parameters.size < dimension_ell:
        raise ValueError('calibration_parameters has incompatible dimensions.')
    return variance_positivity_constraint(calibration_parameters[:dimension_ell], **constraint_inputs)

# Convert calibration prices to implied volatilities and plot the three smiles.
def plot_calibration_smiles(
    model_prices_spx, model_prices_vix, model_prices_vxx, model_future_vix, strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx,
    bid_iv_spx, ask_iv_spx, bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx, maturity_spx, maturity_vix, maturity_vxx, rate, cp_spx, cp_vix, cp_vxx,
):
    model_iv_spx = np.array([implied_vol_black76(price, forward_spx, strike, maturity_spx, rate, cp) for price, strike, cp in zip(model_prices_spx, strikes_spx, cp_spx)])
    model_iv_vix = np.array([implied_vol_black76(price, forward_vix, strike, maturity_vix, rate, cp) for price, strike, cp in zip(model_prices_vix, strikes_vix, cp_vix)])
    model_iv_vxx = np.array([implied_vol_black76(price, forward_vxx, strike, maturity_vxx, rate, cp) for price, strike, cp in zip(model_prices_vxx, strikes_vxx, cp_vxx)])
    return plot_out_of_sample_smiles(model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix, strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx, bid_iv_spx, ask_iv_spx, bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx)

# Plot the implied-volatility smiles against their market bid-ask bands.
def plot_out_of_sample_smiles(
    model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix, strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx, bid_iv_spx,
    ask_iv_spx, bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx, *, model_label='calibrated iv', show=True,
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
    model_color = '#315efb'
    market_color = '#e45756'
    band_color = '#f3a6a5'
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120, constrained_layout=True)

    # Draw one model smile and the corresponding market band.
    def draw_smile(axis, x_values, model_iv, bid_iv, ask_iv, xlabel):
        axis.fill_between(x_values, bid_iv, ask_iv, color=band_color, alpha=0.3, label='Call bid-ask band')
        axis.plot(x_values, bid_iv, '+', color=market_color, markersize=6)
        axis.plot(x_values, ask_iv, '+', color=market_color, markersize=6)
        axis.plot(x_values, model_iv, 'o', color=model_color, linestyle='None', markersize=4.5, label=model_label)
        axis.set_xlabel(xlabel)
        axis.set_ylabel('Implied volatility')
        axis.grid(True, linestyle='--', alpha=0.25)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.legend(frameon=False)
    draw_smile(axes[0], moneyness_spx, model_iv_spx, bid_iv_spx, ask_iv_spx, 'Moneyness K/F')
    draw_smile(axes[1], strikes_vix, model_iv_vix, bid_iv_vix, ask_iv_vix, 'Strike')
    draw_smile(axes[2], moneyness_vxx, model_iv_vxx, bid_iv_vxx, ask_iv_vxx, 'Moneyness K/F')
    axes[1].axvline(forward_vix, color=market_color, linestyle='-.', linewidth=1.4, label='Market future')
    axes[1].axvline(model_future_vix, color=model_color, linestyle='--', linewidth=1.4, label='Model future')
    axes[1].legend(frameon=False)
    if show:
        plt.show()
    return (fig, axes)

# Build drift and covariance coefficients for two OU factors and one CIR.
def ou_cir_generator_coefficients(kappa, theta, eta, rho_ou, kappa_cir, theta_cir, eta_cir):
    b_const = np.zeros(4)
    b_lin = np.zeros((4, 4))
    a_const = np.zeros((4, 4))
    a_lin = np.zeros((4, 4, 4))
    a_quad = np.zeros((4, 4, 4, 4))
    b_const[0] = 1.0
    b_const[1:3] = np.asarray(kappa) * np.asarray(theta)
    b_lin[1:3, 1:3] = -np.diag(kappa)
    b_const[3] = kappa_cir * theta_cir
    b_lin[3, 3] = -kappa_cir
    a_const[1:3, 1:3] = np.asarray(rho_ou) * np.outer(eta, eta)
    a_lin[3, 3, 3] = eta_cir ** 2
    return (b_const, b_lin, a_const, a_lin, a_quad)

# Advance the CIR factor using the original implicit square-root scheme.
def cir_sqrt_implicit_step(current_cir, brownian_increment, dt, kappa_cir, theta_cir, eta_cir):
    current_cir = np.asarray(current_cir, dtype=float)
    brownian_increment = np.asarray(brownian_increment, dtype=float)
    if dt <= 0 or kappa_cir <= 0 or theta_cir <= 0 or (eta_cir < 0) or (4 * kappa_cir * theta_cir <= eta_cir ** 2):
        raise ValueError('The CIR square-root scheme requires positive dt,kappa,theta and 4*kappa*theta > eta**2.')
    if not np.all(np.isfinite(current_cir)) or np.any(current_cir < 0) or (not np.all(np.isfinite(brownian_increment))):
        raise ValueError('CIR states must be finite/nonnegative and Brownian increments finite.')
    if eta_cir == 0:
        return theta_cir + (current_cir - theta_cir) * np.exp(-kappa_cir * dt)
    a = 1.0 + 0.5 * kappa_cir * dt
    c = (0.5 * kappa_cir * theta_cir - 0.125 * eta_cir ** 2) * dt
    q = np.sqrt(current_cir) + 0.5 * eta_cir * brownian_increment
    root = np.sqrt(q * q + 4.0 * a * c)
    next_root = np.empty_like(q)
    positive = q >= 0
    next_root[positive] = (q[positive] + root[positive]) / (2.0 * a)
    next_root[~positive] = 2.0 * c / (root[~positive] - q[~positive])
    return next_root ** 2

# Construct the predictable SPX Brownian coupling with leverage saturation.
def brownian_increments_with_saturation(rho_b, rho_primary, variance_left, factor_increments, independent_increments, leverage_cap):
    rho_b = np.asarray(rho_b, dtype=float)
    rho_primary = np.asarray(rho_primary, dtype=float)
    variance_left = np.asarray(variance_left, dtype=float)
    factor_increments = np.asarray(factor_increments, dtype=float)
    independent_increments = np.asarray(independent_increments, dtype=float)
    if variance_left.ndim != 2 or independent_increments.shape != variance_left.shape or factor_increments.shape != variance_left.shape + (rho_b.size,):
        raise ValueError('Variance and Brownian increments have incompatible shapes.')
    if rho_b.ndim != 1 or rho_primary.shape != (rho_b.size, rho_b.size):
        raise ValueError('rho_b and rho_primary have incompatible shapes.')
    if np.isnan(leverage_cap) or leverage_cap <= 0.0:
        raise ValueError('leverage_cap must be strictly positive.')
    if not np.all(np.isfinite(rho_b)) or not np.all(np.isfinite(rho_primary)):
        raise ValueError('The correlations must be finite.')
    if not np.all(np.isfinite(variance_left)) or np.any(variance_left < 0.0) or (not np.all(np.isfinite(factor_increments))) or (not np.all(np.isfinite(independent_increments))):
        return np.full(variance_left.shape, np.nan)
    weights = np.linalg.solve(rho_primary, rho_b)
    q_rho = float(rho_b @ weights)
    if q_rho < -1e-12 or q_rho > 1.0 + 1e-12:
        raise ValueError('rho_b must satisfy rho_b.T @ inv(rho_primary) @ rho_b <= 1.')
    q_rho = float(np.clip(q_rho, 0.0, 1.0))
    sigma = np.sqrt(variance_left)
    chi = np.ones_like(sigma)
    np.divide(leverage_cap, sigma, out=chi, where=sigma > leverage_cap)
    correlated_increments = np.einsum('tpi,i->tp', factor_increments, weights, optimize=True)
    independent_weight = np.sqrt(np.maximum(1.0 - chi ** 2 * q_rho, 0.0))
    return chi * correlated_increments + independent_weight * independent_increments

# Construct the optional unsaturated coupling from the primary drivers.
def brownian_increments_from_ou_cir_drivers(rho_b, rho_primary, driver_increments, independent_normals, dt):
    rho_b = np.asarray(rho_b, dtype=float)
    rho_primary = np.asarray(rho_primary, dtype=float)
    driver_increments = np.asarray(driver_increments, dtype=float)
    independent_normals = np.asarray(independent_normals, dtype=float)
    if rho_b.shape != (3,) or rho_primary.shape != (3, 3) or driver_increments.shape != independent_normals.shape + (3,):
        raise ValueError('The OU/CIR coupling requires three primary Brownian drivers.')
    if not np.isfinite(dt) or dt <= 0 or (not np.all(np.isfinite(rho_b))):
        raise ValueError('Invalid dt or SPX correlations.')
    weights = np.linalg.solve(rho_primary, rho_b)
    residual = 1.0 - float(rho_b @ weights)
    if residual < -1e-12 or not np.isfinite(residual):
        raise ValueError('The joint OU/CIR/SPX correlation matrix is not positive semidefinite.')
    return np.einsum('tpi,i->tp', driver_increments, weights, optimize=True) + np.sqrt(dt * max(residual, 0.0)) * independent_normals

# Simulate exact OU transitions, implicit CIR steps and their Brownian drivers.
def simulate_time_extended_ou_cir_with_brownian(
    kappa, theta, eta, rho_ou, rho_b, x0, kappa_cir, theta_cir, eta_cir, y0_cir, Tmax, nsteps, npaths, seed=None, driver_steps=None,
    return_brownian_increments=True,
):
    kappa, theta, eta, x0 = [np.asarray(value, dtype=float) for value in (kappa, theta, eta, x0)]
    rho_ou = np.asarray(rho_ou, dtype=float)
    if any((value.shape != (2,) for value in (kappa, theta, eta, x0))) or rho_ou.shape != (2, 2):
        raise ValueError('This experiment uses exactly two OU factors and one CIR.')
    if not all((np.all(np.isfinite(value)) for value in (kappa, theta, eta, x0, rho_ou))) or np.any(kappa <= 0) or np.any(eta < 0):
        raise ValueError('Invalid OU parameters.')
    if not np.allclose(rho_ou, rho_ou.T) or not np.allclose(np.diag(rho_ou), 1.0):
        raise ValueError('rho_ou must be a symmetric correlation matrix.')
    if not np.all(np.isfinite([Tmax, kappa_cir, theta_cir, eta_cir, y0_cir])) or Tmax <= 0 or (not isinstance(nsteps, (int, np.integer))) or (not isinstance(npaths, (int, np.integer))) or (nsteps <= 0) or (npaths <= 0) or (y0_cir < 0):
        raise ValueError('Invalid simulation dimensions, horizon or CIR inputs.')
    cir_sqrt_implicit_step(np.array([y0_cir]), np.zeros(1), Tmax / nsteps, kappa_cir, theta_cir, eta_cir)
    rho_primary = np.eye(3)
    rho_primary[:2, :2] = rho_ou
    parameters_from_rho_b(np.asarray(rho_b), rho_primary)
    cholesky_ou = np.linalg.cholesky(rho_ou)
    dt = Tmax / nsteps
    time_grid = np.linspace(0.0, Tmax, nsteps + 1)
    decay = np.exp(-kappa * dt)
    integral = -np.expm1(-kappa * dt) / kappa
    kappa_sum = kappa[:, None] + kappa[None, :]
    innovation_covariance = rho_ou * np.outer(eta, eta) * -np.expm1(-kappa_sum * dt) / kappa_sum
    bridge_covariance = innovation_covariance - rho_ou * np.outer(eta * integral, eta * integral) / dt
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (bridge_covariance + bridge_covariance.T))
    if np.min(eigenvalues) < -1e-12 * max(float(np.max(np.diag(innovation_covariance))), 1e-30):
        raise ValueError('The conditional OU bridge covariance is invalid.')
    bridge_root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    rng = np.random.default_rng(seed)
    paths = np.empty((nsteps + 1, npaths, 4), dtype=float)
    paths[:, :, 0] = time_grid[:, None]
    paths[0, :, 1:3] = x0
    paths[0, :, 3] = y0_cir
    driver_steps = nsteps if driver_steps is None else driver_steps
    if not isinstance(driver_steps, (int, np.integer)) or not 0 <= driver_steps <= nsteps:
        raise ValueError('driver_steps must be an integer between zero and nsteps.')
    driver_increments = np.empty((driver_steps, npaths, 3), dtype=float)
    independent_normals = np.empty((driver_steps, npaths), dtype=float)
    for step in range(nsteps):
        normals = rng.standard_normal((npaths, 6))
        dw_ou = np.sqrt(dt) * (normals[:, :2] @ cholesky_ou.T)
        dw_cir = np.sqrt(dt) * normals[:, 2]
        bridge = normals[:, 3:5] @ bridge_root.T
        innovations = dw_ou * (eta * integral / dt) + bridge
        paths[step + 1, :, 1:3] = theta + (paths[step, :, 1:3] - theta) * decay + innovations
        paths[step + 1, :, 3] = cir_sqrt_implicit_step(paths[step, :, 3], dw_cir, dt, kappa_cir, theta_cir, eta_cir)
        if step < driver_steps:
            driver_increments[step, :, :2] = dw_ou
            driver_increments[step, :, 2] = dw_cir
            independent_normals[step] = normals[:, 5]
    brownian_increments = brownian_increments_from_ou_cir_drivers(rho_b, rho_primary, driver_increments, independent_normals, dt) if return_brownian_increments else None
    if not np.all(np.isfinite(paths)):
        raise RuntimeError('The OU/CIR simulation produced non-finite states.')
    return (paths, time_grid, brownian_increments, driver_increments, independent_normals)

# Evaluate the joint loss while fitting all three SPX correlations.
def joint_calibration_objective_with_cir_correlations(
    calibration_parameters, dimension_ell, rho_primary, driver_increments, independent_normals, objective_inputs, return_components=False,
):
    calibration_parameters = np.asarray(calibration_parameters, dtype=float)
    if calibration_parameters.shape != (dimension_ell + 3,):
        raise ValueError('Expected VIX coefficients followed by three SPX correlation parameters.')
    if not np.all(np.isfinite(calibration_parameters)):
        details = {'SPX': 1000000000000.0, 'VIX': 1000000000000.0, 'VXX': 1000000000000.0, 'rho_b': np.full(3, np.nan)}
        return (1000000000000.0, details) if return_components else 1000000000000.0
    rho_b = rho_b_from_parameters(calibration_parameters[dimension_ell:], rho_primary)
    result = joint_calibration_objective(calibration_parameters[:dimension_ell], brownian_increments=None, price_rho_b=rho_b, price_correlation_matrix=rho_primary, primary_brownian_increments=driver_increments, independent_brownian_normals=independent_normals, return_components=return_components, **objective_inputs)
    if return_components:
        loss, details = result
        details['rho_b'] = rho_b
        return (loss, details)
    return result

# Close the selected cubic signature coordinates under the generator.
def generator_closed_cubic_basis(coeff, x0, requested_words, tolerance=1e-12):
    generator = sparse_matrix_L(coeff, x0, 3, 4).toarray()
    selected = set(range(21))
    for word in requested_words:
        if len(word) != 3 or any((letter not in range(4) for letter in word)):
            raise ValueError('Cubic words must have three letters from (time,OU1,OU2,CIR).')
        selected.add(from_word_to_num(word, 3))
    while True:
        required = set(map(int, np.flatnonzero(np.any(np.abs(generator[sorted(selected)]) > tolerance, axis=0))))
        missing = required - selected
        if not missing:
            break
        selected.update(missing)
    indices = np.array(sorted(selected), dtype=int)
    excluded = np.setdiff1d(np.arange(generator.shape[0]), indices)
    error = np.linalg.norm(generator[np.ix_(indices, excluded)]) / max(np.linalg.norm(generator[indices]), 1.0)
    if error > tolerance:
        raise np.linalg.LinAlgError('The enriched VIX signature space is not generator-invariant.')
    words = [list(map(int, from_num_to_word(int(i), 3))) for i in indices if i >= 21]
    return (indices, generator[np.ix_(indices, indices)], float(error), words)

# Compute selected signatures and disk-backed reduced variance features.
def compute_cubic_signature_features(paths, selected_indices, reduced_variance_model, cache_directory, show_progress=True):
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    n_times, npaths, alphabet_size = paths.shape
    if alphabet_size != 4:
        raise ValueError('Expected exactly two OU factors, one CIR and time.')
    level = max((len(from_num_to_word(int(i), 3)) for i in selected_indices))
    k = len(selected_indices)
    signatures = np.lib.format.open_memmap(cache_directory / 'signature_selected_CIR.npy', mode='w+', dtype=float, shape=(n_times, npaths, k))
    features = np.lib.format.open_memmap(cache_directory / 'variance_selected_CIR.npy', mode='w+', dtype=float, shape=(n_times, npaths, reduced_variance_model['dimension']))
    pairs = reduced_variance_model['pair_positions']
    transform = reduced_variance_model['pair_to_variance_features']
    nonempty = selected_indices > 0
    if iisignature is None:
        if level not in (2, 3):
            raise ValueError('The NumPy signature implementation supports levels two and three.')
        full_dimension = from_level_to_num(level, 3) + 1
        state = np.zeros((npaths, full_dimension))
        state[:, 0] = 1.0
        for step in tqdm(range(n_times), disable=not show_progress):
            if step:
                dx = paths[step] - paths[step - 1]
                if level == 3:
                    third = state[:, 5:21].reshape(npaths, 4, 4, 1) * dx[:, None, None, :] + 0.5 * state[:, 1:5, None, None] * dx[:, None, :, None] * dx[:, None, None, :] + dx[:, :, None, None] * dx[:, None, :, None] * dx[:, None, None, :] / 6.0
                    state[:, 21:85] += third.reshape(npaths, 64)
                second = state[:, 1:5, None] * dx[:, None, :] + 0.5 * dx[:, :, None] * dx[:, None, :]
                state[:, 5:21] += second.reshape(npaths, 16)
                state[:, 1:5] += dx
            selected = state[:, selected_indices]
            pair_values = selected[:, pairs[:, 0]] * selected[:, pairs[:, 1]]
            values = pair_values @ transform.T
            if not np.all(np.isfinite(values)):
                raise RuntimeError('Non-finite enriched variance features.')
            signatures[step] = selected
            features[step] = values
        signatures.flush()
        features.flush()
        return (signatures, features)
    for p in tqdm(range(npaths), disable=not show_progress):
        sig = np.zeros((n_times, k))
        sig[:, 0] = 1
        sig[1:, nonempty] = iisignature.sig(paths[:, p, :], level, 2)[:, selected_indices[nonempty] - 1]
        pair_values = sig[:, pairs[:, 0]] * sig[:, pairs[:, 1]]
        values = pair_values @ transform.T
        if not np.all(np.isfinite(values)):
            raise RuntimeError('Non-finite enriched variance features.')
        signatures[:, p] = sig
        features[:, p] = values
    signatures.flush()
    features.flush()
    return (signatures, features)

# Build the quadratic variance constraints at selected path-time points.
def sampled_variance_matrices(features, pair_positions, shuffle_coordinates, window, indices, dimension_ell):
    flat = np.asarray(features).reshape(-1, features.shape[-1])
    pair_values = window * (flat[np.asarray(indices, dtype=int)] @ shuffle_coordinates.T)
    h = np.zeros((len(indices), dimension_ell, dimension_ell))
    left, right = pair_positions.T
    h[:, left, right] = pair_values
    h[:, right, left] = pair_values
    return h

# Run the supplied SLSQP constraint-exchange calibration.
def calibrate_cubic_with_active_constraints(
    objective, initial_parameters, dimension_ell, features, signatures, pair_positions, shuffle_coordinates, window, minimum_variance=1e-06, minimum_vix=1e-08,
    maxiter=240, stage_iterations=40, max_stages=8, ftol=3e-05, progress=True, initial_future_loadings=None, minimum_initial_future=0.0,
):
    from scipy.optimize import minimize, OptimizeResult
    from time import perf_counter
    progress_started = perf_counter()
    k = dimension_ell
    initial_parameters = np.asarray(initial_parameters, dtype=float)
    if initial_parameters.shape != (k + 3,):
        raise ValueError('Expected ell and three SPX-correlation parameters.')
    flat_signatures = np.asarray(signatures).reshape(-1, k)
    constraint_tolerance = 1e-10
    initial_future_loadings = np.empty((0, k)) if initial_future_loadings is None else np.asarray(initial_future_loadings, dtype=float)

    # Check sampled variance, signed VIX and every initial futures value.
    def full_check(parameters):
        ell = parameters[:k]
        product = reduced_shuffle_product_ell(ell, pair_positions, shuffle_coordinates)
        variance = variance_path(product, features.shape[0], features, window).ravel()
        vix = flat_signatures @ ell
        initial_futures = initial_future_loadings @ ell
        finite = np.all(np.isfinite(variance)) and np.all(np.isfinite(vix)) and np.all(np.isfinite(initial_futures))
        feasible = (finite and np.min(variance) >= minimum_variance - constraint_tolerance
            and np.min(vix) >= minimum_vix - constraint_tolerance
            and np.all(initial_futures >= minimum_initial_future - constraint_tolerance))
        return (bool(feasible), variance, vix)
    anchor = initial_parameters.copy()
    anchor[:k] = 0
    anchor[0] = max(abs(float(initial_parameters[0])), 0.2, np.sqrt(minimum_variance) * 2)
    current = initial_parameters.copy()
    warm_weight = 1.0
    for _ in range(50):
        feasible, variance, vix = full_check(current)
        if feasible:
            break
        warm_weight *= 0.8
        current = anchor + warm_weight * (initial_parameters - anchor)
    else:
        raise RuntimeError('Could not construct a sampled-admissible starting point.')
    sample = flat_signatures[::max(1, len(flat_signatures) // 20000)]
    rms = np.sqrt(np.mean(sample * sample, axis=0))
    scale = np.r_[np.minimum(0.05 / np.maximum(rms, 1e-05), 5000.0), np.ones(3)]
    scale[0] = 0.1
    best = current.copy()
    best_loss = float(objective(best))
    if not np.isfinite(best_loss) or best_loss >= 1000000000000.0:
        raise RuntimeError('Invalid objective at repaired warm start.')
    active_variance = set()
    active_vix = set()
    history = []
    total_iterations = 0
    converged = False

    # Add the worst sampled constraints and each path minimum to the active set.
    def add_worst(values, active, count=48):
        count = min(count, len(values))
        active.update(map(int, np.argpartition(values, count - 1)[:count]))
        table = values.reshape(signatures.shape[:2])
        times = np.argmin(table, axis=0)
        active.update(map(int, times * table.shape[1] + np.arange(table.shape[1])))

    # Retain the best candidate satisfying the complete sampled constraints.
    def record_candidate(parameters):
        nonlocal best, best_loss
        ok, rv, yv = full_check(parameters)
        loss = float(objective(parameters)) if ok else float('inf')
        if not ok and np.all(np.isfinite(parameters)):
            for weight in [0.5, 0.25, 0.125, 0.0625]:
                damped = best + weight * (parameters - best)
                safe, _, _ = full_check(damped)
                if safe:
                    damped_loss = float(objective(damped))
                    if np.isfinite(damped_loss) and damped_loss < best_loss:
                        best = damped.copy()
                        best_loss = damped_loss
                    break
        if ok and np.isfinite(loss) and (loss < best_loss):
            best = parameters.copy()
            best_loss = loss
        return (ok, rv, yv, loss)
    progress_header = f"{'Event':<6} {'Iter':>5} {'Total':>6} {'Best loss':>14} {'Min variance':>14} {'Min VIX':>10} {'Trial':>10} {'Time':>9}"

    # Print optimizer progress from the quantities already evaluated.
    def print_progress_row(event, iteration, total, feasible, raw_variance, raw_vix):
        elapsed = int(perf_counter() - progress_started)
        elapsed_text = f'{elapsed // 60:02d}:{elapsed % 60:02d}'
        status = 'OK' if feasible else 'VIOLATED'
        print(f'{event:<6} {iteration:5d} {total:6d} {best_loss:14.6e} {np.min(raw_variance):14.5e} {100 * np.min(raw_vix):10.4f} {status:>10} {elapsed_text:>9}', flush=True)
    if progress:
        print('\n' + '=' * len(progress_header), flush=True)
        print('CONSTRAINED CALIBRATION', flush=True)
        print(f'Feasible starting loss: {best_loss:.6e} | warm-start weight: {warm_weight:.4f}', flush=True)
        print('Best loss: best feasible checkpoint. Minima / Trial: current proposal.', flush=True)
        print('Min VIX is in index points. Updates every 10 iterations and at each stage end.', flush=True)
        print('=' * len(progress_header), flush=True)
    for stage in range(max_stages):
        if total_iterations >= maxiter:
            break
        _, variance, vix = full_check(current)
        add_worst(variance, active_variance)
        add_worst(vix, active_vix, 24)
        h = sampled_variance_matrices(features, pair_positions, shuffle_coordinates, window, sorted(active_variance), k)
        linear = flat_signatures[sorted(active_vix)].copy()
        stage_iteration_limit = min(stage_iterations, maxiter - total_iterations)
        if progress:
            print(f'\nSTAGE {stage + 1:02d}/{max_stages:02d} | iteration limit: {stage_iteration_limit} | active constraints: variance {len(h)}, VIX {len(linear)}', flush=True)
            print(progress_header, flush=True)
            print('-' * len(progress_header), flush=True)

        # Evaluate active variance and VIX constraints and all initial futures.
        def inequalities(q):
            ell = (q * scale)[:k]
            return np.r_[np.einsum('i,mij,j->m', ell, h, ell) - minimum_variance, linear @ ell - minimum_vix, initial_future_loadings @ ell - minimum_initial_future]

        # Differentiate the active constraints in scaled optimization coordinates.
        def inequality_jacobian(q):
            ell = (q * scale)[:k]
            jac = np.zeros((len(h) + len(linear) + len(initial_future_loadings), k + 3))
            jac[:len(h), :k] = 2 * np.einsum('mij,j->mi', h, ell)
            jac[len(h):len(h) + len(linear), :k] = linear
            jac[len(h) + len(linear):, :k] = initial_future_loadings
            return jac * scale[None, :]
        count = [0]

        # Check accepted optimizer iterates at the original monitoring frequency.
        def callback(q):
            count[0] += 1
            if count[0] % 10 == 0:
                ok, rv, yv, loss = record_candidate(q * scale)
                if progress:
                    print_progress_row('CHECK', count[0], total_iterations + count[0], ok, rv, yv)
        q0 = current / scale
        result = minimize(lambda q: float(objective(q * scale)), q0, method='SLSQP', bounds=list(zip(q0 - 0.3, q0 + 0.3)), constraints=({'type': 'ineq', 'fun': inequalities, 'jac': inequality_jacobian},), callback=callback, options={'maxiter': min(stage_iterations, maxiter - total_iterations), 'ftol': ftol, 'eps': 1e-06, 'disp': False})
        proposal = result.x * scale
        ok, variance, vix, loss = record_candidate(proposal)
        total_iterations += int(result.nit)
        history.append({'stage': stage + 1, 'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit), 'sample_admissible': ok, 'minimum_variance': float(np.min(variance)), 'minimum_vix_fraction': float(np.min(vix)), 'best_feasible_loss': best_loss, 'active_variance': len(active_variance), 'active_vix': len(active_vix)})
        if progress:
            print_progress_row('END', int(result.nit), total_iterations, ok, variance, vix)
            print(f'Stage stop: {result.message}', flush=True)
        if ok and result.success and (np.max(np.abs(result.x - q0)) < 0.299):
            converged = np.allclose(best, proposal, rtol=1e-07, atol=1e-09)
            break
        add_worst(variance, active_variance, 64)
        add_worst(vix, active_vix, 32)
        current = best.copy()
    ok, variance, vix = full_check(best)
    if not ok:
        raise RuntimeError('Internal error: selected checkpoint is not sampled-admissible.')
    if progress:
        print('\n' + '=' * len(progress_header), flush=True)
        print('CALIBRATION SUMMARY', flush=True)
        print(f"Status       : {('CONVERGED' if converged else 'STOPPED WITHOUT CONFIRMED CONVERGENCE')}", flush=True)
        print(f'Iterations   : {total_iterations} / {maxiter} | stages: {len(history)} / {max_stages}', flush=True)
        print(f'Best loss    : {best_loss:.8e}', flush=True)
        print(f'Min variance : {np.min(variance):.8e} | min VIX: {100 * np.min(vix):.4f}', flush=True)
        print('Returned point: feasible on the complete calibration sample.', flush=True)
        print('=' * len(progress_header), flush=True)
    return OptimizeResult(x=best, fun=best_loss, success=converged, nit=total_iterations, message='Converged with full sampled admissibility check.' if converged else 'Iteration/stage budget reached; returning best fully checked admissible candidate.', stage_history=history, warm_start_weight=warm_weight, minimum_variance=float(np.min(variance)), minimum_vix=float(np.min(vix)), parameter_scale=scale)

# Create an isolated disk cache for one calibration run.
def create_signature_cache():
    from tempfile import mkdtemp
    root = Path(__file__).resolve().with_name('cache_CIR')
    root.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix='run_', dir=root))

# Normalize the three product losses and return their weighted contributions.
def normalized_joint_loss(block_losses, *, lambda_spx, lambda_vix, lambda_vxx, loss_scales, objective_scale):
    contributions = {'SPX': objective_scale * lambda_spx * float(block_losses['SPX']) / loss_scales['SPX'], 'VIX': objective_scale * lambda_vix * float(block_losses['VIX']) / loss_scales['VIX'], 'VXX': objective_scale * lambda_vxx * float(block_losses['VXX']) / loss_scales['VXX']}
    return (float(sum(contributions.values())), contributions)

# Evaluate the normalized objective with the original invalid-price penalty.
def normalized_cir_objective(parameters, *, objective_with_components, normalized_joint_loss):
    raw_loss, block_losses = objective_with_components(parameters)
    if not np.isfinite(raw_loss) or raw_loss >= 1000000000000.0:
        return 1000000000000.0
    return normalized_joint_loss(block_losses)[0]

# Build the fixed quadratic variance map for out-of-sample pricing.
def fixed_ell_variance_matrix(ell, reduced_model, window):
    ell = np.asarray(ell, dtype=float)
    pairs = reduced_model['pair_positions']
    product = reduced_shuffle_product_ell(ell, pairs, reduced_model['shuffle_product_coordinates'])
    weights = window * (reduced_model['pair_to_variance_features'].T @ product)
    matrix = np.zeros((ell.size, ell.size))
    np.add.at(matrix, tuple(pairs.T), weights)
    return matrix

# Update level-three signatures and raw variance by Chen concatenation.
def compute_fixed_cir_signatures_and_variance(paths, selected_indices, variance_matrix):
    n_times, npaths, dimension = paths.shape
    selected_indices = np.asarray(selected_indices, dtype=int)
    signatures = np.empty((n_times, npaths, selected_indices.size))
    raw_variance = np.empty((n_times, npaths))
    state = np.zeros((npaths, 85))
    state[:, 0] = 1.0
    for node in range(n_times):
        if node:
            dx = paths[node] - paths[node - 1]
            third = state[:, 5:21].reshape(npaths, 4, 4, 1) * dx[:, None, None, :] + 0.5 * state[:, 1:5, None, None] * dx[:, None, :, None] * dx[:, None, None, :] + dx[:, :, None, None] * dx[:, None, :, None] * dx[:, None, None, :] / 6.0
            state[:, 21:85] += third.reshape(npaths, 64)
            second = state[:, 1:5, None] * dx[:, None, :] + 0.5 * dx[:, :, None] * dx[:, None, :]
            state[:, 5:21] += second.reshape(npaths, 16)
            state[:, 1:5] += dx
        selected = state[:, selected_indices]
        signatures[node] = selected
        raw_variance[node] = np.sum(selected @ variance_matrix * selected, axis=1)
    return (signatures, raw_variance)

# Invert call prices using the original CIR Black-76 convention.
def invert_prices(prices, forward, strikes, maturity, *, r):
    return np.array([implied_vol_black76(float(price), forward, float(strike), maturity, r, cp=1) if np.isfinite(price) else np.nan for price, strike in zip(prices, strikes)])
