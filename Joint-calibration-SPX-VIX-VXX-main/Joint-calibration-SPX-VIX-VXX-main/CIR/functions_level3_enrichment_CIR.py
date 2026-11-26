import math
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.special import ndtr

try:
    import iisignature
except ImportError:
    iisignature = None

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

# Differentiate a weighted sum of reduced quadratic variance coordinates.
def reduced_variance_gradient(ell, pair_positions, shuffle_coordinates, feature_sensitivity):
    left, right = pair_positions.T
    pair_sensitivity = shuffle_coordinates @ feature_sensitivity
    gradient = np.zeros_like(ell)
    np.add.at(gradient, left, 2.0 * pair_sensitivity * ell[right])
    off_diagonal = left != right
    np.add.at(gradient, right[off_diagonal], 2.0 * pair_sensitivity[off_diagonal] * ell[left[off_diagonal]])
    return gradient

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
    if isinstance(variance_signature_features, dict):
        signatures = variance_signature_features['signature_path']
        if N <= 0 or N > signatures.shape[0]:
            raise ValueError('N must be between 1 and the number of available time points.')
        matrix = variance_quadratic_matrix(ell_shuffle, variance_signature_features, signatures.shape[-1], window)
        selected = signatures[:N]
        return np.einsum('tpi,tpi->tp', selected @ matrix, selected, optimize=True)
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
def vxx_path(initial_vxx, ell, time_steps, truncated_signature_path, roll_data, future_floor=1e-8, return_diagnostics=False, *, future_paths=None):
    floor_points = future_floor_in_points(future_floor)
    times = np.asarray(time_steps, dtype=float)
    signatures = np.asarray(truncated_signature_path, dtype=float)
    ell = np.asarray(ell, dtype=float)
    n_times, n_paths = signatures.shape[:2]
    invalid = np.full((n_times, n_paths), np.nan)
    diagnostics = {
        "future_floor_decimal": float(future_floor),
        "future_floor_points": floor_points,
        "future_stopping_scheme": "first_grid_hit_freeze_at_m",
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

    if future_paths is None:
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

# Differentiate the bid-ask loss with respect to option prices and the future.
def calibration_block_loss_gradient(model_prices, bid_prices, ask_prices, normalization, beta, model_future=None, market_future=None, deltas=None):
    mid_prices = 0.5 * (bid_prices + ask_prices)
    outside_band = np.maximum(bid_prices - model_prices, 0.0) + np.maximum(model_prices - ask_prices, 0.0)
    numerator = (1.0 - beta) * np.abs(model_prices - mid_prices) + beta * outside_band
    if model_future is not None:
        numerator += np.abs(deltas * (model_future - market_future))
    price_slope = (1.0 - beta) * np.sign(model_prices - mid_prices) + beta * ((model_prices > ask_prices).astype(float) - (model_prices < bid_prices).astype(float))
    residual_slope = 2.0 * numerator / normalization**2
    future_gradient = 0.0 if model_future is None else float(residual_slope @ np.abs(deltas) * np.sign(model_future - market_future))
    return residual_slope * price_slope, future_gradient

# Return the sampled minimum raw-variance margin.
def variance_positivity_constraint(
    ell, shuffle_tensor_nn, N, variance_signature_features, window, minimum_variance=0.0, variance_pair_positions=None, variance_shuffle_coordinates=None, return_jacobian=False,
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
        return (-1e12, np.zeros_like(ell)) if return_jacobian else -1e12
    with np.errstate(over='ignore', invalid='ignore'):
        ell_shuffle = reduced_shuffle_product_ell(ell, variance_pair_positions, variance_shuffle_coordinates) if use_reduced_variance else shuffle_product_ell(ell, shuffle_tensor_nn)
        raw_variance = variance_path(ell_shuffle, N, variance_signature_features, window)
    if not np.all(np.isfinite(raw_variance)):
        return (-1e12, np.zeros_like(ell)) if return_jacobian else -1e12
    active_index = np.unravel_index(np.argmin(raw_variance), raw_variance.shape)
    margin = float(raw_variance[active_index] - minimum_variance)
    if not return_jacobian:
        return margin
    if isinstance(variance_signature_features, dict):
        signature = variance_signature_features['signature_path'][active_index]
        left, right = variance_signature_features['pair_positions'].T
        active_features = variance_signature_features['pair_to_variance_features'] @ (signature[left] * signature[right])
    else:
        active_features = np.asarray(variance_signature_features)[active_index]
    if use_reduced_variance:
        gradient = reduced_variance_gradient(ell, variance_pair_positions, variance_shuffle_coordinates, window * active_features)
    else:
        matrix = window * (shuffle_tensor_nn @ active_features)
        gradient = (matrix + matrix.T) @ ell
    return margin, gradient

# Price the three products and evaluate their joint calibration loss.
def joint_calibration_objective(
    ell, shuffle_tensor_nn, variance_signature_features, brownian_increments, truncated_signature_path, window, dt, N_spx, N_vix, time_steps_vxx,
    vix_future_propagator, vxx_roll_data, forward_spx, vxx_spot, strikes_spx, strikes_vix, strikes_vxx, maturity_spx, maturity_vix, maturity_vxx, rate,
    bid_spx, ask_spx, bid_vix, ask_vix, bid_vxx, ask_vxx, normalization_spx, normalization_vix, normalization_vxx, deltas_vix, market_future_vix,
    beta_spx=0.98, beta_vix=0.98, beta_vxx=0.98, lambda_spx=0.25, lambda_vix=0.5, lambda_vxx=0.25, variance_softness=1e-06, return_components=False,
    cp_spx=None, cp_vix=None, cp_vxx=None, variance_pair_positions=None, variance_shuffle_coordinates=None, variance_treatment='positive_part',
    future_floor=1e-08, vxx_annual_fee=0.0089, leverage_cap=100.0, price_rho_b=None, price_correlation_matrix=None, primary_brownian_increments=None,
    independent_brownian_normals=None, use_price_control_variates=False, deltas_spx=None, vix_control_loadings=None, collect_gradient_inputs=False,
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
        raw_terminal_vix = vix_path(ell, N_vix, truncated_signature_path)[-1]
        terminal_vix = vix_option_proxy(raw_terminal_vix)
        model_prices_vix = monte_carlo_option_prices(terminal_vix, strikes_vix, maturity_vix, rate, cp_vix)
        model_future_vix_raw = 100.0 * ell @ vix_future_propagator @ truncated_signature_path[0, 0]
        model_future_vix = float(model_future_vix_raw)
        if use_price_control_variates:
            model_prices_spx -= deltas_spx * (np.mean(terminal_spx) - forward_spx)
            model_prices_vix -= deltas_vix * (np.mean(raw_terminal_vix) - 100.0 * (ell @ vix_control_loadings))
        signatures_vxx = truncated_signature_path[:time_steps_vxx.size]
        future_paths = stopped_vxx_future_paths(ell, signatures_vxx, vxx_roll_data, future_floor) if collect_gradient_inputs else None
        discounted_vxx_path = vxx_path(vxx_spot, ell, time_steps_vxx, signatures_vxx, vxx_roll_data, future_floor=future_floor, future_paths=future_paths)
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
        if collect_gradient_inputs:
            calibration_details['gradient_inputs'] = {
                'raw_variance_spx': variance_spx, 'variance_spx': effective_variance_spx, 'terminal_spx': terminal_spx,
                'raw_terminal_vix': raw_terminal_vix, 'terminal_vix': terminal_vix, 'terminal_vxx': terminal_vxx,
                'future_paths': future_paths,
            }
        return (joint_loss, calibration_details)
    return joint_loss

# Apply the variance constraint to the full parameter vector.
def variance_positivity_constraint_with_correlations(calibration_parameters, dimension_ell, constraint_inputs, return_jacobian=False, number_of_time_blocks=1):
    calibration_parameters = np.asarray(calibration_parameters, dtype=float)
    if calibration_parameters.ndim != 1 or calibration_parameters.size < dimension_ell:
        raise ValueError('calibration_parameters has incompatible dimensions.')
    ell = calibration_parameters[:dimension_ell]
    if number_of_time_blocks == 1:
        result = variance_positivity_constraint(ell, return_jacobian=return_jacobian, **constraint_inputs)
    else:
        features = constraint_inputs['variance_signature_features']
        n_times = constraint_inputs['N']
        edges = np.linspace(0, n_times, min(number_of_time_blocks, n_times) + 1, dtype=int)
        results = []
        for start, stop in zip(edges[:-1], edges[1:]):
            block_features = {**features, 'signature_path': features['signature_path'][start:stop]} if isinstance(features, dict) else features[start:stop]
            block_inputs = {**constraint_inputs, 'N': stop - start, 'variance_signature_features': block_features}
            results.append(variance_positivity_constraint(ell, return_jacobian=return_jacobian, **block_inputs))
        result = (np.array([value for value, _ in results]), np.stack([gradient for _, gradient in results])) if return_jacobian else np.asarray(results)
    if not return_jacobian:
        return result
    margin, ell_gradient = result
    gradient = np.zeros(ell_gradient.shape[:-1] + calibration_parameters.shape)
    gradient[..., :dimension_ell] = ell_gradient
    return margin, gradient

# Plot the implied-volatility smiles against their market bid-ask bands.
def plot_out_of_sample_smiles(
    model_iv_spx, model_iv_vix, model_iv_vxx, model_future_vix, strikes_spx, strikes_vix, strikes_vxx, forward_spx, forward_vix, forward_vxx, bid_iv_spx,
    ask_iv_spx, bid_iv_vix, ask_iv_vix, bid_iv_vxx, ask_iv_vxx, *, model_label='Model IV', show=True, model_forward_vix=None,
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
        axis.fill_between(x_values, bid_iv, ask_iv, color=band_color, alpha=0.3, label='Market bid-ask IV')
        axis.plot(x_values, bid_iv, '+', color=market_color, markersize=6)
        axis.plot(x_values, ask_iv, '+', color=market_color, markersize=6)
        axis.plot(x_values, model_iv, 'o', color=model_color, linestyle='None', markersize=4.5, label=model_label)
        axis.set_xlabel(xlabel)
        axis.set_ylabel('Implied volatility')
        axis.grid(True, linestyle='--', alpha=0.25)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    draw_smile(axes[0], moneyness_spx, model_iv_spx, bid_iv_spx, ask_iv_spx, 'Moneyness K/F')
    draw_smile(axes[1], strikes_vix, model_iv_vix, bid_iv_vix, ask_iv_vix, 'Strike')
    draw_smile(axes[2], moneyness_vxx, model_iv_vxx, bid_iv_vxx, ask_iv_vxx, r'Moneyness $K/F_{\mathrm{mkt}}$')
    axes[1].axvline(forward_vix, color=market_color, linestyle='-.', linewidth=1.4, label='Market future')
    if model_forward_vix is None:
        axes[1].axvline(model_future_vix, color=model_color, linestyle='--', linewidth=1.4, label='Model future')
    else:
        axes[1].axvline(model_forward_vix, color=model_color, linestyle=':', linewidth=1.4, label='Model future')
    for axis in axes:
        axis.legend(frameon=False)
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

# Compute the VIX control mean for the piecewise-linear simulated paths.
def expected_selected_signature_on_cir_grid(kappa, theta, eta, rho_ou, x0, kappa_cir, theta_cir, eta_cir, y0_cir,
    time_grid, selected_indices, *, state_grid_size=257, quadrature_order=24):
    from scipy.interpolate import CubicSpline

    time_grid = np.asarray(time_grid, dtype=float)
    selected_indices = np.asarray(selected_indices, dtype=int)
    if time_grid.size == 1:
        return (selected_indices == 0).astype(float)
    dt = float(time_grid[1] - time_grid[0])
    if dt <= 0.0 or not np.allclose(np.diff(time_grid), dt, rtol=1e-10, atol=0.0):
        raise ValueError('The CIR control mean requires the uniform simulation grid.')
    kappa, theta, eta, x0 = map(lambda value: np.asarray(value, dtype=float), (kappa, theta, eta, x0))
    times = time_grid - time_grid[0]
    mean_ou = theta + (x0 - theta) * np.exp(-times[:, None] * kappa)
    ou_increments = np.diff(mean_ou, axis=0)
    decay = np.r_[1.0, np.exp(-kappa * dt)]
    drift_increment = decay - 1.0
    kappa_sum = kappa[:, None] + kappa[None, :]
    innovation = np.zeros((3, 3))
    innovation[1:, 1:] = rho_ou * np.outer(eta, eta) * -np.expm1(-kappa_sum * dt) / kappa_sum
    covariance = np.zeros_like(innovation)
    second_signature = np.zeros_like(innovation)
    for time, mean, increment in zip(times[:-1], mean_ou[:-1], ou_increments):
        mean_signature = np.r_[time, mean - x0]
        mean_increment = np.r_[dt, increment]
        increment_covariance = covariance * np.outer(drift_increment, drift_increment) + innovation
        second_signature += np.outer(mean_signature, mean_increment) + covariance * drift_increment[None, :]
        second_signature += 0.5 * (np.outer(mean_increment, mean_increment) + increment_covariance)
        covariance = covariance * np.outer(decay, decay) + innovation
    mean_signature = np.r_[times[-1], mean_ou[-1] - x0]
    expected = np.empty(selected_indices.size)
    cir_words = {}
    for i, index in enumerate(selected_indices):
        word = tuple(from_num_to_word(int(index), 3))
        if 3 in word:
            if sum(letter in (1, 2) for letter in word) > 1:
                raise ValueError('CIR control words may contain at most one OU letter.')
            cir_words[i] = word
        elif not word:
            expected[i] = 1.0
        elif len(word) == 1:
            expected[i] = mean_signature[word[0]]
        elif len(word) == 2:
            expected[i] = second_signature[word]
        elif len(word) == 3 and len(set(word)) == 1:
            letter = word[0]
            mean = mean_signature[letter]
            expected[i] = (mean**3 + 3.0 * mean * covariance[letter, letter]) / 6.0
        else:
            raise ValueError(f'Unsupported cubic control word {word}.')
    if not cir_words:
        return expected

    # Backward Chen recursion, integrating the same implicit CIR transition as the simulator.
    # An OU coordinate appearing once in a mixed word can be replaced by its mean path.
    suffixes = {word[i:] for word in cir_words.values() for i in range(len(word) + 1)}
    word_positions = {word: i for i, word in enumerate(sorted(suffixes, key=lambda word: (len(word), word)))}
    upper = max(np.sqrt(y0_cir), np.sqrt(theta_cir)) + 6.0 * eta_cir * np.sqrt(times[-1])
    root_grid = np.linspace(0.0, upper, state_grid_size)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    weights /= np.sqrt(np.pi)
    current_cir = root_grid[:, None] ** 2
    next_cir = cir_sqrt_implicit_step(current_cir, np.sqrt(2.0 * dt) * nodes[None, :], dt, kappa_cir, theta_cir, eta_cir)
    next_cir = np.minimum(np.broadcast_to(next_cir, (state_grid_size, quadrature_order)), upper**2)
    next_roots = np.sqrt(next_cir)
    cir_increment = next_cir - current_cir
    continuation_values = np.zeros((state_grid_size, len(word_positions)))
    continuation_values[:, word_positions[()]] = 1.0
    for ou_increment in ou_increments[::-1]:
        increments = (dt, *ou_increment, cir_increment)
        continuation = CubicSpline(root_grid, continuation_values)(next_roots)
        for word, position in word_positions.items():
            conditional_signature = continuation[:, :, position].copy()
            prefix_signature = 1.0
            for length, letter in enumerate(word, start=1):
                prefix_signature = prefix_signature * increments[letter] / length
                conditional_signature += prefix_signature * continuation[:, :, word_positions[word[length:]]]
            continuation_values[:, position] = np.einsum('ij,j->i', conditional_signature, weights)
    initial_values = CubicSpline(root_grid, continuation_values)(np.sqrt(y0_cir))
    for i, word in cir_words.items():
        expected[i] = initial_values[word_positions[word]]
    return expected

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

# Differentiate the normalized loss on the fixed CIR calibration sample.
def joint_calibration_gradient_with_cir_correlations(calibration_parameters, dimension_ell, rho_primary, driver_increments,
    independent_normals, objective_inputs, calibration_details, loss_factors):
    inputs = objective_inputs
    cached = calibration_details['gradient_inputs']
    ell = calibration_parameters[:dimension_ell]
    correlation_parameters = calibration_parameters[dimension_ell:]
    rho_b = calibration_details['rho_b']
    weights = np.linalg.solve(rho_primary, rho_b)
    q_rho = float(np.clip(rho_b @ weights, 0.0, 1.0))
    signature_path = inputs['truncated_signature_path']
    n_paths = signature_path.shape[1]
    use_controls = inputs.get('use_price_control_variates', False)
    price_gradients = {}
    for product in ('SPX', 'VIX', 'VXX'):
        key = product.lower()
        future_inputs = {'model_future': calibration_details['model_future_vix'], 'market_future': inputs['market_future_vix'],
            'deltas': inputs['deltas_vix']} if product == 'VIX' else {}
        price_gradient, future_gradient = calibration_block_loss_gradient(
            calibration_details['model_prices'][product], inputs['bid_' + key], inputs['ask_' + key],
            inputs['normalization_' + key], inputs['beta_' + key], **future_inputs)
        price_gradients[product] = loss_factors[product] * price_gradient
        if product == 'VIX':
            vix_future_gradient = loss_factors[product] * future_gradient

    def terminal_loss_sensitivity(terminal, product):
        key = product.lower()
        strikes = inputs['strikes_' + key]
        signs = inputs['cp_' + key]
        payoff_slopes = signs * (signs * (terminal[:, None] - strikes) > 0.0)
        discount = np.exp(-inputs['rate'] * inputs['maturity_' + key])
        return discount * (payoff_slopes @ price_gradients[product]) / n_paths

    # SPX log increments are a*dW_corr + b*dW_ind - v*dt/2 under the capped coupling.
    dt = inputs['dt']
    n_spx = inputs['N_spx']
    factor_drivers = driver_increments[:n_spx]
    independent_drivers = np.sqrt(dt) * independent_normals[:n_spx]
    raw_variance = cached['raw_variance_spx']
    sigma = np.sqrt(cached['variance_spx'])
    leverage_cap = inputs['leverage_cap']
    unsaturated = sigma < leverage_cap
    a = np.minimum(sigma, leverage_cap)
    chi = np.ones_like(sigma)
    np.divide(leverage_cap, sigma, out=chi, where=sigma > leverage_cap)
    b = sigma * np.sqrt(np.maximum(1.0 - q_rho * chi**2, 0.0))
    correlated_driver = factor_drivers @ weights
    terminal_spx = cached['terminal_spx']
    spx_sensitivity = terminal_loss_sensitivity(terminal_spx, 'SPX')
    if use_controls:
        spx_sensitivity -= (price_gradients['SPX'] @ inputs['deltas_spx']) / n_paths
    stock_log_sensitivity = spx_sensitivity * terminal_spx

    da_dv = np.zeros_like(sigma)
    np.divide(0.5, sigma, out=da_dv, where=(sigma > 0.0) & unsaturated)
    db_dv = np.zeros_like(b)
    np.divide(1.0 - q_rho * unsaturated, 2.0 * b, out=db_dv, where=b > 0.0)
    if inputs.get('variance_treatment', 'positive_part') == 'smooth':
        shifted = raw_variance - 1e-12
        variance_slope = 0.5 * (1.0 + shifted / np.hypot(shifted, 2.0 * inputs['variance_softness']))
    else:
        variance_slope = raw_variance > 0.0
    step_sensitivity = (da_dv * correlated_driver + db_dv * independent_drivers - 0.5 * dt) * variance_slope * stock_log_sensitivity
    features = inputs['variance_signature_features']
    if isinstance(features, dict):
        selected = features['signature_path'][:n_spx].reshape(-1, dimension_ell)
        signature_products = selected.T @ (selected * step_sensitivity.reshape(-1, 1))
        pair_moments = signature_products[tuple(features['pair_positions'].T)]
        feature_sensitivity = features['pair_to_variance_features'] @ pair_moments
    else:
        feature_sensitivity = np.einsum('tp,tpi->i', step_sensitivity, features[:n_spx], optimize=True)
    ell_gradient = reduced_variance_gradient(ell, inputs['variance_pair_positions'], inputs['variance_shuffle_coordinates'], inputs['window'] * feature_sensitivity)

    radius = np.sqrt(1.0 + correlation_parameters @ correlation_parameters)
    correlation_jacobian = np.linalg.cholesky(rho_primary) @ (np.eye(correlation_parameters.size) / radius - np.outer(correlation_parameters, correlation_parameters) / radius**3)
    weights_jacobian = np.linalg.solve(rho_primary, correlation_jacobian)
    q_gradient = 2.0 * weights @ correlation_jacobian
    factor_sensitivity = np.einsum('p,tp,tpi->i', stock_log_sensitivity, a, factor_drivers, optimize=True)
    independent_coefficient = np.zeros_like(b)
    np.divide(a**2, 2.0 * b, out=independent_coefficient, where=b > 0.0)
    independent_sensitivity = np.einsum('p,tp,tp->', stock_log_sensitivity, independent_coefficient, independent_drivers, optimize=True)
    correlation_gradient = factor_sensitivity @ weights_jacobian - independent_sensitivity * q_gradient

    # VIX options use the absolute proxy, while the control and future use the signed proxy.
    terminal_signature_vix = signature_path[inputs['N_vix']]
    vix_sensitivity = terminal_loss_sensitivity(cached['terminal_vix'], 'VIX') * np.sign(cached['raw_terminal_vix'])
    if use_controls:
        control_sensitivity = price_gradients['VIX'] @ inputs['deltas_vix']
        vix_sensitivity -= control_sensitivity / n_paths
        ell_gradient += 100.0 * control_sensitivity * inputs['vix_control_loadings']
    ell_gradient += 100.0 * (vix_sensitivity @ terminal_signature_vix)
    future_loadings = inputs['vix_future_propagator'] @ signature_path[0, 0]
    ell_gradient += 100.0 * vix_future_gradient * future_loadings

    # Differentiate the VXX log roll factors with frozen futures contributing zero slope.
    terminal_vxx = cached['terminal_vxx']
    vxx_log_sensitivity = terminal_loss_sensitivity(terminal_vxx, 'VXX') * terminal_vxx
    futures = cached['future_paths']
    for indices, previous, front_expiry, second_expiry, front_propagators, second_propagators, _, _ in inputs['vxx_roll_data']:
        times = inputs['time_steps_vxx'][indices]
        front = futures[front_expiry]['stopped'][indices]
        second = futures[second_expiry]['stopped'][indices]
        front_weight = (front_expiry - times)[:, None]
        second_weight = (times - previous)[:, None]
        basket = front_weight * front + second_weight * second
        quadrature_weights = np.zeros(times.size)
        half_steps = 0.5 * np.diff(times)
        quadrature_weights[:-1] += half_steps
        quadrature_weights[1:] += half_steps
        integral_sensitivity = quadrature_weights[:, None] * (front_expiry - previous) / basket**2
        sensitivities = ((front_expiry, front_propagators, front_weight, integral_sensitivity * second),
            (second_expiry, second_propagators, second_weight, -integral_sensitivity * front))
        segment_signature = signature_path[indices]
        for maturity, propagators, roll_weight, sensitivity in sensitivities:
            sensitivity[-1] += roll_weight[-1] / basket[-1]
            sensitivity[0] -= roll_weight[0] / basket[0]
            sensitivity *= futures[maturity]['alive'][indices] * vxx_log_sensitivity
            signature_sensitivity = np.einsum('tp,tpj->tj', sensitivity, segment_signature, optimize=True)
            ell_gradient += 100.0 * np.einsum('tij,tj->i', propagators, signature_sensitivity, optimize=True)
    return np.r_[ell_gradient, correlation_gradient]

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

# Compute selected signatures and store the compact variance features in memory.
def compute_cubic_signature_features(paths, selected_indices, reduced_variance_model):
    n_times, npaths, alphabet_size = paths.shape
    if alphabet_size != 4:
        raise ValueError('Expected exactly two OU factors, one CIR and time.')
    level = max((len(from_num_to_word(int(i), 3)) for i in selected_indices))
    k = len(selected_indices)
    pairs = reduced_variance_model['pair_positions']
    transform = reduced_variance_model['pair_to_variance_features']
    signatures = np.empty((n_times, npaths, k))
    features = {'signature_path': signatures, 'pair_positions': pairs, 'pair_to_variance_features': transform}

    nonempty = selected_indices > 0
    if iisignature is None:
        if level not in (2, 3):
            raise ValueError('The NumPy signature implementation supports levels two and three.')
        full_dimension = from_level_to_num(level, 3) + 1
        state = np.zeros((npaths, full_dimension))
        state[:, 0] = 1.0
        for step in range(n_times):
            if step:
                dx = paths[step] - paths[step - 1]
                if level == 3:
                    third = state[:, 5:21].reshape(npaths, 4, 4, 1) * dx[:, None, None, :] + 0.5 * state[:, 1:5, None, None] * dx[:, None, :, None] * dx[:, None, None, :] + dx[:, :, None, None] * dx[:, None, :, None] * dx[:, None, None, :] / 6.0
                    state[:, 21:85] += third.reshape(npaths, 64)
                second = state[:, 1:5, None] * dx[:, None, :] + 0.5 * dx[:, :, None] * dx[:, None, :]
                state[:, 5:21] += second.reshape(npaths, 16)
                state[:, 1:5] += dx
            signatures[step] = state[:, selected_indices]
    else:
        for p in range(npaths):
            sig = np.zeros((n_times, k))
            sig[:, 0] = 1
            sig[1:, nonempty] = iisignature.sig(paths[:, p, :], level, 2)[:, selected_indices[nonempty] - 1]
            signatures[:, p] = sig
    return (signatures, features)

# Build the quadratic variance constraints at selected path-time points.
def sampled_variance_matrices(features, pair_positions, shuffle_coordinates, window, indices, dimension_ell):
    indices = np.asarray(indices, dtype=int)
    if isinstance(features, dict):
        signatures = features['signature_path'].reshape(-1, dimension_ell)[indices]
        left, right = features['pair_positions'].T
        selected_features = (signatures[:, left] * signatures[:, right]) @ features['pair_to_variance_features'].T
    else:
        selected_features = np.asarray(features).reshape(-1, features.shape[-1])[indices]
    pair_values = window * (selected_features @ shuffle_coordinates.T)
    h = np.zeros((len(indices), dimension_ell, dimension_ell))
    left, right = pair_positions.T
    h[:, left, right] = pair_values
    h[:, right, left] = pair_values
    return h

# Run the supplied SLSQP constraint-exchange calibration.
def calibrate_cubic_with_active_constraints(
    objective, initial_parameters, dimension_ell, features, signatures, pair_positions, shuffle_coordinates, window, minimum_variance=1e-06, minimum_vix=1e-08,
    maxiter=240, stage_iterations=40, max_stages=8, ftol=3e-05, progress=True, initial_future_loadings=None, minimum_initial_future=0.0,
    loss_reporter=None,
):
    from scipy.optimize import minimize, OptimizeResult
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
        previous_best_loss = best_loss
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
        if loss_reporter is not None and best_loss < previous_best_loss:
            loss_reporter(best)
        return (ok, rv, yv, loss)
    if progress:
        print(f'Starting loss = {best_loss:.8e}', flush=True)
    for stage in range(max_stages):
        if total_iterations >= maxiter:
            break
        _, variance, vix = full_check(current)
        add_worst(variance, active_variance)
        add_worst(vix, active_vix, 24)
        h = sampled_variance_matrices(features, pair_positions, shuffle_coordinates, window, sorted(active_variance), k)
        linear = flat_signatures[sorted(active_vix)].copy()

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
                record_candidate(q * scale)
                if progress:
                    print(f'Iteration {total_iterations + count[0]:4d} | best feasible loss = {best_loss:.8e}', flush=True)
        q0 = current / scale
        result = minimize(lambda q: float(objective(q * scale)), q0, method='SLSQP', bounds=list(zip(q0 - 0.3, q0 + 0.3)), constraints=({'type': 'ineq', 'fun': inequalities, 'jac': inequality_jacobian},), callback=callback, options={'maxiter': min(stage_iterations, maxiter - total_iterations), 'ftol': ftol, 'eps': 1e-06, 'disp': False})
        proposal = result.x * scale
        ok, variance, vix, loss = record_candidate(proposal)
        total_iterations += int(result.nit)
        history.append({'stage': stage + 1, 'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit), 'sample_admissible': ok, 'minimum_variance': float(np.min(variance)), 'minimum_vix_fraction': float(np.min(vix)), 'best_feasible_loss': best_loss, 'active_variance': len(active_variance), 'active_vix': len(active_vix)})
        if progress:
            print(f'End of stage {stage + 1} | best feasible loss = {best_loss:.8e}', flush=True)
        if ok and result.success and (np.max(np.abs(result.x - q0)) < 0.299):
            converged = np.allclose(best, proposal, rtol=1e-07, atol=1e-09)
            break
        add_worst(variance, active_variance, 64)
        add_worst(vix, active_vix, 32)
        current = best.copy()
    ok, variance, vix = full_check(best)
    if not ok:
        raise RuntimeError('Internal error: selected checkpoint is not sampled-admissible.')
    return OptimizeResult(x=best, fun=best_loss, success=converged, nit=total_iterations, message='Converged with full sampled admissibility check.' if converged else 'Iteration/stage budget reached; returning best fully checked admissible candidate.', stage_history=history, warm_start_weight=warm_weight, minimum_variance=float(np.min(variance)), minimum_vix=float(np.min(vix)), parameter_scale=scale)

# Normalize the three product losses and return their weighted contributions.
def normalized_joint_loss(block_losses, *, lambda_spx, lambda_vix, lambda_vxx, loss_scales, objective_scale):
    contributions = {'SPX': objective_scale * lambda_spx * float(block_losses['SPX']) / loss_scales['SPX'], 'VIX': objective_scale * lambda_vix * float(block_losses['VIX']) / loss_scales['VIX'], 'VXX': objective_scale * lambda_vxx * float(block_losses['VXX']) / loss_scales['VXX']}
    return (float(sum(contributions.values())), contributions)

def initial_future_constraint(parameters, *, initial_future_loadings, minimum_initial_future):
    return initial_future_loadings @ parameters[:initial_future_loadings.shape[1]] - minimum_initial_future

# Keep as much of the candidate as possible along the segment to a feasible anchor.
def feasible_point_toward_candidate(anchor, candidate, is_feasible):
    if is_feasible(candidate):
        return candidate
    direction = candidate - anchor
    lower, upper = 0.0, 1.0
    feasible_parameters = anchor
    for _ in range(16):
        weight = 0.5 * (lower + upper)
        parameters = anchor + weight * direction
        if is_feasible(parameters):
            lower = weight
            feasible_parameters = parameters
        else:
            upper = weight
    return feasible_parameters

# Run the three SLSQP passes with smooth constraints at the critical sample points.
def calibrate_cir(*, initial_parameters, initial_block_losses, objective_with_components, objective_gradient, normalized_joint_loss,
    variance_constraint_inputs, constraint_scale, initial_future_loadings, minimum_initial_future, parameter_scales, rho_primary,
    optimizer_runs, correlation_restart_shrink=0.9):
    from scipy.optimize import minimize

    if len(optimizer_runs) != 3:
        raise ValueError('Expected three optimizer passes: descent, correlation restart and refinement.')
    dimension_ell = initial_future_loadings.shape[1]
    initial_loss, initial_contributions = normalized_joint_loss(initial_block_losses)
    objective_scale = max(initial_loss, 1.0)
    variance_features = variance_constraint_inputs['variance_signature_features']
    variance_steps = variance_constraint_inputs['N']
    variance_pairs = variance_constraint_inputs['variance_pair_positions']
    variance_coordinates = variance_constraint_inputs['variance_shuffle_coordinates']
    window = variance_constraint_inputs['window']
    minimum_variance = variance_constraint_inputs['minimum_variance']
    # Give the solver room for its numerical feasibility tolerance.
    solver_minimum_variance = minimum_variance + 1e-8
    future_constraint = lambda parameters: initial_future_constraint(parameters, initial_future_loadings=initial_future_loadings,
        minimum_initial_future=minimum_initial_future)
    future_jacobian = np.zeros((initial_future_loadings.shape[0], initial_parameters.size))
    future_jacobian[:, :dimension_ell] = initial_future_loadings * parameter_scales[:dimension_ell]
    constraint_ell = None
    constraint_cache = None
    active_points = set()
    rejected_parameters = None
    rejected_loss = np.inf
    progress = {'parameters': initial_parameters.copy(), 'loss': initial_loss, 'block_losses': initial_block_losses,
        'best_parameters': initial_parameters.copy(), 'best_loss': initial_loss, 'evaluations': 0}

    def parameters_from_scaled(scaled_parameters):
        return initial_parameters + parameter_scales * scaled_parameters

    def sample_variance(parameters):
        nonlocal constraint_ell, constraint_cache
        ell = parameters[:dimension_ell]
        if not np.array_equal(ell, constraint_ell):
            with np.errstate(over='ignore', invalid='ignore'):
                product = reduced_shuffle_product_ell(ell, variance_pairs, variance_coordinates)
                constraint_cache = variance_path(product, variance_steps, variance_features, window)
            constraint_ell = ell.copy()
        return constraint_cache

    def sample_feasible(parameters):
        future_margins = future_constraint(parameters)
        variance = sample_variance(parameters)
        return (np.all(np.isfinite(variance)) and np.min(variance) >= minimum_variance
            and np.all(np.isfinite(future_margins)) and np.all(future_margins >= 0.0))

    if not sample_feasible(initial_parameters):
        raise ValueError('The initial parameters violate the calibration-sample variance or futures constraints.')

    print(f"Initial loss | joint = {initial_loss:.8e} | SPX = {initial_contributions['SPX']:.8e} | VIX = {initial_contributions['VIX']:.8e} | VXX = {initial_contributions['VXX']:.8e}", flush=True)

    def critical_points(variance):
        # Screen across trajectories so neighboring times of one path do not fill the working set.
        paths = np.arange(variance.shape[1])
        worst_times = np.argmin(variance, axis=0)
        path_minima = variance[worst_times, paths]
        count = min(24, paths.size)
        worst_paths = np.argpartition(path_minima, count - 1)[:count]
        return set(worst_times[worst_paths] * paths.size + worst_paths)

    def monitored_objective(parameters):
        nonlocal rejected_parameters, rejected_loss
        if np.array_equal(parameters, progress['parameters']):
            return progress['loss']
        raw_loss, details = objective_with_components(parameters)
        progress['evaluations'] += 1
        progress['parameters'] = parameters.copy()
        progress['block_losses'] = details
        if not np.isfinite(raw_loss) or raw_loss >= 1e12 or any(not np.isfinite(details[p]) or details[p] >= 1e12 for p in ('SPX', 'VIX', 'VXX')):
            progress['loss'] = 1e12
            return progress['loss']
        loss, contributions = normalized_joint_loss(details)
        progress['loss'] = loss
        if loss < progress['best_loss']:
            if sample_feasible(parameters):
                progress['best_loss'] = loss
                progress['best_parameters'] = parameters.copy()
                print(f"Evaluation {progress['evaluations']:5d} | joint = {loss:.8e} | SPX = {contributions['SPX']:.8e} | VIX = {contributions['VIX']:.8e} | VXX = {contributions['VXX']:.8e}", flush=True)
            elif loss < rejected_loss:
                rejected_parameters = parameters.copy()
                rejected_loss = loss
        return loss

    def scaled_objective(scaled_parameters):
        return monitored_objective(parameters_from_scaled(scaled_parameters)) / objective_scale

    def scaled_objective_jacobian(scaled_parameters):
        parameters = parameters_from_scaled(scaled_parameters)
        if monitored_objective(parameters) >= 1e12:
            return np.zeros_like(scaled_parameters)
        gradient = objective_gradient(parameters, calibration_details=progress['block_losses'])
        return gradient * parameter_scales / objective_scale

    def solve_stage(starting_parameters, options, trust_radius):
        active_points.update(critical_points(sample_variance(starting_parameters)))
        matrices = sampled_variance_matrices(variance_features, variance_pairs, variance_coordinates, window,
            sorted(active_points), dimension_ell)

        def variance_margins(scaled_parameters):
            ell = parameters_from_scaled(scaled_parameters)[:dimension_ell]
            return (np.einsum('i,mij,j->m', ell, matrices, ell) - solver_minimum_variance) / constraint_scale

        def variance_jacobian(scaled_parameters):
            ell = parameters_from_scaled(scaled_parameters)[:dimension_ell]
            jacobian = np.zeros((len(matrices), initial_parameters.size))
            jacobian[:, :dimension_ell] = 2.0 * (matrices @ ell) * parameter_scales[:dimension_ell] / constraint_scale
            return jacobian

        starting_scaled = (starting_parameters - initial_parameters) / parameter_scales
        constraints = (
            {'type': 'ineq', 'fun': variance_margins, 'jac': variance_jacobian},
            {'type': 'ineq', 'fun': lambda q: future_constraint(parameters_from_scaled(q)), 'jac': lambda q: future_jacobian},
        )
        iteration = 0

        def report_iteration(_):
            nonlocal iteration
            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                print(f"SLSQP stage iteration {iteration:3d} | last trial loss = {progress['loss']:.8e} | best feasible loss = {progress['best_loss']:.8e}", flush=True)

        result = minimize(scaled_objective, starting_scaled, method='SLSQP', jac=scaled_objective_jacobian,
            constraints=constraints, bounds=list(zip(starting_scaled - trust_radius, starting_scaled + trust_radius)),
            callback=report_iteration, options=options)
        reached_trust_boundary = np.max(np.abs(result.x - starting_scaled)) >= 0.99 * trust_radius
        return result, reached_trust_boundary

    for run, options in enumerate(optimizer_runs, start=1):
        starting_parameters = progress['best_parameters'].copy()
        if run == 2:
            rho_b = rho_b_from_parameters(starting_parameters[dimension_ell:], rho_primary)
            starting_parameters[dimension_ell:] = parameters_from_rho_b(correlation_restart_shrink * rho_b, rho_primary)
        starting_loss = monitored_objective(starting_parameters)
        print(f"SLSQP run {run}/{len(optimizer_runs)} | starting joint loss = {starting_loss:.8e}", flush=True)
        iterations_left = options['maxiter']
        trust_radius = 0.3
        while iterations_left > 0:
            previous_best = progress['best_loss']
            rejected_parameters, rejected_loss = None, np.inf
            stage_options = {**options, 'maxiter': min(40, iterations_left)}
            result, reached_trust_boundary = solve_stage(starting_parameters, stage_options, trust_radius)
            iterations_left -= max(1, result.nit)
            trial_parameters = parameters_from_scaled(result.x)
            monitored_objective(trial_parameters)

            new_points = set()
            if rejected_loss < progress['best_loss']:
                # Include constraints at a better trial rejected by the full-sample check.
                variance = sample_variance(rejected_parameters)
                if np.all(np.isfinite(variance)) and np.min(variance) < minimum_variance:
                    new_points = critical_points(variance) - active_points
                    active_points.update(new_points)
                repaired = feasible_point_toward_candidate(progress['best_parameters'], rejected_parameters, sample_feasible)
                monitored_objective(repaired)

            improved = progress['best_loss'] < previous_best
            if not new_points:
                if result.success and not reached_trust_boundary and sample_feasible(trial_parameters):
                    break
                if not improved:
                    if trust_radius <= 0.0375 or iterations_left <= 0:
                        break
                    trust_radius *= 0.5
            starting_parameters = progress['best_parameters']
        print(f"End of run {run} | best feasible joint loss = {progress['best_loss']:.8e}", flush=True)
    return result, progress

# Evaluate the normalized objective with the original invalid-price penalty.
def normalized_cir_objective(parameters, *, objective_with_components, normalized_joint_loss):
    raw_loss, block_losses = objective_with_components(parameters)
    if not np.isfinite(raw_loss) or raw_loss >= 1000000000000.0:
        return 1000000000000.0
    return normalized_joint_loss(block_losses)[0]

# Build the quadratic variance map from reduced shuffle coefficients.
def variance_quadratic_matrix(ell_shuffle, features, dimension, window):
    weights = window * (features['pair_to_variance_features'].T @ ell_shuffle)
    matrix = np.zeros((dimension, dimension))
    np.add.at(matrix, tuple(features['pair_positions'].T), weights)
    return matrix

# Build the fixed quadratic variance map for out-of-sample pricing.
def fixed_ell_variance_matrix(ell, reduced_model, window):
    ell = np.asarray(ell, dtype=float)
    product = reduced_shuffle_product_ell(ell, reduced_model['pair_positions'], reduced_model['shuffle_product_coordinates'])
    return variance_quadratic_matrix(product, reduced_model, ell.size, window)

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

