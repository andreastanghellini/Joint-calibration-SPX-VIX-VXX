"""Synthetic SPX, VIX and VXX inputs for the OU-CIR scripts.

The quotes are the compact OU3D synthetic panel, with the calendar fields
required by the CIR calibration and validation scripts. Prices are in the
underlying's price units and the annual interest rate is in decimal form.
"""
from datetime import date

import numpy as np

valuation_date = date(2026, 2, 24)
previous_vix_future_expiry = date(2026, 2, 18)
futures_maturities = np.array([22, 50, 84, 113], dtype=float) / 365
T_spx = 24 / 365
T_vix = 22 / 365
T_vxx = 24 / 365

spx_spot = 6890.0
vix_spot = 18.3159828
market_future_vix = 20.28452
vxx_spot = 28.5
r = 0.0367

strikes_spx = np.array([
    5600, 5800, 6000, 6200, 6350, 6500, 6650, 6750, 6850, 6900, 7000, 7100, 7200, 7350,
    7575,
], dtype=float)
bid_spx = np.array([
    1303.93, 1106.18, 909.71, 715.67, 572.67, 433.77, 301.96, 220.26, 146.12, 113, 57.57,
    20.91, 5.16, 0.69, 0.03,
], dtype=float)
ask_spx = np.array([
    1307.38, 1109.39, 912.69, 718.31, 575.57, 436.92, 305.01, 223.29, 149.3, 116.16, 60.37,
    22.81, 6.04, 1.13, 0.23,
], dtype=float)
put_bid_spx = np.array([
    0.43264828607175, 2.2006000105744, 5.2485517350769, 10.726503459579, 17.364967252956,
    28.103431046333, 45.93189483971, 63.990870701962, 89.609846564213, 106.36933449534,
    150.69831035759, 213.79728621984, 297.80626208209, 442.97472587547, 666.77242156553,
], dtype=float)
put_ask_spx = np.array([
    3.8826482860718, 5.4106000105744, 8.228551735077, 13.366503459579, 20.264967252956,
    31.253431046333, 48.98189483971, 67.020870701962, 92.789846564213, 109.52933449534,
    153.49831035759, 215.69728621984, 298.68626208209, 443.41472587547, 666.97242156553,
], dtype=float)

strikes_vix = np.array([
    16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 23, 24.5, 26, 28,
], dtype=float)
bid_vix = np.array([
    3.86, 3.22, 2.72, 2.31, 1.99, 1.72, 1.39, 1.13, 0.93, 0.73,
], dtype=float)
ask_vix = np.array([
    4, 3.32, 2.8, 2.39, 2.05, 1.78, 1.47, 1.21, 1.01, 0.81,
], dtype=float)

strikes_vxx = np.array([
    25, 26.5, 27.5, 28, 29, 30.5, 32.5, 33.5, 35, 37,
], dtype=float)
bid_vxx = np.array([
    3.76, 2.76, 2.06, 2.11, 1.82, 1.35, 0.87, 0.83, 0.8, 0.45,
], dtype=float)
ask_vxx = np.array([
    3.9, 3.03, 2.82, 2.39, 2.02, 1.75, 1.51, 1.26, 0.96, 0.97,
], dtype=float)
