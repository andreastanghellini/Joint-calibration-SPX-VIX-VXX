# Copy to market_data_private_CIR.py and fill in your own market inputs.
# Times and futures maturities are year fractions on an ACT/365 basis.
from datetime import date
import numpy as np


valuation_date = None
previous_vix_future_expiry = None
futures_maturities = np.array([], dtype=float)
T_spx = None
T_vix = None
T_vxx = None
spx_spot = None
vix_spot = None
vxx_spot = None
r = None

# SPX: use matching strike and quote arrays.
strikes_spx = np.array([], dtype=float)
bid_spx = np.array([], dtype=float)
ask_spx = np.array([], dtype=float)
put_bid_spx = np.array([], dtype=float)
put_ask_spx = np.array([], dtype=float)

# VIX: use matching strike and quote arrays.
strikes_vix = np.array([], dtype=float)
bid_vix = np.array([], dtype=float)
ask_vix = np.array([], dtype=float)
put_bid_vix = np.array([], dtype=float)
put_ask_vix = np.array([], dtype=float)

# VXX: use matching strike and quote arrays.
strikes_vxx = np.array([], dtype=float)
bid_vxx = np.array([], dtype=float)
ask_vxx = np.array([], dtype=float)
put_bid_vxx = np.array([], dtype=float)
put_ask_vxx = np.array([], dtype=float)
