# Copy this file to market_data_private.py and enter your own market data.
# Fill all inputs before running the calibration or out-of-sample script.
import numpy as np


spx_spot = None
vix_spot = None
vxx_spot = None
r = None

# SPX: arrays must have matching lengths.
strikes_spx = np.array([], dtype=float)
bid_spx = np.array([], dtype=float)
ask_spx = np.array([], dtype=float)
put_bid_spx = np.array([], dtype=float)
put_ask_spx = np.array([], dtype=float)

# VIX: arrays must have matching lengths.
strikes_vix = np.array([], dtype=float)
bid_vix = np.array([], dtype=float)
ask_vix = np.array([], dtype=float)
put_bid_vix = np.array([], dtype=float)
put_ask_vix = np.array([], dtype=float)

# VXX: arrays must have matching lengths.
strikes_vxx = np.array([], dtype=float)
bid_vxx = np.array([], dtype=float)
ask_vxx = np.array([], dtype=float)
put_bid_vxx = np.array([], dtype=float)
put_ask_vxx = np.array([], dtype=float)
