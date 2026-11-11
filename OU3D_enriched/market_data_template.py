"""Fill in the spot levels, annual rate in decimal form, and option quotes.

For each asset, enter the bid and ask quotes in the same order as the
strikes. The bid and ask arrays without a put_ prefix contain call quotes.
"""

import numpy as np

spx_spot = None
vix_spot = None
vxx_spot = None
r = None

strikes_spx = np.array([], dtype=float)
bid_spx = np.array([], dtype=float)
ask_spx = np.array([], dtype=float)
put_bid_spx = np.array([], dtype=float)
put_ask_spx = np.array([], dtype=float)

strikes_vix = np.array([], dtype=float)
bid_vix = np.array([], dtype=float)
ask_vix = np.array([], dtype=float)
put_bid_vix = np.array([], dtype=float)
put_ask_vix = np.array([], dtype=float)

strikes_vxx = np.array([], dtype=float)
bid_vxx = np.array([], dtype=float)
ask_vxx = np.array([], dtype=float)
