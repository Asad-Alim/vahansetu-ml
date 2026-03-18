# constants.py
# Single source of truth for all cost weights.
# Import from here in alns_env.py, alns_operators.py, and presolve.py.

W_TRAVEL_TIME = 1.0
W_LATENESS    = 25.0
W_CARBON      = 0.05
W_FUEL        = 0.1
W_INFEASIBLE  = 1e5