
def evap_3(p1, S, Smax, Ep, dt):
    # Flux function
    # Description: Evaporation based on scaled current water storage and wilting point
    # Constraints: f <= Ep, f < (S - p1 * Smax) / ((1 - p1) * Smax) * Ep
    f = min(Ep, (S - p1 * Smax) / ((1 - p1) * Smax) * Ep)
    return f
