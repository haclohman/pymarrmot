
def evap_18(p1, p2, p3, S, Ep):
    # Flux function
    # Description: Exponentially declining evaporation from deficit store
    # Constraints: -
    # @(Inputs): p1 - linear scaling parameter [-]
    #            p2 - linear scaling parameter [-]
    #            p3 - storage scaling parameter [mm]
    #            S - current storage [mm]
    #            Ep - potential evapotranspiration rate [mm/d]
    
    out = p1 * np.exp(-1 * p2 * S / p3) * Ep
    return out
