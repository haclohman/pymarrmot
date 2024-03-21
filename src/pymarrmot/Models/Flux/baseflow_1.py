
def baseflow_1(p1, S):
    # Flux function
    # Description: Outflow from a linear reservoir
    # Constraints: -
    # @(Inputs): p1 - time scale parameter [d-1]
    #            S  - current storage [mm]

    out = p1 * S
    return out
