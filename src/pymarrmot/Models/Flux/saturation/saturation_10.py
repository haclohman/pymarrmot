
def saturation_10(p1, p2, p3, S, In):
    # Flux function
    # Description: Saturation excess flow from a store with different degrees of saturation (min-max exponential variant)
    # Constraints: -
    # Inputs: p1 - maximum contributing fraction area [-]
    #         p2 - minimum contributing fraction area [-]
    #         p3 - exponential scaling parameter [-]
    #         S - current storage [mm]
    #         In - incoming flux [mm/d]
    out = min(p1, p2 + p2 * np.exp(p3 * S)) * In
    return out
