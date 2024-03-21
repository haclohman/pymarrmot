
def evap_23(p1, p2, S, Smax, Ep, dt):
    # evap_23 combines evap_5 (evaporation) and evap_6 (transpiration)

    # Flux function
    # Description: Transpiration from vegetation at the potential rate if
    # storage is above field capacity and scaled by relative storage if not (similar to evap_6),
    # addition of Evaporation from bare soil scaled by relative storage (similar to evap_5)
    # Constraints:  Ea <= Ep
    #               Ea <= S/dt
    # @(Inputs):    p1   - fraction vegetated area [-] (0...1)
    #               p2   - field capacity coefficient[-]
    #               S    - current storage [mm]
    #               Smax - maximum storage [mm]
    #               Ep   - potential evaporation [mm]
    #               dt   - time step [h]
    # @(Output):    out  - the water lost to evapotranspiration [mm]

    # Perform calculations here
    # ... (the specific calculations from the MATLAB code)

    # Return the result
    return out
