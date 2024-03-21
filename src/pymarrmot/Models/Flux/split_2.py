
def split_2(p1, In):
    # Flux function
    # ------------------
    # Description:  Split flow (returns flux [mm/d]), counterpart to split_1
    # Constraints:  -
    # @(Inputs):    p1   - fraction of flux to be diverted [-]
    #               In   - incoming flux [mm/d]
    
    out = (1-p1) * In
    return out
