def percolation_6(p1, p2, S, dt):
    # Threshold-based percolation from a store that can reach negative values
    # Constraints: f <= S/dt
    
    #bug fix: 18July2024 - SAS - autoconversion failure
    # MatLab: out = min(S/dt,p1.*min(1,max(0,S)./p2))
    # Autoconversion: out = p1 * (S/dt > p1) + p2 * (S/dt <= p1)
    
    out = min(S/dt, p1 * min(1, max(0, S) / p2))
        
    return out