
# Python code equivalent to the Matlab percolation_6 function

# Function to convert the Matlab percolation_6 function to Python
def percolation_6(p1, p2, S, dt):
    # Threshold-based percolation from a store that can reach negative values
    # Constraints: f <= S/dt
    out = p1 * (S/dt > p1) + p2 * (S/dt <= p1)
    return out
