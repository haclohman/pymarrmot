
def interception_1(In, S, Smax, *varargin):
    # Flux function
    # Description: Interception excess when maximum capacity is reached
    # Constraints: -
    # @(Inputs): In - incoming flux [mm/d]
    #            S  - current storage [mm]
    #            Smax - maximum storage [mm]
    #            varargin(1) - smoothing variable r (default 0.01)
    #            varargin(2) - smoothing variable e (default 5.00)
    
    if len(varargin) == 0:
        r = 0.01  # default value for smoothing variable r
        e = 5.00  # default value for smoothing variable e
    elif len(varargin) == 1:
        r = varargin[0]
        e = 5.00  # default value for smoothing variable e
    elif len(varargin) == 2:
        r = varargin[0]
        e = varargin[1]
    
    # write the logic for the function here
    # ...
    pass  # placeholder for the logic
    
    # replace "pass" with the actual logic for the function

    return out  # replace "out" with the actual output variable name
