# Python version of the phenology_1 Matlab function


    Corrects Ep for phenology effects.
    
    Arguments:
    T -- Current temperature (°C)
    p1 -- Temperature threshold where evaporation stops (°C)
    p2 -- Temperature threshold above which corrected Ep = Ep (°C)
    Ep -- Current potential evapotranspiration (mm/day)
    
    Returns:
    out -- Corrected potential evapotranspiration (mm/day)
    