
def area_1(p1, p2, S, Smin, Smax, r=0.01, e=5.00):
    out = min(1, p1 * (max(0, S - Smin) / (Smax - Smin)) ** p2) * (1 - smoothThreshold(r, e))
    return out
