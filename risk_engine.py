def risk_level(p):
    if p > 0.7:
        return "HIGH RISK"
    elif p > 0.4:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"