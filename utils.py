def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)
