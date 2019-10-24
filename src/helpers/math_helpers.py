import numpy as np
from scipy.stats import iqr


def is_square(positive_int):
    if positive_int == 1:
        return True

    x = positive_int // 2
    seen = {x}
    while x * x != positive_int:
        x = (x + (positive_int // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True

# data1 list of numbers
def get_basic_stats(data1):
    stats = {}
    num = np.array(data1)

    stats['mean'] = num.mean()
    stats['iqr_range'] = iqr(num)
    stats['median'] = np.median(num)
    stats['max'] = num.max()
    stats['min'] = num.min()
    stats['norm_std'] = num.std() / stats['mean']
    stats['count'] = len(data1)
    return stats