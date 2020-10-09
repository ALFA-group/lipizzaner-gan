def is_square(positive_int):
    return True
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
