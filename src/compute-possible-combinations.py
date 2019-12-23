def get_possibilities(size, index):
    if size==0:
        return 0
    elif index==10:
        return get_possibilities(size-1, 0)
    else:
        return index + get_possibilities(size, index+1)


print(get_possibilities(1, 0))




