from math import comb, perm


def kRDM(Q, k):
    one = (2**Q)*comb(Q, k)
    two = (2**Q)*(comb(comb(Q, k), 2)+comb(Q, k))
    thr = perm(Q, k)**2
    return one+two+thr


print(f"{kRDM(10,4):.3e}")
