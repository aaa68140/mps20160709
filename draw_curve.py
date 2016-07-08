
import numpy as np
from matplotlib import pyplot as plt


def e2(e1):
    e2 = np.array([-1/e1[0], -1/e1[1]])
    return e2 / np.linalg.norm(e2)


def e1(s):
    return np.array([1.0, 2.0 * (s - 1.0)])


def draw_curve(p0, e1, ds):
    curve = [p0, ]
    s = 0
    while s < 5:
        curve.append(curve[-1] + e1(s + ds))
        s += ds
    return np.array(curve)


def rotate(theta):
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])


def curve1(ds=0.01):
    return draw_curve(np.array([100.0, 100.0]), e1, ds)


def curve2(ds=0.01):
    return rotate(0.5 * np.pi).dot(curve1(ds).T).T + np.array([-100, -100])


def derive_k(curve, ds=0.01):
    ks = []
    for i in range(len(curve) - 2):
        d_d1 = (curve[i + 2] - 2 * curve[i + 1] + curve[i]) / np.power(ds, 2)
        if d_d1[0] != 0:
            ks.append(d_d1[0]/(e2(curve[i + 1] - curve[i]) / ds)[0])
        else:
            ks.append(d_d1[1]/(e2(curve[i + 1] - curve[i]) / ds)[1])
    return np.array(ks)


# if __name__ == '__main__':
#     cv = curve1()
#     cv2 = curve2()
#
#     ks1 = derive_k(cv)
#     ks2 = derive_k(cv2)
#
#     #plt.scatter([c[0] for c in curve1], [c[1] for c in curve1])
#     #plt.scatter([c[0] for c in curve2], [c[1] for c in curve2])
#
#     plt.scatter(ks1, ks2)
#     plt.show()
