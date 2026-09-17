import numpy as np
from scipy.optimize import brentq


MU_SUN = 1.32712440018e11  # km^3/s^2


def stumpff_C(z):
    if z > 1e-8:
        sz = np.sqrt(z)
        return (1 - np.cos(sz)) / z

    elif z < -1e-8:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1) / (-z)

    else:
        return 0.5


def stumpff_S(z):
    if z > 1e-8:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3

    elif z < -1e-8:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3

    else:
        return 1.0 / 6.0


def find_z_bracket(equation, z_min=-100, z_max=100, n=10000, call_count = 1):

    if call_count == 5:
        raise ValueError("No Lambert solution found.")

    z_values = np.linspace(z_min, z_max, n)

    previous_z = None
    previous_f = None

    for z in z_values:

        f = equation(z)

        if not np.isfinite(f):
            continue

        if previous_f is not None:

            if f * previous_f < 0:
                return previous_z, z

        previous_z = z
        previous_f = f

    return find_z_bracket(equation, z_min*19, z_max*10, n, call_count = call_count + 1)

def lambert(r1_vec, r2_vec, dt, mu=MU_SUN):

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_theta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    A = sin_theta * np.sqrt((r1 * r2) / (1.0 - cos_theta))

    def y(z):

        C = stumpff_C(z)
        S = stumpff_S(z)

        return (r1 + r2 + A * (z * S - 1.0) / np.sqrt(C))

    def tof(z):
        # time of flight

        C = stumpff_C(z)
        S = stumpff_S(z)

        Y = y(z)

        if Y <= 0 or C <= 0:
            return np.nan

        chi = np.sqrt(Y / C)

        return (chi**3 * S + A * np.sqrt(Y)) / np.sqrt(mu)

    def equation(z):
        return tof(z) - dt

    z_min, z_max = find_z_bracket(equation)

    z = brentq(equation, z_min, z_max)

    Y = y(z)

    f = 1.0 - Y / r1

    g = A * np.sqrt(Y / mu)

    g_dot = 1.0 - Y / r2

    v1 = (r2_vec - f * r1_vec) / g

    v2 = (g_dot * r2_vec - r1_vec) / g

    return v1, v2

if __name__ == "__main__":

    mu = 398600.0

    r1 = np.array([
        8000.0,
        0.0,
        0.0
    ])

    r2 = np.array([
        0.0,
        9905.0,
        0.0
    ])

    dt = 1330.0

    v1, v2 = lambert(
        r1,
        r2,
        dt,
        mu
    )

    print("v1 =", v1)
    print("v2 =", v2)

