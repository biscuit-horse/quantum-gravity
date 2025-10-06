"""Order-of-magnitude estimates for phase shifts from foam-induced graviton mass."""
import math

# Physical constants
HBAR = 1.054571817e-34  # J*s
C = 2.99792458e8  # m/s
EV_TO_J = 1.602176634e-19  # J/eV
PARSEC = 3.0856775814913673e16  # m
GPC = 1e9 * PARSEC


def phase_shift(m_geff_ev: float, distance_gpc: float, frequency_hz: float) -> float:
    """Return phase shift Δφ in radians for given parameters."""
    omega = 2 * math.pi * frequency_hz
    D = distance_gpc * GPC
    m_geff_kg = (m_geff_ev * EV_TO_J) / (C**2)
    return (m_geff_kg**2 * (C**4) * D) / (2 * HBAR * omega)


def example():
    for mg in [1e-23, 5e-24, 1e-24]:
        delta_phi = phase_shift(mg, distance_gpc=1.0, frequency_hz=100.0)
        print(f"m_g_eff={mg:.1e} eV -> Δφ ≈ {delta_phi:.3e} rad")


if __name__ == "__main__":
    example()
