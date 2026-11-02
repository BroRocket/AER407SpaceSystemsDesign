
from unitvalue import UnitValue
from rocketprops.rocket_prop import get_prop

class Propellant:

    def __init__(self, name: str):

        self.name = name
        self.prop = get_prop(name)

    def properties(self, T: UnitValue, P: UnitValue):
        """
        Calculate liquid propellant properties.

        Parameters
        ----------
        T : float
            Temperature [K]

        P : float
            Absolute pressure [Pa]

        Returns
        -------
        dict
            Density [kg/m^3]
            Dynamic viscosity [Pa.s]
            Saturation pressure [Pa]
        """

        # Convert to RocketProps units
        T_R = T.to("R")
        P_psia = P.to("psi")

        # Compressed-liquid properties
        rho = self.prop.SG_compressed(T_R.value, P_psia.value)
        mu = self.prop.Visc_compressed(T_R.value, P_psia.value)

        # Saturation pressure at temperature T
        P_sat = self.prop.PvapAtTdegR(T_R)

        return {
            "density": UnitValue.create_unit("kg/m^3", rho * 1000),
            "viscosity": UnitValue.create_unit("kg/ms", mu * 0.1),
            "saturation_pressure": UnitValue.create_unit('psi', P_sat).convert_base_metric()
        }
