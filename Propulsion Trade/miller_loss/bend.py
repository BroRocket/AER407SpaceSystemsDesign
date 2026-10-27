"""
Auhtor: Brody Howard
Date: 2/8/20226
"""

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline
import matplotlib.pyplot as plt

class Bend():
    """
    Calculates head loss through a bend using Miller's Method.
    Expects all units in SI (meters, kg, seconds, Pascals).
    """
    
    _rd_axis = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12, 20]) 
    _angle_axis = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 120.0, 150.0, 180.0])

    _k_matrix = np.array([
        # 10,   20,   30,   40,   50,     60,   90,   120,  150,  180 angle degrees
        [0.03, 0.05, 0.12, 0.22, 0.33,   0.50, 1.00, 1.00, 1.00, 1.00], # r/D = 0.5
        [0.02, 0.035,0.06, 0.08, 0.12,   0.15, 0.24, 0.26, 0.28, 0.28], # r/D = 1.0 
        [0.02, 0.04, 0.06, 0.08, 0.10,   0.12, 0.18, 0.20, 0.21, 0.21], # r/D = 1.5 
        [0.02, 0.045,0.06, 0.08, 0.095,  0.11, 0.16, 0.18, 0.19, 0.19], # r/D = 2.0 
        [0.03, 0.05, 0.07, 0.082,0.10,   0.115,0.155,0.19, 0.21, 0.22], # r/D = 3.0
        [0.035,0.055,0.07, 0.09, 0.11,   0.122,0.17, 0.22, 0.24, 0.29], # r/D = 4.0
        [0.04, 0.061,0.08, 0.105,0.13,   0.14, 0.20, 0.25, 0.30, 0.30], # r/D = 6.0
        [0.045,0.07, 0.095,0.12, 0.14,   0.16, 0.22, 0.27, 0.30, 0.30],  # r/D = 8.0
        [0.055,0.088,0.125,0.15, 0.16,   0.20, 0.26, 0.31, 0.30, 0.30], #r/D = 12 
        [0.075,0.124,0.185,0.20, 0.21,   0.28, 0.34, 0.39, 0.35, 0.35] #r/D = 20
    ])
    # Note the rows for r/D = 12 and 20 are linear extrapolations as the miller plot does not cover this

    # move to log space
    _log_k_matrix = np.log10(_k_matrix)

    #Miller Kb interp
    _miller_log_interp = RegularGridInterpolator(
        (_rd_axis, _angle_axis), 
        _log_k_matrix, 
        method='linear',  # Robust, no waviness
        bounds_error=False, 
        fill_value=None   # Allow extrapolation (safe because linear)
    )

    # Reynolds Corrections, Log10 of Re for smoother spline interpolation
    _re = [10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]
    
    _cre_rd1_data = [2.20, 1.85, 1.46, 1.25, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    _cre_rd1_5_data = [2.20, 1.95, 1.60, 1.40, 1.20, 1.00, 1.00, 1.00, 1.00, 1.00]
    _cre_rd2_data = [2.20, 1.98, 1.68, 1.50, 1.32, 1.15, 1.02, 0.90, 0.78, 0.72]

    # Outlet Correction Factor
    _outlet_l_over_d = [0, 0.25, 0.5, 0.75, 1, 2, 4, 6, 10, 20, 30, 40, 50]

    _Co_kb01_spline = CubicSpline(_outlet_l_over_d, [0.61, 0.57, 0.55, 0.5, 0.485, 0.52, 0.57, 0.63, 0.725, 0.9, 1.0, 1.0, 1.0])
    _Co_kb015_spline = CubicSpline(_outlet_l_over_d, [0.85, 0.78, 0.68, 0.63, 0.6, 0.61, 0.68, 0.73, 0.8, 0.95, 1.0, 1.0, 1.0])
    _Co_kb020_spline = CubicSpline(_outlet_l_over_d, [1.33, 1.18, 0.85, 0.74, 0.68, 0.67, 0.73, 0.79, 0.86, 1.0, 1.0, 1.0, 1.0])
    _Co_kb025_spline = CubicSpline(_outlet_l_over_d, [2.75, 2.6, 2.2, 1.15, 0.77, 0.73, 0.8, 0.85, 0.91, 1.0, 1.0, 1.0, 1.0])
    _Co_kb1_spline = CubicSpline(_outlet_l_over_d, [2.75, 2.6, 2.5, 2.0, 1.6, 0.98, 0.9, 0.93, 0.985, 1.0, 1.0, 1.0, 1.0])

    def __init__(self, diameter: float, density: float, dynamic_viscosity: float, bend_angle: float, r_D: float, outlet_length: float):
        self.pipe_diameter = diameter
        self.rho = density
        self.dynamic_viscosity = dynamic_viscosity
        self.bend_angle = bend_angle
        self.r_over_D = r_D
        self.l_over_d = outlet_length/diameter
        self.area = math.pi * (self.pipe_diameter**2) / 4 
        
        #Calculate base Kb
        log_kb = float(self._miller_log_interp([self.r_over_D, self.bend_angle]).item())
        self.Kb_star = 10**log_kb
        
    def get_outlet_correction(self) -> float:
        if self.l_over_d >= 40:
            return 1.0
        
        if self.Kb_star <= 0.1:
            return float(self._Co_kb01_spline(self.l_over_d).item())
        if self.Kb_star <= 0.15:
            factor = (self.Kb_star - 0.1) / 0.05 
            val1 = float(self._Co_kb01_spline(self.l_over_d).item())
            val2 = float(self._Co_kb015_spline(self.l_over_d).item())
            return val1 + factor * (val2 - val1)
        if self.Kb_star <= 0.2:
            factor = (self.Kb_star - 0.15) / 0.05 
            val1 = float(self._Co_kb015_spline(self.l_over_d).item())
            val2 = float(self._Co_kb020_spline(self.l_over_d).item())
            return val1 + factor * (val2 - val1)
        if self.Kb_star <= 0.25:
            factor = (self.Kb_star - 0.2) / 0.05 
            val1 = float(self._Co_kb020_spline(self.l_over_d).item())
            val2 = float(self._Co_kb025_spline(self.l_over_d).item())
            return val1 + factor * (val2 - val1)
        if self.Kb_star < 1:
            factor = (self.Kb_star - 0.25) / 0.75 
            val1 = float(self._Co_kb025_spline(self.l_over_d).item())
            val2 = float(self._Co_kb1_spline(self.l_over_d).item())
            return val1 + factor * (val2 - val1)
        if self.Kb_star >= 1:
            return float(self._Co_kb1_spline(self.l_over_d).item())

    def _log_interp(self, x_target : float, x_points: list|np.ndarray, y_points: list|np.ndarray) -> float:
        """
        Performs Linear Interpolation in Log-Log space. forces the curve to follow a power law for Cre between points.
        """
        log_x_data = np.log10(x_points)
        log_y_data = np.log10(y_points)
        
        log_y_result = np.interp(math.log10(x_target), log_x_data, log_y_data)
        return float(10**log_y_result)

    def get_reynolds_correction(self, Re_d) -> float:
        """Interpolates C_re based on Re and r/D"""

        Cre_rd1 = self._log_interp(Re_d, self._re, self._cre_rd1_data)
        Cre_rd1_5 = self._log_interp(Re_d, self._re, self._cre_rd1_5_data)
        Cre_rd2 = self._log_interp(Re_d, self._re, self._cre_rd2_data)
        
        if self.r_over_D < 0.7 and self.Kb_star >= 0.4:
            #MILLER equation 9.2 for sharp bends
            if Re_d < 10000:
                return self.Kb_star / (self.Kb_star - (0.2 * 9.5376*(Re_d**-0.158)) + 0.2)
            else:
                return self.Kb_star / (self.Kb_star - (0.2 * Cre_rd1) + 0.2)
        
        #Use fit curve for Re outisde of Miller diagrams
        if Re_d < 10000: 
            return 9.5376*(Re_d**-0.158)
        
        #Interpolate based on r/D
        if self.r_over_D <= 1.0:
            return Cre_rd1
        elif self.r_over_D < 1.5:
            # Linear interpolation between the r/D=1 and r/D=1.5
            factor = (self.r_over_D - 1.0) / 0.5
            return Cre_rd1 + factor * (Cre_rd1_5 - Cre_rd1)
        elif self.r_over_D < 2.0:
            # Linear interpolation between r/D=1.5 and r/D=2.0
            factor = (self.r_over_D - 1.5) / 0.5
            return Cre_rd1_5 + factor * (Cre_rd2 - Cre_rd1_5)
        else:
            #r/D >= 2, 
            return Cre_rd2

    def pressure_loss(self, mass_flow: float) -> float:
        """Returns Pressure Loss in Pascals. Expect mass flow in kg/s"""
        if mass_flow <= 0: return 0.0

        
        velocity = mass_flow / (self.rho * self.area)

        Re_d = (4 * mass_flow) / (self.dynamic_viscosity * math.pi * self.pipe_diameter)

        Cre = self.get_reynolds_correction(Re_d)
        Co = self.get_outlet_correction()

        Kb = self.Kb_star * Cre * Co
        p_loss = 0.5* Kb * self.rho * velocity**2
        return p_loss


## Validation ### 
if __name__ == "__main__":

    bend1 = Bend(61.5/1000, 752.9610, 6.445417E-04, 46, 1.3, 318/1000)
    bend2 = Bend(61.5/1000, 752.9610, 6.445417E-04, 42, 1.3, 281/1000)
    bend3 = Bend(61.5/1000, 752.9610, 6.445417E-04, 15, 1.6, 675/1000) 
    bend4 = Bend(61.5/1000, 752.9610, 6.445417E-04, 23, 1.6, 801/1000)
    bend5 = Bend(61.5/1000, 752.9610, 6.445417E-04, 12, 1.6, 1481/1000)
    bend6 = Bend(61.5/1000, 752.9610, 6.445417E-04, 95, 1.8, 680/1000)
    bend7 = Bend(61.5/1000, 752.9610, 6.445417E-04, 89, 1.5, 360/1000)
    bend8 = Bend(61.5/1000, 752.9610, 6.445417E-04, 154, 2.0, 320/1000)
    
    test = bend1._miller_log_interp([2, 0])
    print(10**test)

    flow = 7225/3600
    report_results = [38.83724,
                    34.52199,
                    12.94575,
                    18.55557,
                    9.92507,
                    77.67447,
                    71.20160,
                    79.83210]
    
    model_results = [bend1.pressure_loss(flow), 
                     bend2.pressure_loss(flow), 
                     bend3.pressure_loss(flow), 
                     bend4.pressure_loss(flow),
                     bend5.pressure_loss(flow),
                     bend6.pressure_loss(flow),
                     bend7.pressure_loss(flow),
                     bend8.pressure_loss(flow)]
    
    print(f"Previous Model Results: {report_results}")
    print(f"Model Results: {model_results}")

    absolute_err = []
    percent_diff = []
    for model_result, report_result in zip(model_results, report_results):
        absolute_err.append(model_result - report_result)
        percent_diff.append((100*(model_result - report_result))/report_result)

    print(f"Absoulte Err: {absolute_err}")
    temp = 0
    diff_avg = sum(percent_diff)/len(percent_diff)
    for val in percent_diff:
        temp += (val - diff_avg)**2
    print(f"Percent Diff: {percent_diff}\n Avg Pdiff: {diff_avg}, STD Pdif = {math.sqrt(temp/(len(percent_diff)-1))} ")

    print(10**Bend._miller_log_interp([0.8, 120]))
    print(10**Bend._miller_log_interp([6, 20]))
    print(10**Bend._miller_log_interp([4, 90]))

    r_d_plot = np.linspace(0.5, 8.0, 200)
    angle_plot = np.linspace(10.0, 180.0, 200)
    R_grid, A_grid = np.meshgrid(r_d_plot, angle_plot)

    points = np.column_stack((R_grid.ravel(), A_grid.ravel()))
    kb_values = Bend._miller_log_interp(points).reshape(R_grid.shape)
    kb_values = 10**kb_values
    # 3. Plot
    plt.figure(figsize=(10, 6))
    # Note: R_grid is X, A_grid is Y
    cp = plt.contourf(R_grid, A_grid, kb_values, levels=40, cmap='viridis') 
    plt.colorbar(cp, label='Loss Coefficient ($K_b^*$)')
    plt.xlabel('Radius Ratio ($r/D$)')
    plt.ylabel('Bend Angle (degrees)')
    plt.title('Miller $K_b^*$ Values ($r/D$ vs Angle)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.show()

    re_plot_values = np.logspace(4, 7, 500) # Re from 1,000 to 10,000,000

    # Calculate Y values
    bend1 = Bend(1, 1, 1, 1, 1.0, 1)
    bend2 = Bend(1, 1, 1, 1, 1.5, 1)
    bend3 = Bend(1, 1, 1, 1, 2.0, 1)

    y_rd1 = [bend1.get_reynolds_correction(r) for r in re_plot_values]
    y_rd15 = [bend2.get_reynolds_correction(r) for r in re_plot_values]
    y_rd2 = [bend3.get_reynolds_correction(r) for r in re_plot_values]

    plt.figure(figsize=(10, 6))
    plt.plot(re_plot_values, y_rd1, label='r/D = 1.0 (Class 2)', linewidth=2)
    plt.plot(re_plot_values, y_rd15, label='r/D = 1.5', linewidth=2)
    plt.plot(re_plot_values, y_rd2, label='r/D = 2.0 (Class 1)', linewidth=2)

    plt.xscale('log') # Miller plots X in Log scale
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.title('Reynolds Correction Factor ($C_{Re}$) Interpolation')
    plt.xlabel('Reynolds Number ($Re$)')
    plt.ylabel('Correction Factor ($C_{Re}$)')
    plt.legend()
    plt.ylim(0.6, 2.5)
    plt.show()

    ld_plot_values = np.linspace(0, 50, 200) # L/D from 0 to 50
    kb_cases = [0.1, 0.15, 0.2, 0.25, 1.0]

    bends = []
    for ld in ld_plot_values:
        bends.append(Bend(1, 1, 1, 10, 1, ld))

    plt.figure(figsize=(10, 6))
    for kb in kb_cases:
        for i in range(len(bends)): 
            bends[i].Kb_star = kb 
        co_vals = [bend.get_outlet_correction() for bend in bends]
        plt.plot(ld_plot_values, co_vals, label=f'Kb* = {kb}')

    plt.grid(True, alpha=0.5)
    plt.title('Outlet Length Correction Factor ($C_o$) Interpolation')
    plt.xlabel('Outlet Length Ratio ($L/D$)')
    plt.ylabel('Correction Factor ($C_o$)')
    plt.legend()
    plt.show()
