"""
Auhtor: Brody Howard, Fluid Mechanical Engineering Intern
Date: 2/8/20226
"""

import math
from scipy.optimize import root

import matplotlib as plt

class Pipe():
    """
    Expects everyhting to be in SI units
    """
    def __init__(self, length : float, diameter : float, density : float, dynamic_visocity : float, roughness : float = 2.5e-5):
        self.pipe_length = length
        self.pipe_diameter = diameter
        self.rho = density
        self.dynamic_viscocity = dynamic_visocity
        self.roughness = roughness
        self.area = math.pi * (self.pipe_diameter**2) / 4

    def pressure_loss(self, mass_flow : float):
        """
        Computes pressure loss in pipe including the coupling.
        Mass flow must be in kg/s
        """
        if mass_flow <= 0:
            return 0.0
        
        velocity = mass_flow / (self.rho * self.area)
        
        Re_d = (4 * mass_flow) / (self.dynamic_viscocity * math.pi * self.pipe_diameter)
        print(f"{velocity}, {Re_d}")

        def Colebrook_White(f):
            LHS = 1 / math.sqrt(f[0])
            RHS = -2 * math.log10((self.roughness/(3.7*self.pipe_diameter)) + (2.51/(Re_d*math.sqrt(f[0]))))
            return RHS - LHS
       
        if Re_d < 2500:
            # Laminar Flow 
            f = 64 / Re_d
        else: 
            # Turbulent 
            sol = root(Colebrook_White, 0.01, tol=1e-11)
            f = sol.x[0]

        k = f * (self.pipe_length / self.pipe_diameter)
        p_loss = k * 0.5 * self.rho * (velocity**2)
        
        return float(p_loss)


## Validation ##
if __name__ == "__main__":
    D1 = 0.00635
    D2 = 0.0127
    D3 = 0.0254
    roughness = 0.015/1000
    m_dot = 0.0307
    rho = 800
   
    pipe1 = Pipe(1.0, D1, 800, 0.00164, roughness)
    p_loss1 = pipe1.pressure_loss(m_dot)

    pipe2 = Pipe(1.0, D2, 800, 0.00164, roughness)
    p_loss2 = pipe2.pressure_loss(m_dot)

    pipe3 = Pipe(1.0, D3, 800, 0.00164, roughness)
    p_loss3 = pipe3.pressure_loss(m_dot)  

    print(f"dp (1/4): {(p_loss1/ 6895):.4f} psi, dp (1/2): {(p_loss2/  6895):.4f} psi, dp (1): {(p_loss3/  6895):.4f} psi")

    # Water hammer
    E = 2e+11
    K = 1.35e+9

    wave_speed = math.sqrt(1 / (rho * ((1/K) +(D2 / (E*roughness)))))
    water_hammer_dp = rho * wave_speed * (m_dot / (rho * ((math.pi/4)*D2**2)))

    print(f"Water Hammer dp: {water_hammer_dp / 6895} psi")
    print(f"Total P: {(water_hammer_dp + 2250000) / 6895} psi")

    lbs_min = m_dot * 7937
    density_lb_gal = rho/119.8
    GPM = lbs_min / (60 * density_lb_gal)
    CV = 7.5

    dp_valve = (rho/1000) * (GPM/CV)**2
    print(f"{dp_valve} psi")


    # p1 = Pipe(718/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p2 = Pipe(318/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p3 = Pipe(281/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p4 = Pipe(675/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p5 = Pipe(801/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p6 = Pipe(1481/1000, 61.5/1000, 752.9610, 6.445417E-04, 5e-8)
    # p7 = Pipe(214/1000, 49.4/1000, 752.9610, 6.445417E-04, 5e-8)
    # p8 = Pipe(716/1000, 49.4/1000, 752.9610, 6.445417E-04, 5e-8)
    # p9 = Pipe(1661/1000, 39.4/1000, 752.9610, 6.445417E-04, 5e-8)

    # # 7.225 tonnes/hr
    # flow = 7225/3600

    # results_from_neo_report = [69.45409, 30.79575, 27.16797, 65.29675, 77.46604, 143.23765, 58.96691, 197.47047, 1353.82608]
    # pipe_model_results = [p1.pressure_loss(flow),
    #                       p2.pressure_loss(flow),
    #                       p3.pressure_loss(flow),
    #                       p4.pressure_loss(flow),
    #                       p5.pressure_loss(flow),
    #                       p6.pressure_loss(flow),
    #                       p7.pressure_loss(flow),
    #                       p8.pressure_loss(flow),
    #                       p9.pressure_loss(flow)]

    # print(f"Model Results: {pipe_model_results}")

    # absolute_err = []
    # percent_diff = []
    
    # for model_result, report_result in zip(pipe_model_results, results_from_neo_report):
    #     absolute_err.append(model_result - report_result)
    #     percent_diff.append((100*(model_result - report_result))/report_result)
       

    # print(f"Absoulte Err: {absolute_err}")
    # temp = 0
    # diff_avg = sum(percent_diff)/len(percent_diff)
    # for val in percent_diff:
    #     temp += (val - diff_avg)**2
    # print(f"Percent Diff: {percent_diff}\n, Avg Pdiff: {diff_avg}, STD Pdif = {math.sqrt(temp/(len(percent_diff)-1))} ")

