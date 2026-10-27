"""
Auhtor: Brody Howard, Fluid Mechanical Engineering Intern
Date: 2/8/20226
"""

import math

class Coupling():
    def __init__(self, diameter: float, density: float, k_coupling : float) -> None:
        '''Expects all units in SI'''
        self.K = k_coupling
        self.diameter = diameter
        self.rho = density
        self.area = math.pi * (self.diameter**2) / 4

    def pressure_loss(self, mass_flow : float) -> float:
        '''Calculates pressure loss in Pascals. Mass flow in kg/s'''
        velocity = mass_flow / (self.rho * self.area)
        p_loss = self.K * 0.5 * self.rho * (velocity**2)

        return p_loss
    
## Validation ##
if __name__ == "__main__":
    coupling1 = Coupling(61.5/1000, 752.9610, 0.02)
    coupling2 = Coupling(49.4/1000, 752.9610, 0.02)
    coupling3 = Coupling(39.4/1000, 752.9610, 0.02)

    flow = 7225/3600

    report_results = [6.06203, 14.56165, 35.98600]
    model_results = [coupling1.pressure_loss(flow), coupling2.pressure_loss(flow), coupling3.pressure_loss(flow)]

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
