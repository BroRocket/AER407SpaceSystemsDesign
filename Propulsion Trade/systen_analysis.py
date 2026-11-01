
import math
from unitvalue import UnitValue
import numpy as np
import matplotlib.pyplot as plt

from miller_loss import pipe, bend, coupling
from propellant import Propellant

'''UNIVERSAL CONSTANTS'''

G = UnitValue.create_unit("m/s^2", 9.807)

'''Propellants'''
MMH = Propellant("MMH")
MON25 = Propellant("MON25")

'''MISSION CONSTANTS'''

REQUIRED_DELTA_V = UnitValue.create_unit("m/s", 1134) # in m/s
INITIAL_MASS = UnitValue.create_unit("kg", 1650)

'''ENGINE SPECIFICATIONS'''

THRUST = UnitValue.create_unit("N", 425*3) # newtons with three main engines
ISP = UnitValue.create_unit("s", 321) # Assumed for hypergolic engine
COMBUSTION_PRESSURE = UnitValue.create_unit("bar", 10.35).convert_base_metric()

# MMH CHEMICAL FORMULA C_H_6_N_2
# MON-25 CHEMICAL FORMULA (1.03 * N_2_O_4 + N_O)

#STOICH
# C_H_6_N_2 + (N_2_O_4 + N_O) = C_O_2 + 3 * H_2_O + * 5/2 * N_2

#Oxidizer to fuel ratio
#  (30.006 g/mol + 92.011 g/mol) / 46.07 g/mol
#  = 2.648513132190145430866073366616 = 2.649

O_F_RATIO =  1.65 #2.649, not stoich so engine does not overheat

'''Propellant Tank Material Properties
 Material Ti-6%Al-4%V (https://www.azom.com/article.aspx?ArticleID=1341)
'''
PROPELLANT_TANK_DENSITY = UnitValue.create_unit("kg/m^3", 4.42 * 1000).convert_base_metric() #g/cm converted to kg/m^3
PROPELLANT_TANK_YIELD_STRENGTH = UnitValue.create_unit("MPa", 1100).convert_base_metric() # 0.02% proof (MPa)

FILL_FACTOR = 0.95

'''Feed System Charecteristics'''
INJECTOR_PRESSURE_RATIO = 0.8 #estimate
ENGINE_VALVE_PLOSS = UnitValue.create_unit("bar", 1.1).convert_base_metric() # From manfacturer data at expected mass flow
VENTURI_PRESSURE_RATIO = 0.6 #to ensure cavitation
FUEL_SYSTEM_COMPONENT_IDIAMETER1 = UnitValue.create_unit("inch", 0.391).convert_base_metric()
FUEL_SYSTEM_COMPONENT_IDIAMETER2 = UnitValue.create_unit("inch", 0.172).convert_base_metric()
FUEL_SYSTEM_PIPE_LENGHTS1 = UnitValue.create_unit("m", 1.5)
FUEL_SYSTEM_PIPE_LENGHTS2 = UnitValue.create_unit("m", 0.75)
PIPE_ROUGHNESS = UnitValue.create_unit("m", 0.015/1000)
MAIN_FUEL_VALVE_IDIAMTER = UnitValue.create_unit("mm", 12.9)
MAIN_FUEL_VALVE_CV = 7.5

'''Helium Tank Material Properties
Assumes Composite Tank
'''
HE_TANK_DENSITY = UnitValue.create_unit("kg/m^3", 1600).convert_base_metric() #g/cm converted to kg/m^3
HE_TANK_YIELD_STRENGTH = UnitValue.create_unit("MPa", 1300).convert_base_metric() # 0.02% proof (MPa)

'''HELIUM TANK CHARACTERSITICS'''
#assume want COPV pressure > 1000psi and start at 2000psi
COPV_MAX_PRESSURE = UnitValue.create_unit("psig", 2000).convert_base_metric()
COPV_MIN_PRESSURE = UnitValue.create_unit("psig", 1000).convert_base_metric()
HE_PIPE_LENGHTS1 = UnitValue.create_unit("m", 2) # represents parts of lines up to both regulators
HE_PIPE_IDIAMETER1 = UnitValue.create_unit("inch", 0.172).convert_base_metric()
HE_PIPE_LENGHTS2 = UnitValue.create_unit("m", 0.5) # represents parts of lines into one tanks and it's vent channels
HE_PIPE_IDIAMETER2 = UnitValue.create_unit("inch", 0.172).convert_base_metric()

R_HELIUM = UnitValue.create_unit("J/kgK", 2077.1)

'''SAFETY FACTORS AND MARGINS'''
STRUCTURAL_SAFETY_FACTOR = 2
TANK_PRESSURE_MARGIN = 1.5
MAX_LAUNCH_G = 11


def calculate_system_characterisitcs(system_temperature: UnitValue, MMH_tank_pressure_guess: UnitValue, MON25_tank_pressure_guess: UnitValue, verbose: bool = False):
    '''Main fucniton to do system sizing calcs'''

    MMH_prev_pressure = UnitValue.create_unit("Pa", 0)
    MMH_new_pressure = MMH_tank_pressure_guess 
    MON25_prev_pressure = UnitValue.create_unit("Pa", 0)
    MON25_new_pressure = MON25_tank_pressure_guess 

    #loop until pressure match is found
    while abs(MMH_new_pressure - MMH_prev_pressure) > 0.1 and abs(MON25_new_pressure - MON25_prev_pressure) > 0.01:

        MMH_prev_pressure = MMH_new_pressure
        MON25_prev_pressure = MON25_new_pressure
        MMH_properties = MMH.properties(system_temperature, MMH_prev_pressure)
        MMH_density = MMH_properties["density"]
        MMH_dynamic_viscocity = MMH_properties["viscosity"]
        MMH_saturation_pressure = MMH_properties["saturation_pressure"]
        MON25_properties = MON25.properties(system_temperature, MON25_prev_pressure)
        MON25_density = MON25_properties["density"]
        MON25_dynamic_viscocity = MON25_properties["viscosity"]
        MON25_saturation_pressure = MON25_properties["saturation_pressure"]

        # Get system total propellant mass and mass flow using rocket eqns
        total_mass_flow_rate = THRUST / (G * ISP)
        total_propelant_mass = INITIAL_MASS * (1 - (1/math.exp(REQUIRED_DELTA_V / (ISP * G))))

        #Get individual fuel and oxidizer mass and mass flow
        mass_MMH = total_propelant_mass / (1 + O_F_RATIO)
        mass_flow_MMH = total_mass_flow_rate / (1 + O_F_RATIO)
        MMH_volume = mass_MMH/MMH_density

        mass_MON25 = total_propelant_mass - mass_MMH
        mass_flow_MON25 = total_mass_flow_rate - mass_flow_MMH
        MON25_volume = mass_MON25/MON25_density

        # Compute pressure in tank
        #Engine INlet Pressure, Injector loss factor plus single engine valve loss
        eip = COMBUSTION_PRESSURE / INJECTOR_PRESSURE_RATIO + ENGINE_VALVE_PLOSS 

        #Add pipe losses
        MMH_sinlge_engine_feed_pipe = pipe.Pipe(FUEL_SYSTEM_PIPE_LENGHTS2, FUEL_SYSTEM_COMPONENT_IDIAMETER2, MMH_density, MMH_dynamic_viscocity, PIPE_ROUGHNESS)
        MON25_sinlge_engine_feed_pipe = pipe.Pipe(FUEL_SYSTEM_PIPE_LENGHTS2, FUEL_SYSTEM_COMPONENT_IDIAMETER2, MON25_density, MON25_dynamic_viscocity, PIPE_ROUGHNESS)
        MMH_single_engine_pipe_losses = MMH_sinlge_engine_feed_pipe.pressure_loss(mass_flow_MMH / 3) # divide by three as these lines feed one engine
        MON25_single_engine_pipe_losses = MON25_sinlge_engine_feed_pipe.pressure_loss(mass_flow_MON25 / 3)

        #main flow valve loss
        k = 2.148e-3 *  ((MAIN_FUEL_VALVE_IDIAMTER.value**4) / (MAIN_FUEL_VALVE_CV**2))
        MMH_mfv = coupling.Coupling(MAIN_FUEL_VALVE_IDIAMTER, MMH_density, k)
        MON25_mfv = coupling.Coupling(MAIN_FUEL_VALVE_IDIAMTER, MON25_density, k)
        # add valve losses
        MMH_main_valve_pressure_loss = MMH_mfv.pressure_loss(mass_flow_MMH)
        MON25_main_valve_pressure_loss = MON25_mfv.pressure_loss(mass_flow_MON25)
        
        # incluides all things upstream of venturri
        MMH_pressure_before_venturi = (eip + MMH_main_valve_pressure_loss + MMH_single_engine_pipe_losses) / VENTURI_PRESSURE_RATIO
        MON25_pressure_before_venturi = (eip + MON25_main_valve_pressure_loss + MON25_single_engine_pipe_losses) / VENTURI_PRESSURE_RATIO

        MMH_combined_engine_feed_pipe = pipe.Pipe(FUEL_SYSTEM_PIPE_LENGHTS1, FUEL_SYSTEM_COMPONENT_IDIAMETER1, MMH_density, MMH_dynamic_viscocity, PIPE_ROUGHNESS)
        MON25_combined_engine_feed_pipe = pipe.Pipe(FUEL_SYSTEM_PIPE_LENGHTS1, FUEL_SYSTEM_COMPONENT_IDIAMETER1, MON25_density, MON25_dynamic_viscocity, PIPE_ROUGHNESS)

        # Equals to required ullage pressure for when tank completeley depletes
        MMH_tank_outlet_pressure = MMH_pressure_before_venturi + MMH_combined_engine_feed_pipe.pressure_loss(mass_flow_MMH)
        MON25_tank_outlet_pressure = MON25_pressure_before_venturi + MON25_combined_engine_feed_pipe.pressure_loss(mass_flow_MON25)

        #Set new pressures
        MMH_new_pressure = MMH_tank_outlet_pressure
        MON25_new_pressure = MON25_tank_outlet_pressure

    #Add Fill factor to ensure sufficient ullage space
    MMH_tank_inner_volume_w_fill_factor = MMH_volume / FILL_FACTOR
    MON25_tank_inner_volume_w_fill_factor = MON25_volume / FILL_FACTOR

    #spherical tank stress /sigma = p*r / 2t
    MMH_sph_tank_inner_rad = ((MMH_tank_inner_volume_w_fill_factor * (3/4)) / math.pi) ** (1/3)
    MMH_tank_inner_rad_rounded = UnitValue.create_unit("m", math.ceil(MMH_sph_tank_inner_rad * 10 ** 3) / 10 **3) #to closest mm, ceiling round to ensure sufficient volume
    MON25_sph_tank_inner_rad = ((MON25_tank_inner_volume_w_fill_factor * (3/4)) / math.pi) ** (1/3)
    MON25_tank_inner_rad_rounded = UnitValue.create_unit("m", math.ceil(MON25_sph_tank_inner_rad * 10 ** 3) / 10 **3)

    #Compute a max tank pressure during launch with margin
    MMH_bottom_of_tank_pressure_at_launch = MMH_density * MAX_LAUNCH_G * G * 2 * MMH_tank_inner_rad_rounded # conservative higher raidus even thoguh it shouldnt be this full
    MON25_bottom_of_tank_pressure_at_launch = MON25_density * MAX_LAUNCH_G * G * 2 * MON25_tank_inner_rad_rounded

    MMH_max_tank_pressure = (MMH_bottom_of_tank_pressure_at_launch + MMH_tank_outlet_pressure) * TANK_PRESSURE_MARGIN
    MON25_max_tank_pressure = (MON25_bottom_of_tank_pressure_at_launch + MON25_tank_outlet_pressure) * TANK_PRESSURE_MARGIN

    # Compute required thickenss
    MMH_tank_thickness = (MMH_max_tank_pressure * MMH_tank_inner_rad_rounded) / (2 * (PROPELLANT_TANK_YIELD_STRENGTH / STRUCTURAL_SAFETY_FACTOR))
    MMH_rounded_thickness = UnitValue.create_unit("m", math.ceil(MMH_tank_thickness * 10 **3) / 10 **3)
    MON25_tank_thickness = (MON25_max_tank_pressure * MON25_tank_inner_rad_rounded) / (2 * (PROPELLANT_TANK_YIELD_STRENGTH / STRUCTURAL_SAFETY_FACTOR))
    MON25_rounded_thickness = UnitValue.create_unit("m", math.ceil(MON25_tank_thickness * 10 **3) / 10 **3)

    #Compute weight of tank
    MON25_tank_material_volume = 4/3 * math.pi * ((MON25_rounded_thickness + MON25_tank_inner_rad_rounded) ** 3 - MON25_tank_inner_rad_rounded ** 3)
    MON25_tank_mass = MON25_tank_material_volume * PROPELLANT_TANK_DENSITY
    MMH_tank_material_volume = 4/3 * math.pi * ((MMH_rounded_thickness + MMH_tank_inner_rad_rounded) ** 3 - MMH_tank_inner_rad_rounded ** 3)
    MMH_tank_mass = MMH_tank_material_volume * PROPELLANT_TANK_DENSITY

    #Compute final rounded volumes
    MMH_rounded_tank_inner_volume = 4/3 * math.pi * (MMH_tank_inner_rad_rounded ** 3)
    MON25_rounded_tank_inner_volume = 4/3 * math.pi * (MON25_tank_inner_rad_rounded ** 3)
    MMH_rounded_tank_outer_volume = 4/3 * math.pi * ((MMH_tank_inner_rad_rounded + MMH_rounded_thickness)** 3)
    MON25_rounded_tank_outer_volume = 4/3 * math.pi * ((MON25_tank_inner_rad_rounded + MON25_rounded_thickness) ** 3)

    #compute ullage space with rounding
    MMH_Ullage_volume = MMH_rounded_tank_inner_volume - MMH_volume
    MON25_Ullage_volume = MON25_rounded_tank_inner_volume - MON25_volume

    # Compute helium mass requirements for pressuriasation assuming helium fills entire tanks in end, no fuel/oxidizer vapour for uperbound
    # I.e no partial pressures, overesitmate but that's ok, better to have excess He than not enough
    #Assume Ideal ags law holds
    HE_mass_MMH_tank = (MMH_tank_outlet_pressure * MMH_rounded_tank_inner_volume) / (R_HELIUM * system_temperature)
    HE_mass_MON25_tank = (MON25_tank_outlet_pressure * MON25_rounded_tank_inner_volume) / (R_HELIUM * system_temperature)

    #Compute helium mass flow rate to meet volumetric flow out of tanks
    HE_massflow_MMH_tank = (MMH_tank_outlet_pressure / (R_HELIUM * system_temperature)) * (mass_flow_MMH / MMH_density)
    HE_massflow_MON25_tank = (MON25_tank_outlet_pressure / (R_HELIUM * system_temperature)) * (mass_flow_MON25 / MON25_density)

    # Determine estimate of mass of helium in the pipes up to regulators and then pipes leading to tanks and in vent channel
    HE_mass_in_lines =  ((COPV_MAX_PRESSURE / (R_HELIUM * system_temperature)) * HE_PIPE_LENGHTS1 * (math.pi / 4) * HE_PIPE_IDIAMETER1 ** 2) + ((MMH_tank_outlet_pressure / (R_HELIUM * system_temperature)) * HE_PIPE_LENGHTS2 * (math.pi / 4) * HE_PIPE_IDIAMETER2 ** 2 ) + ((MON25_tank_outlet_pressure / (R_HELIUM * system_temperature)) * HE_PIPE_LENGHTS2 * (math.pi / 4) * HE_PIPE_IDIAMETER2 ** 2 )

    #Detemrine HE tank size
    '''Maybe use non ideal gas law here fro Helium'''
    HE_copv_inner_volume = ((HE_mass_MMH_tank + HE_mass_MON25_tank + HE_mass_in_lines) * R_HELIUM * system_temperature) / (COPV_MAX_PRESSURE - COPV_MIN_PRESSURE)
    HE_copv_inner_radius = ((HE_copv_inner_volume * (3/4)) / math.pi) ** (1/3)
    HE_copv_inner_radius_rounded = UnitValue.create_unit("m", math.ceil(HE_copv_inner_radius * 10 ** 3) / 10 **3)

    #Find He tank wall thickness
    HE_copv_thickness = (COPV_MAX_PRESSURE * TANK_PRESSURE_MARGIN * HE_copv_inner_radius_rounded) / (2 * (HE_TANK_YIELD_STRENGTH / STRUCTURAL_SAFETY_FACTOR))
    HE_copv_thickness_rounded = UnitValue.create_unit("m", math.ceil(HE_copv_thickness * 10 ** 3) / 10 ** 3)
    HE_copv_outer_volume_rounded = 4/3 * math.pi * (HE_copv_thickness_rounded + HE_copv_inner_radius_rounded) ** 3
    HE_copv_inner_volume_rounded = 4/3 * math.pi * (HE_copv_inner_radius_rounded) ** 3

    HE_mass_unusable = (COPV_MIN_PRESSURE * HE_copv_inner_volume_rounded) / (R_HELIUM * system_temperature)
    #Detemrine total mass of He required 
    HE_total_mass = HE_mass_MMH_tank + HE_mass_MON25_tank + HE_mass_in_lines + HE_mass_unusable

    HE_copv_material_volume = HE_copv_outer_volume_rounded - HE_copv_inner_volume_rounded
    HE_copv_mass = HE_TANK_DENSITY * HE_copv_material_volume

    if verbose == True:
        print(MMH_properties)
        print(MON25_properties)
        print("="*20)
        print(f"Total Propellant Mass: {total_propelant_mass}")
        print(f"{mass_MMH} of MMH\nMass Flow MMH: {mass_flow_MMH}")
        print(f"{mass_MON25} of MON25\nMass Flow MON25: {mass_flow_MON25}")
        print(f"Volume of MMH: {MMH_volume}")
        print(f"Volume of MON25: {MON25_volume}")
        print("="*20)

        print(f"MMH Tank Inner Radius: {MMH_sph_tank_inner_rad}, rounded: {MMH_tank_inner_rad_rounded}")
        print(f"MON Tank Inner Radius: {MON25_sph_tank_inner_rad}, rounded: {MON25_tank_inner_rad_rounded}")
        print(f"MON25 Tank Thickness: {MON25_tank_thickness}, rounded: {MON25_rounded_thickness}")
        print(f"MON25 Tank Thickness: {MMH_tank_thickness}, rounded: {MMH_rounded_thickness}")

        print("="*20)
        print(f"MMH Tank Required Ullage Pressure: {MMH_tank_outlet_pressure}, MMH Saturation Pressure: {MMH_saturation_pressure}")
        print(f"MON25 Tank Required Ullage Pressure: {MON25_tank_outlet_pressure}, MON25 Saturation Pressure: {MON25_saturation_pressure}")
        print(f'Max Tank pressure (MON25) with {TANK_PRESSURE_MARGIN} Pressure Margin: {MON25_max_tank_pressure}')
        print(f'Max Tank pressure (MMH) with {TANK_PRESSURE_MARGIN} Pressure Margin: {MMH_max_tank_pressure}')

        print("="*20)
        print(f"Final MMH Tank Specifications:\n    -Inner Diameter: {2*MMH_tank_inner_rad_rounded}\n   -Wall Thickness: {MMH_rounded_thickness}\n    -Outer Diameter: {(MMH_tank_inner_rad_rounded + MMH_rounded_thickness)*2}\n   -Outer Volume: {MMH_rounded_tank_outer_volume}\n   -Inner Volume: {MMH_rounded_tank_inner_volume}\n    -Tank Mass: {MMH_tank_mass}\n    -Ullage Volume: {MMH_Ullage_volume}\n  -Ullage  Percent: {MMH_Ullage_volume/MMH_rounded_tank_inner_volume*100}%")
        print(f"Final MON25 Tank Specifications:\n    -Inner Diameter: {2*MON25_tank_inner_rad_rounded}\n   -Wall Thickness: {MON25_rounded_thickness}\n    -Outer Diameter: {(MON25_tank_inner_rad_rounded + MON25_rounded_thickness)*2}\n   -Outer Volume: {MON25_rounded_tank_outer_volume}\n  -Inner Volume: {MON25_rounded_tank_inner_volume}\n    -Tank Mass: {MON25_tank_mass}\n  -Ullage Volume: {MON25_Ullage_volume}\n  -Ullage  Percent: {MON25_Ullage_volume/MON25_rounded_tank_inner_volume*100}%")
        print("="*20)

        print(f"He Mass per Tank: MMH Tank: {HE_mass_MMH_tank}, MON25 Tank: {HE_mass_MON25_tank}")
        print(f"HE Mass Flow per Tank: MMH Tank: {HE_massflow_MMH_tank}, MON25 Tank: {HE_massflow_MON25_tank}")

        print("="*20)
        print(f"Final HE Tank Specifications:\n    -Inner Diameter: {2*HE_copv_inner_radius_rounded}\n   -Wall Thickness: {HE_copv_thickness_rounded}\n    -Outer Diameter: {(HE_copv_thickness_rounded + HE_copv_inner_radius_rounded)*2}\n   -Outer Volume: {HE_copv_outer_volume_rounded}\n   -Inner Volume: {HE_copv_inner_volume_rounded}\n    -Tank Mass: {HE_copv_mass}\n    -He Mass: {HE_total_mass}")
        print("="*20)

    return {"MMH TANK": {
            "Operating Ullage Pressure": MMH_tank_outlet_pressure,
            "Max Design Tank Pressure": MMH_max_tank_pressure,
            "Saturation Pressure": MMH_saturation_pressure,
            "Inner Diameter": 2*MMH_tank_inner_rad_rounded,
            "Wall Thickness": MMH_rounded_thickness,
            "Outer Diameter": (MMH_tank_inner_rad_rounded + MMH_rounded_thickness)*2,
            "Outer Volume": MMH_rounded_tank_outer_volume,
            "Inner Volume": MMH_rounded_tank_inner_volume,
            "Fuel Volume": MMH_volume,
            "Ullage Volume": MMH_Ullage_volume,
            "Ullage Percent": MMH_Ullage_volume / MMH_rounded_tank_inner_volume * 100,
            "Tank Mass": MMH_tank_mass,
            "MMH Mass": mass_MMH,
            "MMH Mass Flow": mass_flow_MMH
            },
            "MON25 TANK": {
            "Operating Ullage Pressure": MON25_tank_outlet_pressure,
            "Max Design Tank Pressure": MON25_max_tank_pressure,
            "Saturation Pressure": MON25_saturation_pressure,
            "Inner Diameter": 2*MON25_tank_inner_rad_rounded,
            "Wall Thickness": MON25_rounded_thickness,
            "Outer Diameter": (MON25_tank_inner_rad_rounded + MON25_rounded_thickness)*2,
            "Outer Volume": MON25_rounded_tank_outer_volume,
            "Inner Volume": MON25_rounded_tank_inner_volume,
            "Oxidizer Volume": MON25_volume,
            "Ullage Volume": MON25_Ullage_volume,
            "Ullage Percent": MON25_Ullage_volume / MON25_rounded_tank_inner_volume * 100,
            "Tank Mass": MON25_tank_mass,
            "MON25 Mass": mass_MON25,
            "MON25 Mass Flow": mass_flow_MON25
            },
            "HE TANK": {
            "Max Design Tank Pressure": COPV_MAX_PRESSURE * TANK_PRESSURE_MARGIN,
            "Inner Diameter": 2*HE_copv_inner_radius_rounded,
            "Wall Thickness": HE_copv_thickness_rounded,
            "Outer Diameter": (HE_copv_inner_radius_rounded + HE_copv_thickness_rounded)*2,
            "Outer Volume": HE_copv_outer_volume_rounded,
            "Inner Volume": HE_copv_inner_volume_rounded,
            "Tank Mass": HE_copv_mass,
            "He Total Mass": HE_total_mass,
            "He Usable Mass": HE_total_mass - HE_mass_unusable,
            "He Unusable Mass": HE_mass_unusable,
            "He_massflow_into_MMH_tank": HE_massflow_MMH_tank,
            "He_massflow_into_MON25_tank": HE_massflow_MON25_tank,
            "He_mass_for_MMH_tank": HE_mass_MMH_tank,
            "He_mass_for_MON25_tank": HE_mass_MON25_tank,
            "He_mass_in_lines": HE_mass_in_lines
            }
        }

def temperature_sweep(t1: UnitValue, t2: UnitValue, t_step: UnitValue, MMH_tank_pressure_guess: UnitValue, MON25_tank_pressure_guess: UnitValue, verbose:bool = False):
    results = {
            "Temperature": [],
            # Masses
            "MMH Tank Mass": [],
            "MON25 Tank Mass": [],
            "HE Tank Mass": [],
            "He Total Mass": [],
            "He Usable Mass": [],
            "He Unusable Mass": [],
            "Total Mass": [],
            # Pressures
            "MMH Operating Pressure": [],
            "MMH Max Pressure": [],
            "MMH Saturation Pressure": [],
            "MON25 Operating Pressure": [],
            "MON25 Max Pressure": [],
            "MON25 Saturation Pressure": [],
            # Tank volumes
            "MMH Inner Volume": [],
            "MMH Outer Volume": [],
            "MMH Fuel Volume": [],
            "MMH Ullage Volume": [],
            "MON25 Inner Volume": [],
            "MON25 Outer Volume": [],
            "MON25 Oxidizer Volume": [],
            "MON25 Ullage Volume": [],
            "HE Inner Volume": [],
            "HE Outer Volume": [],
    
            # Helium breakdown
            "He MMH Tank": [],
            "He MON25 Tank": [],
            "He Lines": [],
            "He MMH Flow": [],
            "He MON25 Flow": []
        }

    t1.to("K")
    t2.to("K")
    t_step.to("K")
    current_temp = t1
    
    while current_temp <= t2:

        print(f"\nCalculating at {current_temp.to('C')}")
        system_result = calculate_system_characterisitcs(current_temp, MMH_tank_pressure_guess, MON25_tank_pressure_guess, verbose=False)

        MMH_result = system_result["MMH TANK"]
        MON25_result = system_result["MON25 TANK"]
        HE_result = system_result["HE TANK"]

        # Helper for extracting numerical values from UnitValue

        # Temperature
        results["Temperature"].append(current_temp.to("C").value)

        # Masses
        results["MMH Tank Mass"].append(MMH_result["Tank Mass"].value)
        results["MON25 Tank Mass"].append(MON25_result["Tank Mass"].value)
        results["HE Tank Mass"].append(HE_result["Tank Mass"].value)

        results["He Total Mass"].append(HE_result["He Total Mass"].value)
        results["He Usable Mass"].append(HE_result["He Usable Mass"].value)
        results["He Unusable Mass"].append(HE_result["He Unusable Mass"].value)

        # Total variable system mass
        total_mass = (
            MMH_result["Tank Mass"]
            + MON25_result["Tank Mass"]
            + HE_result["Tank Mass"]
            + HE_result["He Total Mass"]
        )

        results["Total Mass"].append(total_mass.value)

        # Operating and maximum design pressures
        results["MMH Operating Pressure"].append(MMH_result["Operating Ullage Pressure"].to("psi").value)
        results["MMH Max Pressure"].append(MMH_result["Max Design Tank Pressure"].to("psi").value)
        results["MMH Saturation Pressure"].append(MMH_result["Saturation Pressure"].to("psi").value)

        results["MON25 Operating Pressure"].append(MON25_result["Operating Ullage Pressure"].to("psi").value)
        results["MON25 Max Pressure"].append(MON25_result["Max Design Tank Pressure"].to("psi").value)
        results["MON25 Saturation Pressure"].append(MON25_result["Saturation Pressure"].to("psi").value)

        # Tank volumes
        results["MMH Inner Volume"].append(MMH_result["Inner Volume"].value)
        results["MMH Outer Volume"].append(MMH_result["Outer Volume"].value)
        results["MMH Fuel Volume"].append(MMH_result["Fuel Volume"].value)
        results["MMH Ullage Volume"].append(MMH_result["Ullage Volume"].value)

        results["MON25 Inner Volume"].append(MON25_result["Inner Volume"].value)
        results["MON25 Outer Volume"].append(MON25_result["Outer Volume"].value)
        results["MON25 Oxidizer Volume"].append(MON25_result["Oxidizer Volume"].value)
        results["MON25 Ullage Volume"].append(MON25_result["Ullage Volume"].value)

        results["HE Inner Volume"].append(HE_result["Inner Volume"].value)
        results["HE Outer Volume"].append(HE_result["Outer Volume"].value)

        # Helium mass breakdown
        results["He MMH Tank"].append(HE_result["He_mass_for_MMH_tank"].value)
        results["He MON25 Tank"].append(HE_result["He_mass_for_MON25_tank"].value)
        results["He Lines"].append(HE_result["He_mass_in_lines"].value)

        results["He MMH Flow"].append(HE_result["He_massflow_into_MMH_tank"].value)
        results["He MON25 Flow"].append(HE_result["He_massflow_into_MON25_tank"].value)

        # Increment temperature
        current_temp += t_step


    # Convert results to NumPy arrays
    for key in results:
        results[key] = np.array(results[key])

    if verbose == True:
        T = results["Temperature"]
        # ---------------------------------------------------------
        # PLOT 1: MASS BREAKDOWN
        # ---------------------------------------------------------

        plt.figure(figsize=(11, 7))

        plt.plot(T, results["MMH Tank Mass"], marker="o", label="MMH Tank")
        plt.plot(T, results["MON25 Tank Mass"], marker="o", label="MON25 Tank")
        plt.plot(T, results["HE Tank Mass"], marker="o", label="Helium COPV")

        plt.plot(T, results["He Total Mass"], marker="s", label="Total Helium")
        plt.plot(T, results["Total Mass"], marker="D", linewidth=2.5, label="Total System Mass")

        plt.xlabel("Temperature (°C)")
        plt.ylabel("Mass (kg)")
        plt.title("Propulsion System Mass Breakdown vs System Temperature")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # ---------------------------------------------------------
        # PLOT 2: OPERATING AND MAXIMUM TANK PRESSURES
        # ---------------------------------------------------------

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # MMH
        axes[0].plot(T, results["MMH Operating Pressure"], marker="o", label="Operating Ullage Pressure")
        axes[0].plot(T, results["MMH Max Pressure"], marker="s", label="Maximum Design Pressure")
        axes[0].plot(T, results["MMH Saturation Pressure"], marker="^", label="Saturation Pressure")
        

        axes[0].set_title("MMH Tank Pressure")
        axes[0].set_xlabel("Temperature (°C)")
        axes[0].set_ylabel("Pressure (psi)")
        axes[0].grid(True)
        axes[0].legend()

        # MON25
        axes[1].plot(T, results["MON25 Operating Pressure"], marker="o", label="Operating Ullage Pressure")
        axes[1].plot(T, results["MON25 Max Pressure"], marker="s", label="Maximum Design Pressure")
        axes[1].plot(T, results["MON25 Saturation Pressure"], marker="^", label="Saturation Pressure")

        axes[1].set_title("MON25 Tank Pressure")
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_ylabel("Pressure (psi)")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle("Propellant Tank Pressure vs System Temperature")
        plt.tight_layout()

        # ---------------------------------------------------------
        # PLOT 3: TANK VOLUMES
        # ---------------------------------------------------------

        fig, axes = plt.subplots(1, 3, figsize=(17, 6))

        # MMH
        axes[0].plot(T, results["MMH Inner Volume"], marker="o", label="Inner Volume")
        axes[0].plot(T, results["MMH Outer Volume"], marker="s", label="Outer Volume")
        axes[0].plot(T, results["MMH Fuel Volume"], marker="^", label="Fuel Volume")
        axes[0].plot(T, results["MMH Ullage Volume"], marker="D", label="Ullage Volume")

        axes[0].set_title("MMH Tank")
        axes[0].set_xlabel("Temperature (°C)")
        axes[0].set_ylabel("Volume (m³)")
        axes[0].grid(True)
        axes[0].legend()

        # MON25
        axes[1].plot(T, results["MON25 Inner Volume"], marker="o", label="Inner Volume")
        axes[1].plot(T, results["MON25 Outer Volume"], marker="s", label="Outer Volume")
        axes[1].plot(T, results["MON25 Oxidizer Volume"], marker="^", label="Oxidizer Volume")
        axes[1].plot(T, results["MON25 Ullage Volume"], marker="D", label="Ullage Volume")

        axes[1].set_title("MON25 Tank")
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_ylabel("Volume (m³)")
        axes[1].grid(True)
        axes[1].legend()

        # Helium COPV
        axes[2].plot(T, results["HE Inner Volume"], marker="o", label="Inner Volume")
        axes[2].plot(T, results["HE Outer Volume"], marker="s", label="Outer Volume")

        axes[2].set_title("Helium COPV")
        axes[2].set_xlabel("Temperature (°C)")
        axes[2].set_ylabel("Volume (m³)")
        axes[2].grid(True)
        axes[2].legend()

        fig.suptitle("Tank Volumes vs Temperature")
        plt.tight_layout()

        # ---------------------------------------------------------
        # PLOT 4: HELIUM MASS BREAKDOWN
        # ---------------------------------------------------------

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Overall helium mass breakdown
        axes[0].plot(T, results["He Total Mass"], marker="o", linewidth=2, label="Total Helium")
        axes[0].plot(T, results["He Usable Mass"], marker="s", label="Usable Helium")
        axes[0].plot(T, results["He Unusable Mass"], marker="^", label="Unusable Helium")

        axes[0].set_title("Total Helium Mass")
        axes[0].set_xlabel("Temperature (°C)")
        axes[0].set_ylabel("Mass (kg)")
        axes[0].grid(True)
        axes[0].legend()

        # Allocation of helium
        axes[1].plot(T, results["He MMH Tank"], marker="o", label="MMH Tank Allocation")
        axes[1].plot(T, results["He MON25 Tank"], marker="s", label="MON25 Tank Allocation")
        axes[1].plot(T, results["He Lines"], marker="^", label="Lines")
        axes[1].plot(T, results["He Unusable Mass"], marker="D", label="Left In He Tank")

        axes[1].set_title("Helium Mass Allocation")
        axes[1].set_xlabel("Temperature (°C)")
        axes[1].set_ylabel("Mass (kg)")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle("Helium Mass Breakdown vs Temperature")
        plt.tight_layout()

        # ---------------------------------------------------------
        # PLOT 5: HELIUM MASS FLOW RATES
        # ---------------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(T, results["He MMH Flow"], marker="o", label="MMH Tank")
        plt.plot(T, results["He MON25 Flow"], marker="s", label="MON25 Tank")

        plt.xlabel("Temperature (°C)")
        plt.ylabel("Helium Mass Flow Rate (kg/s)")
        plt.title("Helium Mass Flow Rates vs Temperature")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()

    return results
    

if __name__ == "__main__":

    MMH_boiling = UnitValue.create_unit("R", MMH.prop.Tnbp).convert_base_metric()
    MMH_freezing = UnitValue.create_unit("R", MMH.prop.Tfreeze).convert_base_metric()
    MON25_boiling = UnitValue.create_unit("R", MON25.prop.Tnbp).convert_base_metric()
    MON25_freezing = UnitValue.create_unit("R", MON25.prop.Tfreeze).convert_base_metric()

    print(f'boil: {MMH_boiling.to("C")}, freeze: {MMH_freezing.to("C")}')
    print(f'boil: {MON25_boiling.to("C")}, freeze: {MON25_freezing.to("C")}')

    t1 = UnitValue.create_unit("C", -40.0).to("K") #as per NASA report on MON25 and MMH, at -40 NTO start crystlizing out.
    t2 = UnitValue.create_unit("C", 60.0).to("K")
    step = UnitValue.create_unit("K", 5.0)                                   

    MMH_p = UnitValue.create_unit("MPa", 2.4).convert_base_metric()
    MON25_p = UnitValue.create_unit("MPa", 2.4).convert_base_metric()

    temperature_sweep(t1, t2, step, MMH_p, MON25_p, True) # true to make plots

    calculate_system_characterisitcs(t2, MMH_p, MON25_p, True)

    

                

