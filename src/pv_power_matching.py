import pandas as pd
import numpy as np
import pathlib

from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib import colors

from GA_module_v6 import ga_min

ABS_FILE_PATH = pathlib.Path(__file__).parent.resolve()
FILENAME = pathlib.Path(__file__).stem
OBJECTIVE_FUNCTION = 'objective_function'


def load_characteristic():
    JSON_DATA_FILE_NAME = ABS_FILE_PATH / '..' / 'dat' / \
        'worst_case_power_demand_characteristic.json'
    power_demand = pd.read_json(JSON_DATA_FILE_NAME)
    time_day = power_demand['Time_day'].values
    load_power = power_demand['P_max'].values
    return time_day, load_power


def sun_position_for_time_and_location(time: Time, location: EarthLocation) -> tuple:
    """Calculate the sun position as normal vector that points from the
    location to the suns position in the sky. Supports iterables.

    Args:
        time (Time): Time to calculate the normal for
        location (EarthLocation): Location on earth to calculate the
        normal for

    Returns:
        tuple: 3D np.ndarray sun normal pointing to sun position from
        normal and bool array when sun is above horizon. 
    """
    sun_data = get_sun(time)
    location_frame = AltAz(obstime=time, location=location)
    sun_altitude_azimuth = sun_data.transform_to(location_frame)
    sun_altitude = np.deg2rad(np.array(sun_altitude_azimuth.alt))
    # polar angle is measured from vertical
    sun_polar_angle = np.deg2rad(90) - sun_altitude
    sun_azimuth = -np.deg2rad(np.array(sun_altitude_azimuth.az))
    sun_normal = np.array([np.sin(sun_polar_angle) * np.cos(sun_azimuth),
                          np.sin(sun_polar_angle) * np.sin(sun_azimuth), np.cos(sun_polar_angle)])
    sun_above_horizon = sun_altitude > 0
    return sun_normal, sun_above_horizon, sun_altitude


# Essen coordinates
SYSTEM_LON = 7.014761
SYSTEM_LAT = 51.458069
SYSTEM_ELV = 116.0
# setup:
system_location = EarthLocation(
    lon=SYSTEM_LON, lat=SYSTEM_LAT, height=SYSTEM_ELV)
demand_times, power_demand_characteristic = load_characteristic()
time_start = Time('2024-9-22 00:00:00')
SAMPLES = demand_times.shape[0]
supply_times = time_start + u.hour * np.linspace(0., 24., SAMPLES)
sun_normals, sun_over_horizon_mask, sun_altitude = sun_position_for_time_and_location(
    supply_times, system_location)


def calculate_distance_through_atmosphere(ray_angle: float) -> float:
    """Uses the law of cosine to determine the distance a ray
    hitting the earths surface with ray_angle (rad) has to travel to
    reach the surface. 

    Args:
        ray_angle (float): Angle of incidence of the ray on earth
        surface in rads

    Returns:
        float: travel distance in meters. 
    """
    earth_radius = 6371e3  # m
    stratosphere_height = 20e3  # m

    side_a = earth_radius
    side_c = (earth_radius + stratosphere_height)
    angle_gamma = ray_angle + np.pi/2

    p_coefficient = -2*side_a*np.cos(angle_gamma)
    q_coefficient = side_a**2 - side_c**2

    distance = -p_coefficient/2 + \
        np.sqrt((p_coefficient/2)**2 - (q_coefficient))
    return distance


def influence_atmospheric_transmittance(sun_altitude: float) -> float:
    """Calculates the influence of atmospheric transmittance by assuming
    the transmittance scales with distance traveled through the
    atmosphere. 

    Args:
        sun_altitude (float): Sun altitude in rad

    Returns:
        float: influence fraction. 1 if no influence at all (vertical sunlight)
    """
    atmospheric_travel_distance = calculate_distance_through_atmosphere(
        sun_altitude)
    minimum_distance = calculate_distance_through_atmosphere(np.pi/2)
    transmittance_fraction = minimum_distance/atmospheric_travel_distance
    return transmittance_fraction


def plot_sun_position(sun_altitude, sun_polar_angle, sun_azimuth):
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(nrows=3, ncols=1)
    ax_top.plot(sun_altitude)
    ax_mid.plot(sun_polar_angle)
    ax_bot.plot(sun_azimuth)
    return


def panel_normal_from_tilt_and_az(panel_azimuth: float, panel_tilt: float) -> np.ndarray:
    """Computes the panel surface normal from the panels defining
    orientation angles. Normal is 3D. Supports iterable inputs.

    Args:
        panel_azimuth (float): Panel azimuth measured from north
        panel_tilt (float): Panel tilt measured against horizontal

    Returns:
        np.ndarray: Normal of panel
    """
    panel_normal = np.array([np.sin(panel_tilt) * np.cos(-panel_azimuth),
                            np.sin(panel_tilt) * np.sin(-panel_azimuth), np.cos(panel_tilt)])
    return panel_normal


def cosine_of_incidence_angle(panel_normal: np.ndarray, sun_normal: np.ndarray) -> np.ndarray:
    """Computes the cosine of angle between the two specified vectors
    and thus the effective area fraction of a PV that is orthogonal to
    the sun normal. Supports multidimensional inputs. For this specify
    panel_normal as (3,N) with N the number of panels and sun_normal as
    (3,M) with M as the number of sun normals (this corresponds to
    different times). The result will then be (N,M) with every row n
    containing the data for every panel over all times. 

    Args:
        panel_normal (np.ndarray): Normal of front panel surface
        sun_normal (np.ndarray): Normal pointing to sun position in sky 

    Returns:
        np.ndarray: Area fraction perpendicular to sun rays for all
        panels (rows) and all times (columns)
    """
    cos_theta = (np.dot(panel_normal.T, sun_normal))
    return cos_theta


def plot_normals(sun_normals, panel_normals):
    axis = plt.figure().add_subplot(projection='3d')
    axis.quiver(0, 0, 0, sun_normals[0, ::20],
                sun_normals[1, ::20], sun_normals[2, ::20])
    axis.quiver(
        0, 0, 0, panel_normals[0, :], panel_normals[1, :], panel_normals[2, :], color='red')

    axis.set_xlim([-1, 1])
    axis.set_xlabel('North 1')
    axis.set_ylim([-1, 1])
    axis.set_ylabel('West 1')
    axis.set_zlim([-.5, 1])

    axis.view_init(elev=30, azim=270, roll=0)
    return


def calculate_pv_panel_power(panel_orientation: np.ndarray, PANEL_MAX_POWER: float = 250.0) -> np.ndarray:
    """Calculate the panel power to a specific time, at a certain
    position for a location on earth. Supports multiple times and
    multiple panels. For this provide times as a (1,N) array N being the
    number of times and panel_orientation as a (2,M) array M being the
    number of panels. 

    Args:
        panel_orientation (np.ndarray): panel orientation. Specify
        individual panels as columns where rows are: {0:'azimuth', 1:'tilt'}
        PANEL_MAX_POWER (float, optional): Power of panel under ideal
        conditions in Watt. Defaults to 250.0.

    Returns:
        np.ndarray: PV panels power output.
    """
    panel_azimuth = panel_orientation[0, :]
    panel_tilt = panel_orientation[1, :]
    panel_normals = panel_normal_from_tilt_and_az(panel_azimuth, panel_tilt)
    cos_theta = cosine_of_incidence_angle(panel_normals, sun_normals)
    atmospheric_absorption = influence_atmospheric_transmittance(sun_altitude)
    panel_power_fraction = cos_theta * atmospheric_absorption
    sun_behind_panel_mask = cos_theta < 0
    pv_panel_power = PANEL_MAX_POWER * panel_power_fraction
    pv_panel_power[:, ~sun_over_horizon_mask] = 0
    pv_panel_power[sun_behind_panel_mask] = 0
    return pv_panel_power


def pv_system_power_production_characteristic(
        panel_orientation: np.ndarray) -> np.ndarray:
    """Calculates the power production characteristic for a pv system
    over the specified times. panel_orientation.shape[1] panels make up
    the system. 

    Args:
        panel_orientation (np.ndarray): Panel orientations. Columns are
        panels, rows are {0: 'azimuth', 1: 'tilt'}. Tilt measured
        against horizontal. All angles in rad

    Returns:
        np.ndarray: PV system power characteristic with length of times
    """
    pv_panel_power = calculate_pv_panel_power(
        panel_orientation=panel_orientation)
    pv_system_power = np.sum(pv_panel_power, axis=0)
    return pv_system_power


def mean_absolute_percentage_error(expected: float, predicted: float) -> float:
    """Yields the MAPE of the specified data (sets). Inputs can be arrays.

    Args:
        expected (float): expected data to reference against
        predicted (float): predicted data to reference

    Returns:
        float: MAPE of the data set
    """
    percentage_errors = (expected - predicted)/expected
    absolute_percentage_errors = np.abs(percentage_errors)
    mean_absolute_percentage_error = np.mean(absolute_percentage_errors)
    return mean_absolute_percentage_error


def sum_of_squared_errors(expected: float, predicted: float) -> float:
    """Calculates the sum of squared errors for two data sets

    Args:
        expected (float): expected data
        predicted (float): predicted data

    Returns:
        float: SSE
    """
    errors = expected-predicted
    squared_errors = errors**2
    sum_of_squared_errors = np.sum(squared_errors)
    return sum_of_squared_errors


def map_populations_to_orientation(real_population: np.ndarray, integer_population: np.ndarray) -> np.ndarray:
    """Maps GA populations to panel orientations. The real population
    contains the azimuth and tilt angles of the panels, while the
    integer population specifies the amount of panels that shall be
    used. The mapping works in a way that expects the real population to
    be a horizontal stack of all azimuths and all tilts.

    Args:
        real_population (np.ndarray): Real number population with angles
        of panels
        integer_population (np.ndarray): Integer population with number
        of panels to use. 

    Returns:
        np.ndarray: Trimmed array of panel orientation columns: panels,
        rows: {0='azimuth', 1='tilt'}
    """

    number_of_panels_to_use = integer_population[0]
    new_shape = (2, int(np.max(real_population.shape)/2))
    panel_orientations = np.reshape(
        real_population, shape=new_shape, order='C')
    panel_orientations = panel_orientations[:, :(number_of_panels_to_use)]
    return panel_orientations


def calculate_electricity_cost(power_consumed: np.ndarray, time_step_hours: float, electricity_cost: float=35e-2, electricity_retail_price:float=8e-2) -> float:
    """calculates the electricity cost of a day operating the power
    system and home together. 

    Args:
        power_consumed (np.ndarray): power consumed over the day
        (negative entries mean power has been sold to the grid)
        time_step_hours (float): constant time step for the power array
        electricity_cost (float, optional): Cost of buying electricity
        from the grid in €/kWh. Defaults to 35e-2. 
        electricity_retail_price (float, optional): Retail price for
        selling electricity from the grid in €/kWh. Defaults to 8e-2.

    Returns:
        float: Total cost of electricity. 
    """

    energy_consumed = power_consumed * time_step_hours * 1e-3 # kWh
    buying_energy_mask = energy_consumed > 0
    selling_energy_mask = energy_consumed < 0
    cashflow_energy = np.zeros(energy_consumed.shape)
    cashflow_energy[buying_energy_mask] = energy_consumed[buying_energy_mask] * \
        electricity_cost
    cashflow_energy[selling_energy_mask] = energy_consumed[selling_energy_mask] * \
        electricity_retail_price

    cost = np.sum(cashflow_energy)
    return cost


def objective_function(real_population, integer_population, permutation_population, **kwargs):
    panel_orientations = map_populations_to_orientation(
        real_population, integer_population)
    power_supply_characteristic = pv_system_power_production_characteristic(
        panel_orientation=panel_orientations)
    power_consumed = power_demand_characteristic - power_supply_characteristic
    time_step = (demand_times[1] - demand_times[0]) * 24
    cost = calculate_electricity_cost(
        power_consumed, time_step_hours=time_step, **kwargs)
    sse = sum_of_squared_errors(
        power_demand_characteristic, power_supply_characteristic)
    return cost


def plot_result_characteristics(real_population, integer_population, ax_power:plt.axes=None, title:str="Power characteristics"):
    panel_orientations = map_populations_to_orientation(
        real_population, integer_population)

    power_supply_characteristic = pv_system_power_production_characteristic(
        panel_orientation=panel_orientations)
    if isinstance(ax_power, type(None)):
        fig_chars, ax_power = plt.subplots(nrows=1, ncols=1)
        
    ax_power.plot(demand_times*24, power_demand_characteristic, label='Demand curve')
    ax_power.plot(demand_times*24, power_supply_characteristic,
             label='Production curve')
    ax_power.grid(visible=True, which='both')
    ax_power.set_xlabel("Time [day]")
    ax_power.set_ylabel("Power [W]")
        
    ax_power.set_title(title)
    ax_power.legend()
    
def plot_orientation_histogram(panel_orientation: np.ndarray):
    _, axis_orientation = plt.subplots()
    azimuth_degree = np.rad2deg(panel_orientation[0,:])
    tilt_degree = np.rad2deg(panel_orientation[1,:])
    azimuth_range = [0,360]
    tilt_range = [0,90]
    azimuth_bins = 36
    tilt_bins = 9
    axis_orientation.hist2d(azimuth_degree, tilt_degree, bins=[azimuth_bins, tilt_bins], norm=colors.LogNorm(), range=[azimuth_range, tilt_range])
    axis_orientation.set_xlabel('Panel Azimuth [°]')
    axis_orientation.set_ylabel('Panel Tilt [°]')
    axis_orientation.set_title('Distribution of panel orientations')


def ga_results(Rbest, Ibest, Pbest, PI_best, PI_best_progress):
    print(Ibest)
    print(Rbest)

    # Plot progress
    _, ax_progress = plt.subplots(ncols=1, nrows=1)
    ax_progress.plot(PI_best_progress)
    ax_progress.set_xlabel('Generation')
    ax_progress.set_ylabel('Best score (% target)')

    plot_result_characteristics(Rbest, Ibest)
    panel_orientation = map_populations_to_orientation(
        real_population=Rbest, integer_population=Ibest)
    panel_normals = panel_normal_from_tilt_and_az(
        panel_orientation[0, :], panel_orientation[1, :])
    plot_normals(sun_normals, panel_normals)
    plot_orientation_histogram(panel_orientation)
    plt.show()
    return

def different_costs_run(cost_limits:tuple, price_limits:tuple, num_runs:int=9):
    axis_dimension = int(np.sqrt(num_runs))
    num_runs = axis_dimension*axis_dimension
    cost_axis = np.linspace(cost_limits[0], cost_limits[1], axis_dimension)
    price_axis = np.linspace(price_limits[0], price_limits[1], axis_dimension)
    costs, prices = np.meshgrid(cost_axis, price_axis)
    costs_flat = costs.flatten()
    prices_flat = prices.flatten()
    total_costs = []
    angles = []
    used_panels = []
    histories = []
    
    for index in range(num_runs):
        print(f'Running scenario {index}: Cost {costs_flat[index]}, Price {prices_flat[index]}')
        PI_best, Rbest, Ibest, _, PI_best_progress = optimize_pv_system(num_gen=200, num_pop=1000, verbose=False, electricity_cost=costs_flat[index], electricity_price=prices_flat[index])
        total_costs.append(PI_best)
        angles.append(Rbest)
        used_panels.append(Ibest)
        histories.append(PI_best_progress)
    
    figure_costs, axes_costs = plt.subplots(nrows=axis_dimension, ncols=axis_dimension)
    if num_runs == 1:
        axes_costs = [axes_costs]
        
    for axis, realpop, ipop, cost, price in zip(axes_costs.flatten(), angles, used_panels, costs_flat, prices_flat):
        cus_title = f'Cost: {cost:.3g}€/kWh | Retail price: {price:.3g}€/kWh'
        plot_result_characteristics(realpop, ipop, ax_power=axis, title=cus_title)
        
    figure_costs.suptitle('Power characteristics')
    figure_costs.supxlabel('Electricity cost')
    figure_costs.supylabel('Electricity retail price')
    figure_costs.tight_layout()
    plt.show()
    
    pass



def optimize_pv_system(num_gen:int, num_pop:int, max_num_panels:int=40, verbose:bool=True, electricity_cost:float=35e-2, electricity_price:float=8e-2):
    number_of_generations = num_gen
    number_of_populations = num_pop
    max_number_of_panels = max_num_panels
    number_of_real_variables = max_number_of_panels * 2
    number_of_integer_variables = 1
    number_of_permutation_variables = 0
    azimuth_limit = np.array([[0], [np.deg2rad(360)]])
    tilt_limit = np.array([[0], [np.deg2rad(90)]])
    azimuth_limits = np.repeat(
        azimuth_limit, repeats=max_number_of_panels, axis=1)
    tilt_limits = np.repeat(
        tilt_limit, repeats=max_number_of_panels, axis=1)
    real_variable_limits = np.hstack(
        [azimuth_limits, tilt_limits])  # orientation of panels
    integer_variable_limits = np.array(
        [[0], [max_number_of_panels]]).astype(int)  # number of panels
    tournament_probability = 0.8
    crossover_probability = 0.8
    mutation_probability = 0.075

    size_parameters = np.array([number_of_generations, number_of_populations, number_of_real_variables,
                               number_of_integer_variables, number_of_permutation_variables]).astype(int)
    probability_parameters = np.array(
        [tournament_probability, crossover_probability, mutation_probability])

    PI_best, Rbest, Ibest, Pbest, PI_best_progress = ga_min(
        objective_function, size_parameters, integer_variable_limits, real_variable_limits, probability_parameters, verbose, electricity_cost=electricity_cost, electricity_retail_price=electricity_price)
        
    return PI_best,Rbest,Ibest,Pbest,PI_best_progress

def main():
    # ga_results(np.deg2rad(np.array([60, 60, 300, 60, 60, 300, 60, 60, 300, 60, 60, 300, 60, 60, 300, 60, 60, 300, 80, 40, 60, 80, 40, 60, 80, 40, 60, 80, 40, 60, 80, 40, 60, 80, 40, 60])),np.array([18]), 0,0,0)
    # PI_best, Rbest, Ibest, Pbest, PI_best_progress = optimize_pv_system(num_gen=200, num_pop=1000)
    # ga_results(Rbest, Ibest, Pbest, PI_best, PI_best_progress)
    different_costs_run(cost_limits=[(35-15)*1e-2, (35+15)*1e-2], price_limits=[(8-15)*1e-2, (8+15)*1e-2], num_runs=9)

if __name__ == '__main__':
    main()
