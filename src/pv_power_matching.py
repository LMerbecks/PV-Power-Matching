import pandas as pd
import numpy as np
import pathlib

from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

ABS_FILE_PATH = pathlib.Path(__file__).parent.resolve()


def load_characteristic():
    JSON_DATA_FILE_NAME = ABS_FILE_PATH / '..' / 'dat' / \
        'worst_case_power_demand_characteristic.json'
    power_demand = pd.read_json(JSON_DATA_FILE_NAME)
    time_day = power_demand['Time_day'].values
    load_power = power_demand['P_max'].values
    return time_day, load_power


def sun_position_for_time_and_location(time: Time, location: EarthLocation) -> np.ndarray:
    """Calculate the sun position as normal vector that points from the
    location to the suns position in the sky. Supports iterables.

    Args:
        time (Time): Time to calculate the normal for
        location (EarthLocation): Location on earth to calculate the
        normal for

    Returns:
        np.ndarray: 3D sun normal pointing to sun position from normal
    """
    sun_data = get_sun(time)
    location_frame = AltAz(obstime=time, location=location)
    sun_altitude_azimuth = sun_data.transform_to(location_frame)
    sun_altitude = np.deg2rad(90 - np.array(sun_altitude_azimuth.alt))
    sun_azimuth = np.deg2rad(np.array(sun_altitude_azimuth.az))
    sun_normal = np.array([np.sin(sun_altitude) * np.cos(sun_azimuth),
                          np.sin(sun_altitude) * np.sin(sun_azimuth), np.cos(sun_altitude)])
    return sun_normal


def panel_normal_from_tilt_and_az(panel_azimuth: float, panel_tilt: float) -> np.ndarray:
    """Computes the panel surface normal from the panels defining
    orientation angles. Normal is 3D. Supports iterable inputs.

    Args:
        panel_tilt (float): Panel tilt measured against horizontal
        panel_azimuth (float): Panel azimuth measured from north

    Returns:
        np.ndarray: Normal of panel
    """
    panel_normal = np.array([np.sin(panel_tilt) * np.cos(panel_azimuth),
                            np.sin(panel_tilt) * np.sin(panel_azimuth), np.cos(panel_tilt)])
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


def calculate_pv_panel_power(times: Time, location: EarthLocation, panel_orientation: np.ndarray, PANEL_MAX_POWER: float = 250.0) -> np.ndarray:
    """Calculate the panel power to a specific time, at a certain
    position for a location on earth. Supports multiple times and
    multiple panels. For this provide times as a (1,N) array N being the
    number of times and panel_orientation as a (2,M) array M being the
    number of panels. 

    Args:
        times (Time): time to calculate power at
        location (EarthLocation): position of the panel on earth
        panel_orientation (np.ndarray): panel orientation. Specify
        individual panels as columns where rows are: {0:'azimuth', 1:'tilt'}
        PANEL_MAX_POWER (float, optional): Power of panel under ideal
        conditions in Watt. Defaults to 250.0.

    Returns:
        np.ndarray: PV panels power output.
    """
    panel_azimuth = panel_orientation[0, :]
    panel_tilt = panel_orientation[1, :]
    sun_normals = sun_position_for_time_and_location(times, location)
    panel_normals = panel_normal_from_tilt_and_az(panel_tilt, panel_azimuth)
    cos_theta = cosine_of_incidence_angle(panel_normals, sun_normals)
    # TODO: Here we could add another factor for atmospheric absorption.
    atmospheric_absorption = 1
    panel_power_fraction = cos_theta * atmospheric_absorption
    pv_panel_power = PANEL_MAX_POWER * panel_power_fraction
    return pv_panel_power


def pv_system_power_production_characteristic(
        panel_orientation: np.ndarray,
        system_location: EarthLocation,
        times: Time) -> np.ndarray:
    """Calculates the power production characteristic for a pv system
    over the specified times. panel_orientation.shape[1] panels make up
    the system. 

    Args:
        panel_orientation (np.ndarray): Panel orientations. Columns:
        panels, rows: {0: 'azimuth', 1: 'tilt'}. Tilt measured against horizontal.
        system_location (EarthLocation): System location (on earth)
        times (Time): Times to calculate characteristic for

    Returns:
        np.ndarray: PV system power characteristic with length of times
    """
    pv_panel_power = calculate_pv_panel_power(
        times=times, location=system_location, panel_orientation=panel_orientation)
    pv_system_power = np.sum(pv_panel_power)
    return pv_system_power


def mean_absolute_percentage_error(expected: float, predicted: float) -> float:
    """Yields the MAPE of the specified data (sets). Inputs can be arrays.

    Args:
        expected (float): expected data to reference against
        predicted (float): predicted data to reference

    Returns:
        float: MAPE of the data set
    """
    percentage_errors = (predicted - expected)/expected
    absolute_percentage_errors = np.abs(percentage_errors)
    mean_absolute_percentage_error = np.mean(absolute_percentage_errors)
    return mean_absolute_percentage_error


def map_populations_to_orientation(real_population: np.ndarray, integer_population: np.ndarray) -> np.ndarray:
    """Maps GA populations to panel orientations. The real population
    contains the azimuth and tilt angles of the panels, while the
    integer population specifies the amount of panels that shall be
    used. 

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
    new_shape = (2, int(real_population/2))
    panel_orientations = np.reshape(
        real_population, shape=new_shape, order='F')
    panel_orientations = panel_orientations[:, :(number_of_panels_to_use-1)]
    return panel_orientations


def objective_function(real_population, integer_population, permutation_population):
    # Essen coordinates
    SYSTEM_LON = 51.458069
    SYSTEM_LAT = 7.014761
    SYSTEM_ELV = 116.0

    panel_orientations = map_populations_to_orientation(
        real_population, integer_population)

    system_location = EarthLocation(
        lon=SYSTEM_LON, lat=SYSTEM_LAT, height=SYSTEM_ELV)
    demand_times, power_demand_characteristic = load_characteristic()
    time_start = Time('2024-9-22 00:00:00')
    SAMPLES = demand_times.shape[0]
    supply_times = time_start + u.hour * np.linspace(0., 24., SAMPLES)
    power_supply_characteristic = pv_system_power_production_characteristic(
        panel_orientation=panel_orientations, system_location=system_location, times=supply_times)

    return mean_absolute_percentage_error(power_demand_characteristic, power_supply_characteristic)


if __name__ == '__main__':
    objective_function(real_population=0, integer_population=0,
                       permutation_population=0)
