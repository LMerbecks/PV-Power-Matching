import pandas as pd
import numpy as np
import pathlib

ABS_FILE_PATH = pathlib.Path(__file__).parent.resolve()

def load_characteristic():
    JSON_DATA_FILE_NAME = ABS_FILE_PATH / '..' / 'dat' / 'worst_case_power_demand_characteristic.json'
    power_demand = pd.read_json(JSON_DATA_FILE_NAME)
    time_day = power_demand['Time_day'].values
    load_power = power_demand['P_max'].values

    return time_day, load_power

def pv_system_characteristic():
    
    return 0

def mean_absolute_percentage_error(expected, predicted):
    percentage_errors = (predicted - expected)/expected
    absolute_percentage_errors = np.abs(percentage_errors)
    mean_absolute_percentage_error = np.mean(absolute_percentage_errors)
    return mean_absolute_percentage_error

def objective_function(real_population, integer_population, permutation_population):
    times, power_demand_characteristic = load_characteristic()
    power_supply_characteristic = pv_system_characteristic()    
    
    return mean_absolute_percentage_error(power_demand_characteristic, power_supply_characteristic)

if __name__ == '__main__':
    objective_function(real_population=0, integer_population=0, permutation_population=0)