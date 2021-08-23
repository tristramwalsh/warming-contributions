"""Calculate temperatures from PRIMAP emissions."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
# import streamlit as st


def EFmod(nyr, a):
    """Create linear operator to convert emissions to forcing."""
    Fcal = np.zeros((nyr, nyr))

    # extend time array to compute derivatives
    time = np.arange(nyr + 1)

    # compute constant term (if there is one, otherwise a[0]=0)
    F_0 = a[4] * a[13] * a[0] * time

    # loop over gas decay terms to calculate AGWP using AR5 formula
    for j in [1, 2, 3]:
        F_0 = F_0 + a[j] * a[4] * a[13] * a[j+5] * (1 - np.exp(-time / a[j+5]))

    # first-difference AGWP to obtain AGFP
    for i in range(0, nyr):
        Fcal[i, 0] = F_0[i+1]-F_0[i]

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Fcal[j:nyr, j] = Fcal[0:nyr-j, 0]

    return Fcal



def FTmod(nyr, a):
    """Create linear operator to convert forcing to warming."""
    Tcal = np.zeros((nyr, nyr))

    # shift time array to compute derivatives
    time = np.arange(nyr) + 0.5

    # loop over thermal response times using AR5 formula
    for j in [0, 1]:
        Tcal[:, 0] = Tcal[:, 0] + (a[j+10] / a[j+15]) * np.exp(-time / a[j+15])

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]

    return Tcal


def ETmod(nyr, a):
    """Create linear operator to convert emissions to warming."""
    Tcal = np.zeros((nyr, nyr))

    # add one to the time array for consistency with AR5 formulae
    time = np.arange(nyr) + 1

    # loop over thermal response times using AR5 formula for AGTP
    for j in [0, 1]:
        Tcal[:, 0] = Tcal[:, 0] + a[4] * a[13] * \
            a[0] * a[j+10] * (1 - np.exp(-time / a[j+15]))

        # loop over gas decay terms using AR5 formula for AGTP
        for i in [1, 2, 3]:
            Tcal[:, 0] = Tcal[:, 0]+a[4]*a[13]*a[i]*a[i+5]*a[j+10] * \
                (np.exp(-time/a[i+5])-np.exp(-time/a[j+15]))/(a[i+5]-a[j+15])

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]
  
    return Tcal


def a_params(gas):
    """Return the AR5 model parameter sets, in units GtCO2."""

    # First set up AR5 model parameters,
    # using syntax of FaIRv1.3 but units of GtCO2, not GtC
    
    m_atm = 5.1352 * 10**18  # AR5 official mass of atmosphere in kg
    m_air = 28.97 * 10**-3   # AR5 official molar mass of air
    # m_car = 12.01 * 10**-3   # AR5 official molar mass of carbon
    m_co2 = 44.01 * 10**-3   # AR5 official molar mass of CO2
    m_ch4 = 16.043 * 10**-3  # AR5 official molar mass of methane
    m_n2o = 44.013 * 10**-3  # AR5 official molar mass of nitrous oxide

    # scl = 1 * 10**3
    a_ar5 = np.zeros(20)

    # Set to AR5 Values for CO2
    a_ar5[0:4] = [0.21787, 0.22896, 0.28454, 0.26863]  # AR5 carbon cycle coefficients
    a_ar5[4] = 1.e12 * 1.e6 / m_co2 / (m_atm / m_air)  # old value = 0.471 ppm/GtC # convert GtCO2 to ppm
    a_ar5[5:9] = [1.e8, 381.330, 34.7850, 4.12370]     # AR5 carbon cycle timescales
    a_ar5[10:12] = [0.631, 0.429]                      # AR5 thermal sensitivity coeffs
    a_ar5[13] = 1.37e-2                                # AR5 rad efficiency in W/m2/ppm
    a_ar5[14] = 0
    a_ar5[15:17] = [8.400, 409.5]                      # AR5 thermal time-constants -- could use Geoffroy et al [4.1,249.] 
    a_ar5[18:21] = 0

    ECS = 3.0
    a_ar5[10:12] *= ECS / np.sum(a_ar5[10:12]) / 3.7  # Rescale thermal sensitivity coeffs to prescribed ECS

    # Set to AR5 Values for CH4
    a_ch4 = a_ar5.copy()
    a_ch4[0:4] = [0, 1.0, 0, 0]
    a_ch4[4] = 1.e12 * 1.e9 / m_ch4 / (m_atm / m_air)  # convert GtCH4 to ppb
    a_ch4[5:9] = [1, 12.4, 1, 1]                       # methane lifetime
    a_ch4[13] = 1.65 * 3.6324360e-4                    # Adjusted radiative efficiency in W/m2/ppb

    # Set to AR5 Values for N2O
    a_n2o = a_ar5.copy()
    a_n2o[0:4] = [0, 1.0, 0, 0]
    a_n2o[4] = 1.e12 * 1.e9 / m_n2o / (m_atm / m_air)         # convert GtN2O to ppb
    a_n2o[5:9] = [1, 121., 1, 1]                              # N2O lifetime
    a_n2o[13] = (1.-0.36 * 1.65 * 3.63e-4 / 3.0e-3) * 3.0e-3  # Adjusted radiative efficiency in W/m2/ppb

    if gas == 'CO2':
        return a_ar5
    elif gas == 'CH4':
        return a_ch4
    elif gas == 'N2O':
        return a_n2o
    else:
        print(f'WARNING: {gas} is not a recognised gas')



def emissions_units(PR_input):
    """Returns pandas DataFrame of entities (gases) and unit (units)."""
    entities = np.array(PR_input['entity'])
    units = np.array(PR_input['unit'])
    entity_unit = {}
    for A, B in zip(entities, units):
        entity_unit[A] = B
    emissions_units = pd.DataFrame(data=entity_unit, index=[0])
    # emissions_units = pd.DataFrame(data = {'entity': entities, 'unit': units})
    return emissions_units.T

# @st.cache
def load_data(file):
    return pd.read_csv(file)

# Load PRIMAP DataSet
PR_input = load_data('./data/PRIMAP-hist_v2.2_19-Jan-2021.csv')
# print(PR_input.info())
# st.write(PR_input)

# PR_country = ['EU28', 'USA', 'AOSIS']
# PR_category = ['IPC1', 'IPC2', 'IPCMAG', 'IPC4']
# PR_entity = ['CO2', 'CH4', 'N2O']
# PR_scenario = ['HISTCR']


PR_scenario = list(set(PR_input['scenario']))
PR_country = list(set(PR_input['country']))
PR_category = list(set(PR_input['category']))
PR_entity = ['CO2', 'CH4', 'N2O']


# # Abridged for testing
# PR_country = ['EU28']
# PR_category = ['IPC1', 'IPC2', 'IPCMAG', 'IPC4']
# # PR_category = ['IPCM0EL']
# PR_entity = ['CO2', 'CH4', 'N2O']
# PR_scenario = ['HISTCR']

PR_temp = pd.DataFrame(columns=list(PR_input.columns.values))
# st.write(PR_temp)


# Set up timeseries index
start_year = 1850
end_year = 2018
ny = end_year - start_year + 1
tim2 = np.arange(ny) + 1

# the GWP_100 factors for [CO2, CH4, N2O] respectively
gwp = {'CO2': 1., 'CH4': 28., 'N2O': 265.}
# labels=['Carbon dioxide','Methane','Nitrous Oxide','Total']

PR_year = np.arange(2018 - 1850 + 1) + 1850
PR_ghgs = np.zeros([2018 - 1850 + 1, len(PR_category), len(PR_entity)])
PR_co2e = np.zeros([2018 - 1850 + 1, len(PR_category), len(PR_entity)])

t1 = dt.datetime.now()
i = 1
number_of_series = (len(PR_scenario) * len(PR_country) *
                    len(PR_category) * len(PR_entity))

for scenario in PR_scenario:
    for country in PR_country:
        for category in PR_category:
            for entity in PR_entity:
                # Visually show how far through the calculation we are
                name = f'{scenario}, {country}, {category}, {entity}'
                percentage = int(i/number_of_series*100)
                loading_bar = percentage // 2 * '.'
                print(f'\r{percentage}% {loading_bar} {name}', end='')
                i += 1
                
                individual_timeseries = PR_input[
                    (PR_input['scenario'] == scenario) &
                    (PR_input['country'] == country) &
                    (PR_input['category'] == category) &
                    (PR_input['entity'] == entity)
                    ].transpose().loc['1850':].values.squeeze() / 1.e6

                # Calculate the warming impact from the individual_series
                temp = ETmod(ny, a_params(entity)) @ individual_timeseries



t2 = dt.datetime.now()
print(f'\nComputation time: {t2-t1}')
print(f'Expected total computation time: {(t2-t1) * PR_input.shape[0] / number_of_series}')