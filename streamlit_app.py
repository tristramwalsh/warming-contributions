"""Streamlit web app calculating and exploring contributions to warming."""
import datetime as dt
import io
import csv

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import altair as alt
import seaborn as sns
import plotly.graph_objects as go
import pycountry


st.set_page_config(
    page_title="Warming Contributions Explorer",
    page_icon="https://images.emojiterra.com/twitter/512px/1f321.png",
    layout="wide"
)

# alt.renderers.set_embed_options(actions=False)
# Without this following text it isn't possible to see hover tooltips in the
# fullscreen version of the plot
# https://discuss.streamlit.io/t/tool-tips-in-fullscreen-mode-for-charts/6800/10
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
            unsafe_allow_html=True)

# This next piece prevents the altair three dot menu from appearing.
# source:
# https://discuss.streamlit.io/t/does-altairs-set-embed-options-work-with-streamlit/1675
# Note, these lines broke the st.expander feature, so I have commented it out
# as a quick fix, as the expanders are an essential part of the app design.
# st.markdown("""
#     <style type='text/css'>
#         details {
#             display: none;
#         }
#     </style>
# """, unsafe_allow_html=True)

# Add custom CSS to hide the hamburger menu, 'made with streamlit' footer, and
# github icon.
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2
# also this:
# https://discuss.streamlit.io/t/how-do-i-hide-remove-the-menu-in-production/362/7
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Disable opacity fading every time a widget is changed.
# https://discuss.streamlit.io/t/disable-reloading-of-image-every-time-a-widget-is-updated/1612/4
st.markdown("<style>.element-container{opacity:1 !important}</style>",
            unsafe_allow_html=True)


st.markdown(
    """
    # Contributions to Global Warming
    *across scenarios, countries, sectors, and the main gases*
    """
)


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


# @numba.jit(nopython=True, parallel=True)  # This didn't help
# @st.cache(show_spinner=False)
# @cache
def ETmod(nyr, a):
    """Create linear operator to convert emissions to warming."""
    Tcal = np.zeros((nyr, nyr))

    # add one to the time array for consistency with AR5 formulae
    time = np.arange(nyr) + 1

    # loop over thermal response times using AR5 formula for AGTP
    # for j in numba.prange(2):
    for j in [0, 1]:
        Tcal[:, 0] = Tcal[:, 0] + a[4] * a[13] * \
            a[0] * a[j+10] * (1 - np.exp(-time / a[j+15]))

        # loop over gas decay terms using AR5 formula for AGTP
        # for i in numba.prange(1, 4, 1):
        for i in [1, 2, 3]:
            Tcal[:, 0] = Tcal[:, 0]+a[4]*a[13]*a[i]*a[i+5]*a[j+10] * \
                (np.exp(-time/a[i+5])-np.exp(-time/a[j+15]))/(a[i+5]-a[j+15])

    # build up the rest of the Toeplitz matrix
    # for j in numba.prange(1, nyr):
    for j in range(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]

    return Tcal


# @numba.jit(nopython=True)  # This didn't help...
# @st.cache(show_spinner=False)
# @cache
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

    if gas == 'Carbon Dioxide':
        return a_ar5
    elif gas == 'Methane':
        return a_ch4
    elif gas == 'Nitrous Oxide':
        return a_n2o


def PR_emissions_units(df):
    """Return pandas DataFrame of entities (gases) and unit (units)."""
    entities = np.array(df['entity'])
    units = np.array(df['unit'])
    entity_unit = {}
    for A, B in zip(entities, units):
        entity_unit[A] = B
    PR_emissions_units = pd.DataFrame(data=entity_unit, index=[0])

    return PR_emissions_units.T


def adjusted_scientific_notation(val, letter, num_decimals=2, exponent_pad=2):
    """Return number in scientific form."""
    exponent_template = "{:0>%d}" % exponent_pad
    mantissa_template = "{:.%df}" % num_decimals

    if val != 0:  # Catch issue handling 0 emissions.
        order_of_magnitude = np.floor(np.log10(abs(val)))
    else:
        order_of_magnitude = 0
    nearest_lower_third = 3*(order_of_magnitude//3)
    adjusted_mantissa = val*10**(-nearest_lower_third)
    adjusted_mantissa_string = mantissa_template.format(adjusted_mantissa)
    # if nearest_lower_third < 0:
    #     sign = '-'
    # elif nearest_lower_third >=0:
    #     sign = '+'
    sign = '-' if nearest_lower_third < 0 else '+'
    # adjusted_exponent_string = (
    #                             "+-"[nearest_lower_third < 0] +
    #                             exponent_template.format(
    #                                 abs(nearest_lower_third))
    #                             )
    adjusted_exponent_string = (sign + exponent_template.format(
        abs(nearest_lower_third))
                                )
    if letter:
        names = {'-12.0': ' p', '-9.0': ' n', '-6.0': ' \u03BC', '-3.0': ' m',
                 '+00': ' ', '+0.0': ' ', '+3.0': ' k', '+6.0': ' M',
                 '+9.0': ' G', '+12.0': ' T', '+15.0': ' P', '+18.0': ' E'}
        return adjusted_mantissa_string + names[adjusted_exponent_string]
    else:
        return adjusted_mantissa_string+"E"+adjusted_exponent_string


@st.cache_data(show_spinner=False)
def load_data(file):
    """Load the dataset, and rename codes with human-friendly terms."""
    # NOTE: function approach allows streamlit caching,
    loading_message = st.sidebar.text('please wait while data initialises')

    df = pd.read_csv(file)

    # Rename labels in PRIMAP 2.3.1 to match those in PRIMAP 2.2
    df.rename({'area (ISO3)': 'country',
               'scenario (PRIMAP-hist)': 'scenario',
               'category (IPCC2006_PRIMAP)': 'category'},
              axis='columns', inplace=True)

    not_country = ['EARTH', 'ANNEXI', 'NONANNEXI', 'AOSIS',
                   'BASIC', 'EU27BX', 'LDC', 'UMBRELLA']
    iso_country = list(set(df['country']) - set(not_country) - set(['ANT']))

    country_names = {x: pycountry.countries.get(alpha_3=x).name
                     for x in iso_country}
    country_names['ANT'] = 'Netherlands Antilles'
    # NB 'ANT' is handled separately as this isn't included in pycountry.

    # Note, the space at the beginning of the long names below is used so that
    # these group-style regions appear first in the multiselect box. It doesn't
    # affect display in plots.
    country_names.update(
        {'EARTH': ' Aggregated emissions for all countries',
         'ANNEXI': ' Annex I Parties to the Convention',
         'NONANNEXI': ' Non-Annex I Parties to the Convention',
         'AOSIS': ' Alliance of Small Island States',
         'BASIC': ' BASIC countries (Brazil, South Africa, India, and China)',
         'EU27BX': ' European Union post Brexit',
         'LDC': ' Least Developed Countries',
         'UMBRELLA': ' Umbrella Group'}
    )

    df['country'] = df['country'].replace(country_names)

    scenario_names = {'HISTCR': 'Prioritise Country-Reported Data',
                      'HISTTP': 'Prioritise Third-Party Data'}
    df['scenario'] = df['scenario'].replace(scenario_names)

    category_names = {
        '0': '0: Total',
        'M.0.EL': '0: Total excluding LULUCF',
        '1': '1: Energy',
        '1.A': '1A: Fuel Combustion Activities',
        '1.B': '1B: Fugitive Emissions from Fuels',
        '1.B.1': '1B1: Solid Fuels',
        '1.B.2': '1B2: Oil and Natural Gas',
        '1.B.3': '1B3: Other Emissions from Energy Production',
        '1.C': '1C: Carbon Dioxide Transport and Storage',
        '2': '2: Industrial Processes and Product Use (IPPU)',
        '2.A': '2A: Mineral Industry',
        '2.B': '2B: Chemical Industry',
        '2.C': '2C: Metal Industry',
        '2.D': '2D: Non-Energy Products from Fuels and Solvent Use',
        '2.E': '2E: Electronics Industry',
        '2.F': '2F: Product uses as Substitutes for Ozone Depleting\
                Substances',
        '2.G': '2G: Other Product Manufacture and Use',
        '2.H': '2H: Other',
        '3': '3: Agriculture, Forestry, and Other Land Use',
        '3.A': '3A: Livestock',
        'M.AG': '3: Agriculture, sum of 3A and 3B',
        'M.AG.ELV': '3B: Agriculture excluding Livestock',
        'M.LULUCF': '3C: Land Use, Land Use Change, and Forestry',
        '4': '4: Waste',
        '5': '5: Other'}
    df['category'] = df['category'].replace(category_names)

    gas_names = {
        'CO2': 'Carbon Dioxide', 'CH4': 'Methane', 'N2O': 'Nitrous Oxide'
    }
    df['entity'] = df['entity'].replace(gas_names)

    loading_message.empty()

    return df


@st.cache_data(show_spinner=False)
def calc(df, scenarios, countries, categories, entities, baseline,
         emissions_units, future_toggle,
         future_co2_zero_year, future_ch4_rate, future_n2o_rate):
    """Calculate warming impact, and emissions metrics, for given selection."""

    # the GWP_100 factors for [CO2, CH4, N2O] respectively
    gwp = {'Carbon Dioxide': 1., 'Methane': 28., 'Nitrous Oxide': 265.}

    # Collect the emissions timeseries that we want to calculate warming from
    emis_to_calculate = df[
                    (df['scenario'] == scenarios) &
                    (df['country'].isin(countries)) &
                    (df['category'].isin(categories)) &
                    (df['entity'].isin(entities))]

    # Prepare the virtual csv files to output calculated things to
    column_names = ['scenario', 'country', 'category', 'entity', 'unit']
    ny = future_co2_zero_year - yr0 + 1 if future_toggle else yr1 - yr0 + 1
    PR_year = np.arange(ny) + yr0
    column_names.extend([str(i) for i in PR_year])

    # For the following simple climate model calculations, it is far more
    # performant to write calculation results to a virtual in-memory CSV and
    # read that CSV into pandas after all calculations are complete, compared
    # with writing over an existing pandas dataframe, or creating a new one "on
    # the fly".

    # Create in memory virtual csv to write temperatures to
    output_T = io.StringIO()
    csv_writer_T = csv.DictWriter(output_T, fieldnames=column_names)
    csv_writer_T.writeheader()

    # Create in memory virtual csv to write emissions to
    output_E = io.StringIO()
    csv_writer_E = csv.DictWriter(output_E, fieldnames=column_names)
    csv_writer_E.writeheader()

    t1 = dt.datetime.now()

    # Invert linear model to calculate CO2-forcing-equivalent emissions.
    fe_matrix = np.linalg.inv(EFmod(ny, a_params('Carbon Dioxide')))

    i = 1
    number_of_series = len(countries) * len(categories) * len(entities)
    # (Update to st.cache)
    calc_text = st.sidebar.empty()
    calc_text.text('Calculating...')
    # progress_text = "Calculating..."
    # calc_bar = st.sidebar.progress(0, text=progress_text)

    for country in countries:
        for category in categories:
            for entity in entities:

                # Visually show how far through the calculation we are
                percentage = int(i/number_of_series*100)
                loading_bar = percentage // 5*'|' + (20 - percentage // 5)*'.'
                calc_text.text(f'calculating {loading_bar} {percentage}% ')
                # calc_bar.progress(percentage, text=progress_text)
                i += 1

                df_timeseries = emis_to_calculate[
                    (emis_to_calculate['country'] == country) &
                    (emis_to_calculate['category'] == category) &
                    (emis_to_calculate['entity'] == entity)
                    ].transpose().loc[str(yr0):]

                # NOTE: PRIMARP doesn't have emissions timeseries for all
                # combinations of scenario, country, category, entity.
                # Thereofore proceed only `if` this data is present in PRIMAP
                # Crucially, not doing this leads to entries of different data
                # types in the year columns, which prevents pandas doing number
                # operations on them later on down the line!
                if not df_timeseries.empty:
                    # FIRST compute temperatures for warming virtual csv
                    arr_timeseries = df_timeseries.values.squeeze() / 1.e6

                    if future_toggle:
                        fut_yrs = np.arange(yr1 + 1, future_co2_zero_year + 1)
                        if entity == 'Carbon Dioxide':
                            arr_timeseries = np.append(
                                arr_timeseries,
                                np.linspace(arr_timeseries[-1], 0,
                                            future_co2_zero_year-yr1)
                                )

                        elif entity == 'Methane':
                            arr_timeseries = np.append(
                                arr_timeseries,
                                np.array([arr_timeseries[-1] *
                                          (1 + future_ch4_rate)**i
                                          for i in fut_yrs-yr1])
                                )
                        elif entity == 'Nitrous Oxide':
                            arr_timeseries = np.append(
                                arr_timeseries,
                                np.array([arr_timeseries[-1] *
                                         (1 + future_n2o_rate)**i
                                         for i in fut_yrs-yr1])
                                )

                    # Calculate the warming impact from the individual_series
                    temp = ETmod(ny, a_params(entity)) @ arr_timeseries

                    # PICK BETWEEN THE TWO METHODS OF BASELINING TEMPERATURES
                    if 'Absolute Temperature Change' in temp_calc_method:
                        # The temperature profile above is the temperature
                        # required.

                        # Apply the baseline subtraction
                        if '-' in baseline:
                            filter = np.array(
                                [0 if (int(baseline.split('-')[0]) <= int(y) <
                                       int(baseline.split('-')[1]))
                                 else 1
                                 for y
                                 in (np.arange(len(temp)) + 1750)])
                            pre_ind_temp = np.ma.masked_array(
                                temp, mask=filter).mean()

                        elif '-' not in baseline:
                            filter = np.array(
                                [0 if int(y) == int(baseline)
                                 else 1
                                 for y
                                 in (np.arange(len(arr_timeseries)) + 1750)])
                            pre_ind_temp = np.ma.masked_array(
                                temp, mask=filter).mean()

                        temp = temp - pre_ind_temp

                    elif 'Relative Temperature Change' in temp_calc_method:
                        # Here we calculate the reative temperature response,
                        # defined as the difference between the temperature
                        # of the "real world" and the temperature of a
                        # counterfactual world where emissions stop in baseline
                        # year. For the period before the baseline, we
                        # filter = np.array([1 if int(x) < int(baseline) else 0
                        #                    for x in df_timeseries.index])
                        filter = np.array(
                            [1 if int(x) < int(baseline) else 0
                             for x in (np.arange(len(arr_timeseries)) + yr0)])
                        arr_timeseries2 = arr_timeseries * filter

                        # temp_c stands for temperature in counterfactual world
                        temp_c = ETmod(ny, a_params(entity)) @ arr_timeseries2

                        # In the first approach below, we simply call relative
                        # emissions 0 in the years before the baseline.
                        # temp = temp-temp_c

                        # In this second approach option, backtrack before the
                        # baseline year with the absolute historical
                        # temperature response
                        temp = ((temp-temp_c) +
                                filter * (temp - temp[int(baseline) - yr0]))

                    # Create dictionary with the new temp data in
                    new_row_T = {'scenario': scenarios,
                                 'country': country,
                                 'category': category,
                                 'entity': entity,
                                 'unit': 'K'}
                    new_row_T.update({str(PR_year[i]): temp[i]
                                     for i in range(len(temp))})
                    csv_writer_T.writerow(new_row_T)

                    ####################################################
                    # SECOND compute emissions for emissions virtual csv
                    if emissions_units == 'GWP100':
                        EMIS = arr_timeseries * gwp[entity]
                        unit = 'GWP GtC CO2-e yr-1'
                    elif emissions_units == 'CO2-fe':
                        EMIS = (fe_matrix
                                @ EFmod(ny, a_params(entity))
                                @ arr_timeseries)
                        unit = 'GtCO2-fe yr-1'
                    elif emissions_units == 'Absolute Mass':
                        EMIS = arr_timeseries
                        unit = 'Gt yr-1'
                    new_row_E = {'scenario': scenarios,
                                 'country': country,
                                 'category': category,
                                 'entity': entity,
                                 'unit': unit}
                    new_row_E.update({str(PR_year[i]): EMIS[i]
                                      for i in range(len(EMIS))})
                    # Write this dictionary to the in-memory csv file.
                    csv_writer_E.writerow(new_row_E)

    t2 = dt.datetime.now()
    calc_text.text(f'New calculation took: {(t2-t1)}')
    # calc_bar.text(f'New calculation took: {(t2-t1)}')

    output_T.seek(0)  # we need to get back to the start of the StringIO
    df_T = pd.read_csv(output_T)
    output_E.seek(0)  # we need to get back to the start of the StringIO
    df_E = pd.read_csv(output_E)

    # Just to reassure myself that data isn't leaking between app user
    # interactions (and ending up with data duplications, if each subsequent
    # data selection adds on top of the existing StringIO in memory virtual
    # csv):
    output_T = io.StringIO()
    output_E = io.StringIO()

    return df_T, df_E


@st.cache_data(show_spinner=False)
def prepare_data(df, scenarios, countries, categories, entities,
                 dis_aggregation, date_range, include_total):
    """Group, time-slice, and calculate sum as required."""
    data = df[(df['scenario'] == scenarios) &
              (df['country'].isin(countries)) &
              (df['category'].isin(categories)) &
              (df['entity'].isin(entities))]

    # Group data
    grouped_data = data.groupby(dis_aggregation).sum(numeric_only=False)

    # Restrict the data to just that selected by the data_range slider
    grouped_data = grouped_data.loc[:, str(date_range[0]):str(date_range[1])]

    # Add 'SUM' of data to data, if there are multiple lines
    if grouped_data.shape[0] > 1 and include_total is True:
        grouped_data = grouped_data.T
        grouped_data["SUM"] = grouped_data.sum(axis=1)
        grouped_data = grouped_data.T

    return grouped_data


def colour_range(domain, include_total, variable):
    """Create colormap with 'SUM' black."""
    if variable != 'entity':
        if variable == 'country':
            cm = 'pastel'
            # cm = 'bright'
            # cm = 'winter'
            # cm= 'rainbow'
        elif variable == 'category':
            cm = 'Set2'
            # cm = 'autumn'
            # cm = "pastel"
        else:
            print('PANIC')

        if len(domain) > 1 and include_total is True:
            # domain = list(grouped_data.index)
            # colour_map = plt.get_cmap(cm)
            # cols = np.array(list(iter(
            #   colour_map(np.linspace(0, 1, len(domain)-1)))))
            cols = np.array(sns.color_palette(cm, len(domain)-1))
            cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
                        for i in range(cols.shape[0])]
            # Final item in index is 'SUM'; add 'black' to range for this
            cols_hex.append('#000000')

        else:
            cols = np.array(sns.color_palette(cm, len(domain)))
            cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
                        for i in range(cols.shape[0])]

    elif variable == 'entity':
        # Handle separately so that, unlike above flexible handling (for now),
        # each gas has a signature colour that doesn't change depending on
        # which gases are selected.

        # cm = 'flare'
        # cm = 'cool'
        cm = 'viridis'
        # cm = 'muted'
        entity_cols = {
            'Carbon Dioxide': '#3b528b',
            'Methane': '#21918c',
            'Nitrous Oxide': '#5ec962',
            'SUM': '#000000'
        }
        cols_hex = [entity_cols[gas] for gas in domain]

    return cols_hex


# st.sidebar.write('## Make a selection')

####
# LOAD DATA
####

st.sidebar.markdown('# Select Data to Explore')

side_expand = st.sidebar.expander('Customise Calculation')
climate_model = side_expand.selectbox(
    'Choose warming model', ['IPCC AR5 Linear Impulse Response Model'], 0)

# PR_url = ('https://zenodo.org/record/4479172/files/' +
#           'PRIMAP-hist_v2.2_19-Jan-2021.csv')
# PR_url = ('https://zenodo.org/record/5494497/files/' +
#           'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_20-Sep_2021.csv')
# The no_rounding dataset variant includes LULUCF, which we manually in/exclude

# using the LULUCF selection in the sidebar:
# PR_url = ('https://zenodo.org/record/5494497/files/' +
#           'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_no_rounding_20-Sep_2021.csv')
PR_url = ('https://zenodo.org/record/7179775/files/' +
          'Guetschow-et-al-2022-PRIMAP-hist_v2.4_no_rounding_11-Oct-2022.csv')

df = load_data(PR_url)
PRIMAP_years = [int(x) for x in df.columns if x.isdigit()]
yr0 = min(PRIMAP_years)
yr1 = max(PRIMAP_years)

####
# WIDGETS FOR USERS TO SELECT DATA
####
scenarios = side_expand.selectbox(
    "Choose historical emissions prioritisation",
    list(set(df['scenario'])),
    # index=list(set(df['scenario'])).index('HISTCR')
    index=list(set(df['scenario'])).index('Prioritise Third-Party Data')
)

LULUCF = side_expand.selectbox(
    "Choose whether to enable LULUCF emissions",
    ['Include LULUCF', 'Do Not Include LULUCF'], 1,
    help='Only use with caution; see PRIMAP-hist documentation for details')

temp_calc_method = side_expand.selectbox(
    "Choose temperature calculation method",
    ['Absolute Temperature Change (from Preindustrial Baseline)',
     'Relative Temperature Change (to Emissions Stopping in Baseline Year)'],
    0,
    help=('"Absolute": warming relative to preindustrial baseline period. ' +
          '"Relative": warming relative to emissions halting in baseline ' +
          'year. For more please scoll down for the main text.')
)

emissions_units = side_expand.selectbox(
    "Choose units to display emissions in",
    ['Absolute Mass', 'GWP100', 'CO2-fe'], 1,
    help='For details please scoll down for the main text.'
)

future_expand = st.sidebar.expander('Future Emissions')
future_toggle = future_expand.checkbox('Explore Future Projections?',
                                       value=False,
                                       help='Explore a range of net zero\
                                             emission-reduction pathways.')
if future_toggle:
    # future_co2_zero_year = future_expand.slider(
    #     'Year of achieving net zero CO2 emissions', 2025, 2100, 2050)
    # future_ch4_rate = future_expand.slider(
    #     'Change in CH4 emissions by 2030 (%)', -10., 0., -2.2,
    #     step=0.1, format='%f') / 100
    # future_n2o_rate = future_expand.slider(
    #     'Change in N2O emissions by 2030 (%)', -10., 0., -0.7,
    #     step=0.1, format='%f') / 100

    future_co2_zero_year = future_expand.slider(
        'Year of achieving net zero CO2 emissions', 2025, 2100, 2050)
    interim_ch4_rate = future_expand.slider(
        'Change in CH4 emissions by 2030 (%)', -90, 0, -40,
        step=1, format='%f') / 100
    interim_n2o_rate = future_expand.slider(
        'Change in N2O emissions by 2030 (%)', -90, 0, -15,
        step=1, format='%f') / 100
    future_ch4_rate = (1 + interim_ch4_rate)**(1/(2030-yr1)) - 1
    future_n2o_rate = (1 + interim_n2o_rate)**(1/(2030-yr1)) - 1

else:
    future_co2_zero_year = None
    future_ch4_rate = None
    future_n2o_rate = None


# st.sidebar.write('---')

if 'Absolute Temperature Change' in temp_calc_method:
    baseline = st.sidebar.selectbox(
        "Choose period for pre-industrial temperature baseline",
        ['1850-1900', '1750-1800', '1750'], 0)
if 'Relative Temperature Change' in temp_calc_method:
    baseline = str(st.sidebar.slider(
        "Choose baseline year for relative temperature change",
        min_value=yr0, max_value=yr1, value=1875))

date_range = st.sidebar.slider(
    "Choose Date Range",
    min_value=yr0,
    max_value=future_co2_zero_year if future_toggle else yr1,
    value=[1900, future_co2_zero_year] if future_toggle else [1900, yr1]
    )

countries = sorted(st.sidebar.multiselect(
    "Choose countries and/or regions",
    sorted(list(set(df['country']))),
    [' Aggregated emissions for all countries'],
    help='For a guide to available emissions regions,\
          please scroll down for the main text.'
))

if LULUCF == 'Include LULUCF':
    category_set = sorted(list(set(df['category']) -
                               set(['3: Agriculture, sum of 3A and 3B']) -
                               set(['0: Total excluding LULUCF'])
                               ))
    # Note that we remove the option to select the sub-level aggregation
    # '3: Agriculture, sum of 3A and 3B' to reduce confusion. This aggregation
    # is actually included directly in PRIMAP-hist 2.3.1.
    category_default = ['1: Energy',
                        '2: Industrial Processes and Product Use (IPPU)',
                        '3: Agriculture, Forestry, and Other Land Use',
                        '4: Waste',
                        '5: Other']
elif LULUCF == 'Do Not Include LULUCF':
    category_set = sorted(list(
        set(df['category']) -
        set(['3: Agriculture, Forestry, and Other Land Use']) -
        set(['0: Total'])
                               ))
    # Note that we remove the option to select the sub-level aggregation
    # '3: Agriculture, Forestry, and Other Land Use' to reduce confusion. This
    # aggregation is actually included directly in PRIMAP-hist 2.3.1.
    category_default = ['1: Energy',
                        '2: Industrial Processes and Product Use (IPPU)',
                        '3: Agriculture, sum of 3A and 3B',
                        '4: Waste',
                        '5: Other']


categories = sorted(st.sidebar.multiselect(
    "Choose emissions categories",
    category_set, category_default,
    help='For a guide to available emissions categories/sectors,\
          please scroll down for the main text.'
))

# Warn user about data selections that result in double counting.
codes = [cat.split(':')[0] for cat in categories]
double_counter = []
if 'Total excluding LULUCF' in codes and len(codes) > 1:
    double_counter.append(('`Total excluding LULUCF` covers all ' +
                           '(sub)categories except LULUCF'))
elif '0' in codes and len(codes) > 1:
    double_counter.append('`0: Total` covers all (sub)categories')
for code in codes:
    if len(code) > 1:
        if code[0] in codes:
            double_counter.append(f'`{code}` is also counted in `{code[0]}`')
    if len(code) == 3:
        if code[:2] in codes:
            double_counter.append(f'`{code}` is also counted in `{code[:2]}`')

entities = sorted(st.sidebar.multiselect(
    "Choose entities (gases)",
    ['Methane', 'Carbon Dioxide', 'Nitrous Oxide']\
        if climate_model == 'IPCC AR5 Linear Impulse Response Model'\
        else sorted(list(set(df['entity']))),
    ['Methane', 'Carbon Dioxide', 'Nitrous Oxide'],
    help='For a guide to available greenhouse gases,\
          please scroll down for the main text.'
))


####
# IF 'IPCC AR5 Linear Impulse Response Model' MODEL SELECTED,
# CALCULATE TEMPERATURES
####
if climate_model == 'IPCC AR5 Linear Impulse Response Model':
    df_T, df_E = calc(df, scenarios, countries, categories, entities,
                      baseline, emissions_units,
                      future_toggle, future_co2_zero_year,
                      future_ch4_rate, future_n2o_rate)

if len(double_counter) > 0:
    st.sidebar.write('---')
    st.sidebar.warning('Watch Out: Double Counting')
    double_count_expander = st.sidebar.expander('View Double Counting Details')
    double_count_expander.write(
        'This selection double counts the following (sub)categories;\
         make sure this is what you intended:')
    for double in double_counter:
        double_count_expander.write(f'- {double}')

# CHECK DATA AVAILABLE
if prepare_data(df_E, scenarios, countries, categories, entities, 'country',
                date_range, False).empty:
    st.warning('No emissions data available for this selection')

####
# CREATE ALTAIR PLOTS
####
# year_expander = st.expander("Year Range Selection")
# with year_expander:
c1a, c1b, c2 = st.columns([1.75, 1.75, 1])
c2.subheader(' ')
dis_aggregation = c2.selectbox(
    "Choose the breakdown to plot",
    ['country', 'category', 'entity'],
    index=['country', 'category', 'entity'].index('category')
)


plot_style = c2.selectbox('Choose plot style', ['line', 'area'])

c2.caption(f"""
The timeseries depict depict annual emissions and the
{temp_calc_method.lower()} consequent of those emissions, relative to the
selected {baseline} baseline.

The top horizontal bar shows contributions over the *entire selected time
period*, therefore presenting dominant contributions to *historical*
temperature change.

The bottom horizontal bar shows contributions over the *final decade* of the
selected time period, presenting the dominant contributions to *recent*
temperature change.

Note, the size of a contribution's bar is proportional to absolute value;
large warming and large cooling will therefore have similar sized bars.
""")

# CREATE EMISSIONS PLOT
include_sum = True if plot_style == 'line' else False
grouped_data_E = prepare_data(df_E,
                              scenarios, countries, categories, entities,
                              dis_aggregation, date_range, include_sum)

# Transform from wide data to long data (altair likes long data)
alt_data = (grouped_data_E.T.reset_index()
                            .melt(id_vars=["index"])
                            .rename(columns={"index": "year", 'value': 'GWP'})
            )


# Create colour mapping that accounts for a black 'SUM' line if multiple
# lines are present
# Note, sorting this here, means it matches the order returned by the
# (sorted) output_T form the selection widgets; som colours for plots match.
c_domain = sorted(list(grouped_data_E.index))
# if include_sum and len(c_domain) > 1:
if 'SUM' in c_domain:
    c_domain.append(c_domain.pop(c_domain.index('SUM')))
c_range = colour_range(c_domain, include_sum, dis_aggregation)

if emissions_units == 'Absolute Mass':
    metric_units = ''
if emissions_units == 'GWP100':
    metric_units = 'CO\u2082-e'
if emissions_units == 'CO2-fe':
    metric_units = 'CO\u2082-fe'

if plot_style == 'line':
    chart_base = alt.Chart(alt_data).mark_line(opacity=0.9)
if plot_style == 'area':
    chart_base = alt.Chart(alt_data).mark_area(opacity=0.9)

chart_1a = (
    chart_base
    .encode(x=alt.X("year:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("GWP:Q",
                    title=f'annual emissions (Gt {metric_units} per year)',
                    # stack=None
                    ),
            color=alt.Color(dis_aggregation + ':N',
                            scale=alt.Scale(domain=c_domain, range=c_range)),
            tooltip=[(dis_aggregation + ':N'), 'GWP:Q']
            )
)

bar_data = alt_data[alt_data[dis_aggregation]
                    != 'SUM'].astype(dtype={'year': 'int32'})

chart_1a2 = (alt.Chart(bar_data, height=50).mark_bar(opacity=0.9).encode(
    color=alt.Color(dis_aggregation + ':N',
                    scale=alt.Scale(domain=c_domain, range=c_range),
                    legend=None),
    x=alt.X('sum(GWP):Q', stack="normalize", axis=alt.Axis(
        domain=False, ticks=False, labels=False),
            title=('cumulative emissions breakdown between ' +
                   f'{date_range[0]}-{date_range[1]}')),
    tooltip=[(dis_aggregation + ':N'), 'sum(GWP):Q'])
    .configure_axis(grid=False)
)

last_decade = bar_data.loc[(date_range[1] - bar_data['year'] < 10)]
earliest_year = last_decade['year'].min()
chart_1a3 = (alt.Chart(last_decade, height=50).mark_bar(opacity=0.9).encode(
    color=alt.Color(dis_aggregation + ':N',
                    scale=alt.Scale(domain=c_domain, range=c_range),
                    legend=None),
    x=alt.X('sum(GWP):Q', stack="normalize", axis=alt.Axis(
                domain=False, ticks=False, labels=False),
            title=('cumulative emissions breakdown between ' +
                   f'{earliest_year}-{date_range[1]}')),
    tooltip=[(dis_aggregation + ':N'), 'sum(GWP):Q'])
    .configure_axis(grid=False)
)


# c1a.subheader(f'emissions using GWP_100(Gt CO2-e yr-1)')
c1a.subheader(F'annual emissions ({emissions_units})')
chart_1a = (chart_1a
            .configure_legend(orient='top-left')
            .configure_axis(grid=False)
            .configure_view(strokeOpacity=0.0))


c1a.altair_chart(chart_1a, use_container_width=True, theme="streamlit")
c1a.altair_chart(chart_1a2, use_container_width=True, theme=None)
c1a.altair_chart(chart_1a3, use_container_width=True, theme=None)


if 'SUM' in grouped_data_E.index:
    value = grouped_data_E.loc['SUM'].sum()
elif not grouped_data_E.empty:
    value = grouped_data_E.sum(axis=1).sum()
else:  # for elegent error handling
    value = 0

value = adjusted_scientific_notation(value * 1.e9, True)
c1a.metric(('cumulative emissions between ' +
           f'{date_range[0]}-{date_range[1]} ({emissions_units})'),
           # f'{value:.2E} Mt CO2-e',)
           f'{value}t {metric_units}')


# CREATE WARMING PLOT
# include_sum = True
grouped_data = prepare_data(df_T, scenarios, countries, categories, entities,
                            dis_aggregation, date_range, include_sum)

# Transform from wide data to long data (altair likes long data)
alt_data = (grouped_data.T
                        .reset_index()
                        .melt(id_vars=["index"])
                        .rename(columns={"index": "year", 'value': 'warming'})
            )

# Create colour mapping that accounts for a black 'SUM' line if multiple
# lines are present
# Note, sorting this here, means it matches the order returned by the
# (sorted) output_T form the selection widgets; som colours for plots match.
c_domain = sorted(list(grouped_data.index))
# if include_sum and len(c_domain) > 1:
if 'SUM' in c_domain:
    c_domain.append(c_domain.pop(c_domain.index('SUM')))
c_range = colour_range(c_domain, include_sum, dis_aggregation)

if plot_style == 'line':
    chart_base = alt.Chart(alt_data).mark_line(opacity=0.9)
if plot_style == 'area':
    chart_base = alt.Chart(alt_data).mark_area(opacity=0.9)


chart_1b = (
    chart_base
    .encode(x=alt.X("year:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("warming:Q",
                    title='global temperature change (°C)',
                    # stack=None
                    ),
            color=alt.Color(dis_aggregation + ':N',
                            scale=alt.Scale(domain=c_domain, range=c_range),
                            legend=None),
            tooltip=[(dis_aggregation + ':N'), 'warming:Q']
            )
)


bar_data = (alt_data[alt_data[dis_aggregation] != 'SUM']
            .astype(dtype={'year': 'int32'}))
A = (bar_data.loc[(bar_data['year'] == date_range[1]),
                  [dis_aggregation, 'warming']]
             .set_index(dis_aggregation).to_dict('index'))
B = (bar_data.loc[(bar_data['year'] == date_range[0]),
                  [dis_aggregation, 'warming']]
             .set_index(dis_aggregation).to_dict('index'))
bar_keys = [[key, A[key]['warming'] - B[key]['warming']] for key in A]
bar_keys = pd.DataFrame(bar_keys, columns=[dis_aggregation, 'warming'])

chart_1b2 = (alt.Chart(bar_data[bar_data['year'] == date_range[1]], height=50)
                .mark_bar(opacity=0.9)
                .encode(x=alt.X('warming:Q',
                                stack='normalize',
                                axis=alt.Axis(
                                    domain=False, ticks=False, labels=False),
                                title=('temperature change breakdown in ' +
                                       f'{date_range[1]} ' +
                                       f'relative to {baseline}')),
                        color=alt.Color(dis_aggregation + ':N',
                        scale=alt.Scale(domain=c_domain, range=c_range),
                        legend=None),
                        tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
                .configure_axis(grid=False)
             )

C = (bar_data.loc[(bar_data['year'] == earliest_year),
                  [dis_aggregation, 'warming']]
             .set_index(dis_aggregation).to_dict('index'))
bar_keys = [[key, A[key]['warming'] - C[key]['warming']] for key in A]
bar_keys = pd.DataFrame(bar_keys, columns=[dis_aggregation, 'warming'])

chart_1b3 = (alt.Chart(bar_keys, height=50).mark_bar(opacity=0.9).encode(
    x=alt.X('warming:Q', stack='normalize',
            axis=alt.Axis(domain=False, ticks=False, labels=False),
            title=('temperature change breakdown between ' +
                   f'{earliest_year}-{date_range[1]}')),
    color=alt.Color(dis_aggregation + ':N',
                    scale=alt.Scale(domain=c_domain, range=c_range),
                    legend=None),
    tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
    .configure_axis(grid=False)
)


c1b.subheader(f'warming relative to {baseline} (°C)')
chart_1b = (chart_1b
            .configure_legend(orient='top-left')
            .configure_axis(grid=False)
            .configure_view(strokeOpacity=0.0)
            )

c1b.altair_chart(chart_1b, use_container_width=True, theme="streamlit")
c1b.altair_chart(chart_1b2, use_container_width=True, theme=None)
c1b.altair_chart(chart_1b3, use_container_width=True, theme=None)


if 'SUM' in grouped_data.index:
    value = grouped_data.loc['SUM', str(date_range[1])]
elif not grouped_data.empty:
    value = grouped_data[str(date_range[1])].sum(axis=0).sum()
else:  # for elegent error handling
    value = 0
value = adjusted_scientific_notation(value, True)
c1b.metric(f'net warming in {date_range[1]} relative to {baseline}',
           # f'{value:.2E}°C',
           f'{value} °C')

# ####
# Make Sankey Diagram
####
c3, c4 = st.columns([3.5, 1])
c4.subheader(' ')
# NOTE: We now count warming as relative to the baseline (not the start year
# of the selected date_range), in anticipation of arbitrary start date being
# included in the second relative-warming calculation approach.
sankey_cs = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['country', 'category'], date_range, False)
sankey_sg = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['category', 'entity'], date_range, False)
sankey_gc = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['entity', 'country'], date_range, False)
sankey_gcs = prepare_data(df_T, scenarios, countries, categories, entities,
                          ['entity', 'country', 'category'],
                          date_range, False)

middle = c4.selectbox('Choose the focused variable',
                      ['country', 'category', 'entity'], 1)
labels = countries + categories + entities
node_colors = (colour_range(countries, False, 'country') +
               colour_range(categories, False, 'category') +
               colour_range(entities, False, 'entity'))
sources, targets, values, exceptions = [], [], [], []
for c in countries:
    for s in categories:
        sources.append(labels.index(c))
        targets.append(labels.index(s))
        try:
            values.append(sankey_cs.loc[(c, s), str(date_range[1])])
        except KeyError:
            # exceptions.append(f'(country): `{c}` & (category): `{s}`')
            values.append(0)

for s in categories:
    for g in entities:
        sources.append(labels.index(s))
        targets.append(labels.index(g))
        try:
            values.append(sankey_sg.loc[(s, g), str(date_range[1])])
        except KeyError:
            # exceptions.append(f'(category): `{s}` & (entity): `{g}`')
            values.append(0)

for g in entities:
    for c in countries:
        sources.append(labels.index(g))
        targets.append(labels.index(c))
        try:
            values.append(sankey_gc.loc[(g, c), str(date_range[1])])
        except KeyError:
            # exceptions.append(f'(entity): `{g}` & (country): `{c}`')
            values.append(0)

# The exceptions in the loops above only show connections in the sankey that
# have no data; not the actual three-variable missing data. ie we might not
# have data for UK + CH4 + LULUCF, but there might still be a connection
# between UK + CH4 (due to other categories), and UK + LULUCF (due to other
# gases), etc. Therefore to actually catchall exceptions we have to search
# through separately in a new three-way seach. It might be faster to refactor
# the three two-level for loops above into a single three-level nested loop.
# For now this works though.
for c in countries:
    for s in categories:
        for g in entities:
            try:
                _ = sankey_gcs.loc[(g, c, s)]
            except KeyError:
                exceptions.append(f'(country): `{c}` & ' +
                                  f'(category): `{s}` & ' +
                                  f'(entity): `{g}`')

if len(exceptions) >= 1 and len(double_counter) == 0:
    st.sidebar.write('---')
if len(exceptions) >= 1:
    # st.sidebar.warning('Watch Out: Some Emissions Missing')
    exceptions_expander = st.sidebar.expander('View Exceptions')
    exceptions_expander.write(
        f'This particular selected subset of the dataset contains\
          no emissions data for the following {len(exceptions)} combinations:')
    for exception in exceptions:
        exceptions_expander.write(f'- {exception}')

# The below seems to fix the problem of the exceptions box being cut off at the
# bottom of the sidebar.
st.sidebar.write("""


""")

flow_colors = ['rgba(246, 51, 102, 0.3)' if t > 0
               else 'rgba(58, 213, 203, 0.3)'
               # else '#284960'
               for t in values]
values = [abs(t) for t in values]

cs = len(countries) * len(categories)
sg = len(categories) * len(entities)
gc = len(entities) * len(countries)

if middle == 'country':
    sources = sources[:cs] + sources[-gc:]
    targets = targets[:cs] + targets[-gc:]
    values = values[:cs] + values[-gc:]
    flow_colors = flow_colors[:cs] + flow_colors[-gc:]
elif middle == 'category':
    sources = sources[:cs+sg]
    targets = targets[:cs+sg]
    values = values[:cs+sg]
    flow_colors = flow_colors[:cs+sg]
elif middle == 'entity':
    sources = sources[cs:]
    targets = targets[cs:]
    values = values[cs:]
    flow_colors = flow_colors[cs:]

# col = 'rgba' + str(matplotlib.colors.to_rgba("#374681"))
# col
fig = go.Figure(data=[go.Sankey(
    valuesuffix="K",
    textfont=dict(color="rgba(0.5,0.5,0.5,0.9)", size=10),
    node=dict(
        pad=40,
        thickness=20,
        line=dict(color='black', width=0.0),
        label=labels,
        color=node_colors
        # color=["#42539a" for x in node_colors]
        # color='blue'
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=flow_colors,
        # pad=
    )
)],
    layout=go.Layout(annotations=[
        go.layout.Annotation(
            text='- red flow is warming<br>- blue flow is cooling',
            align='left', showarrow=False, x=0.0, y=1.0)
    ]))

sankey_title = f'warming in {date_range[1]} relative to {baseline} (°C)'
c3.subheader(sankey_title)
fig.update_layout(
    # font=dict(size=10),
    #   font_color=st.get_option('textColor'),
    height=350,
    margin=dict(l=40, r=40, b=20, t=20, pad=4))
c3.plotly_chart(fig, use_container_width=True,
                config=dict({'displayModeBar': False}))

c4.caption("""
The sankey diagram presents a multi-level breakdown of contributions to
*global temperature change*. Selections in the left hand sidebar affect all
plots on this page; data used in the sankey plot corresponds exactly to the
data in the plots above.

Warming contributions (flows) are red; cooling contributions (flows) are blue.
As with the bars above, width of the flows is proportional to the absolute
value of temperature change - a wide red bar is a large warming impact, and a
wide blue bar is a large cooling impact. The "true" size of any given node is
therefore the width of the red flows minus the width of the blue flows.
""")

# about_expander = st.expander('About')
st.markdown('---')
# introduction = st.expander('Introduction')
st.markdown(
    """
## Introduction


**This app provides a quantitative esimation of contributions to global
temperature increase due to emissions from individual greenhouse gases,
countries, and emission categories and subsectors.**

**Make a selection to explore within those groups using the left side bar. The
default mode explores historical emissions and their contributions to global
temperature change; you can also include and experiment with projections of
future emissions.**



The goals of the Paris Agreement open with:

> This Agreement, in enhancing the implementation of the Convention,
> including its objective, aims to strengthen the global response to the threat
> of climate change, in the context of sustainable development and efforts to
> eradicate poverty, including by:
>
> holding the increase in the global average temperature to well below 2°C
> above pre-industrial levels and pursuing efforts to limit the temperature
> increase to 1.5°C above pre-industrial levels, recognizing that this would
> significantly reduce the risks and impacts of climate change

The fundamental question of the climate crisis is not how to reduce emissions,
but how to reduce the risks and negative impacts consequent of global warming
induced climate change. Therefore, as the world mobilises to deliver rapid and
ambitious mitigation, with widespread acceleration in the uptake of net zero
targets, it's important to understand how both historical emissions and future
emissions trajectories have contributed and will continue to contribute towards
global temperature change.

When multiple greenhouse gases are involved this is a non-trivial question, but
it is one that is essential to guide the strongest and most well-informed
decision making. Further, since present and future warming are dependent on
historical emissions, it is not enough to consider only warming caused by
actions from the date that a climate plan was put into place onwards; these
actions also need to be set within context of historical emissions, especially
when factoring justice into transitions.

With this in mind, we've created an interactive app enabling exploration of the
contributions to global temperature change, broken down by each country,
sector, and gas. This translation of emissions into temperature impact is
conducted in a fully transparent and traceable manner; warming is calculated
using the IPCC AR5 Linear Impulse Response Model, driven by emissions from the
comprehensive PRIMAP-hist dataset.

Future updates will provide:
- Inclusion of the (smaller) contributions from other greenhouse-gases.
- Inclusion of per-capita and per-GDP scaling for emissions and warming.
- Option to use FaIR climate model instead of IPCC AR5-IR model
- Additional options for future emissions, eg. using data from common \
scenarios, or national net zero targets.
- Global warming impact calculations for user-uploaded emissions pathways, \
thus widening use of transparent calculation tools to anyone who wishes to \
assess the impact of their actions to global temperature changes.

""")

# coverage = st.expander('Coverage of emissions and temperature impact')
st.markdown("""
## Coverage of emissions and temperature impact
Emissions of individual greenhouse gases are as reported in the PRIMAP
database. [^Gütschow et al 2022]

Temperature change alculations are all based on the linear response model
documented in the IPCC's 5th Assessment Report.
[^IPCC AR5 WG1 (Myhre et al 2013)]


*A note on double counting within selections:* in the 'region' and 'category'
multi-selects in the side bar, subgroupings of emissions can be selected
alongside their umbrella group; this allows for more granular exploration of
the contributions to historical warming, however **double counting is possible
if you select both a group and any of its subgroups:** for example, selecting
'IPC1' and 'IPC1A' will double count 'IPC1A' in the calculations; selecting
'European Union' and 'France' will double count 'France' in the calculations.
""")

with st.expander(
    'Which contributors to temperature change are included?'
):
    list1, list2 = st.columns(2)
    list1.markdown("""
    #### Categories
    (the main categories and subsector divisions from the IPCC 2006 Guidelines
    for national greenhouse gas inventories [^IPCC 2006])
    - 0: Total
        - 1: Energy
            - 1A: Fuel Combustion Activities
            - 1B: Fugitive Emissions from Fuels
                - 1B1: Solid Fuels
                - 1B2: Oil and Natural Gas
                - 1B3: Other Emissions from Energy Production
            - 1C: Carbon Dioxide Transport and Storage (currently no\
                data available)
        - 2: Industrial Processes and Product Use (IPPU)
            - 2A: Mineral Industry
            - 2B: Chemical Industry
            - 2C: Metal Industry
            - 2D: Non-Energy Products from Fuels and Solvent Use
            - 2E: Electronics Industry (no data available as the\
                    category is only used for fluorinated gases which are only\
                    resolved at the level of category 2)
            - 2F: Product uses as Substitutes for Ozone Depleting\
                    Substances (no data available as the category is only used\
                    for fluorinated gases which are only resolved at the level\
                    of category 2)
            - 2G: Other Product Manufacture and Use
            - 2H: Other
        - 3: Agriculture, Forestry, and Other Land Use*
            - 3A: Livestock
            - 3B: Agriculture excluding Livestock*
            - 3C: Land Use, Land Use Change, and Forestry (LULUCF)*
        - 4: Waste
        - 5: Other

    *these categories are aggregates implemented in PRIMAP-hist that differ
    slightly from the official IPCC 20006 Guidelines.
    """)

    list2.markdown(
        """
    #### Regions
    - All UNFCCC member states, as well as most non-UNFCCC territories\
    [^ISO 3166-1 alpha-3]

    - Additional groupings, including:
        - Aggregated emissions for all countries
        - Annex I Parties to the Convention\
            ([list](https://unfccc.int/parties-observers))
        - Non-Annex I Parties to the Convention\
            ([list](https://unfccc.int/parties-observers))
        - Alliance of Small Island States\
            ([list](https://www.aosis.org/about/member-states/))
        - BASIC countries (Brazil, South Africa, India, and China)
        - European Union post Brexit\
            ([list](https://europa.eu/european-union/about-eu/countries_en))
        - Least Developed Countries
        - Umbrella Group\
            ([list](https://unfccc.int/process-and-meetings/parties-non-party-stakeholders/parties/party-groupings))

    #### Entities
    (the gases that dominantly contribute to global warming)
    - (CO2) Carbon Dioxide
    - (CH4) Methane
    - (N2O) Nitrous Oxide

    #### Reporting Scenarios
    - HISTCR: Prioritise Country-Reported Data
    - HISTTP: Prioritise Third-Party Data
    """)

# st.markdown(
#     """

# ### What emissions are not included in this app?
# *A note on the non-inclusion of LULUCF:* land use, land use change, and
# forestry (LULUCF) emissions are not included due to issues with data
# consistency; for more on this,
# [see note here](http://www.pik-potsdam.de/paris-reality-check/primap-hist/).
# As such, **warming calculated in this release of the app is due to all CO2,
# CH4, and N2O emissions excluding LULUCF.**

# *A note on the non-inclusion of the F-gases (HFCs, CFCs, SF6):* the three
# forcing agents that dominate contributions to global warming are CO2, CH4,
# N2O. The behaviours of these gases, in particular the length of time for
# which they reside in the atmosphere, is relativelt unambiguous. Emissions of
# the F-gases are included as aggreage emissions, which cannot be reduced to a
# single atmospheric lifetime; this introduces uncertainty in how to treat
# them. While an upcoming release of this app will provide full coverage of the
# F-gases, their inclusion is not expected to significantly change the warming
# contributions in the majority of cases.

# *A note on the non-inclusion of natural forcing:*

with st.expander(
    'What is the difference between absolute and relative temperature change?'
):
    st.markdown("""
    **Absolute Temperature Change** is defined here as warming consequent of
    emissions relative to the preindustrial baseline. The chosen model
    calculates temperatures from emissions for the entire historical period,
    and then subtracts the average temperature of the chosen baseline period
    (typically 1850-1900 is chosen as a reasonable proxy for pre-industrial
    conditions).

    **Relative Temperature Change** is defined here as warming consequent of
    emissions since a chosen baseline, compared to what would have happened
    if those emissions had ceased in the specified baseline year. The chosen
    model first calculates Absolute Temperature Change for a given period, and
    then subtracts the Absolute Temperature Change timeseries for a
    counterfactual emissions timeseries where emissions cease in the selected
    baseline year. Temperature change *before* the baseline year are given as
    Absolute Temperature Change relative to temperature in the baseline year.
    """)

with st.expander(
    'What emissions metrics are available for presenting emissions data?'
):
    st.markdown("""
    **Absolute Mass** is actual mass of each greenhouse gas emitted each year.

    **GWP100** is Global Warming Potential at a time horizon of 100 years.

    **CO2-fe** is CO2-forcing-equivalent emissions, ie the CO2 emissions
    timeseries that would result in the same forcing as the specified emissions
    timeseries of any greenhouse gas. These are calculated using an inverted
    version of the selected climate model.
    """)

st.markdown(
    """
---

### Contact
Warming impact calculated, and app developed and designed by:

**Tristram Walsh** |
tristram.walsh@ouce.ox.ac.uk |
Environmental Change Institute |
University of Oxford

**Myles Allen** |
myles.allen@ouce.ox.ac.uk |
Environmental Change Institute |
University of Oxford

<p>&nbsp;</p>

**Please get in touch with us if you notice issues, have suggestions, or would
like access to the warming dataset.**

**Additionally, we'd welcome hearing about where and how you you have used
the information and insights presented here.**

<p>&nbsp;</p>

With support from [4C](https://4c-carbon.eu) (Climate-Carbon Interactions in
the Current Century)
""",
    unsafe_allow_html=True
)

logo1, logo2, logo3, logo4, logo5, _ = st.columns([1, 1, 1, 1, 1, 5])
# logo1.image('https://drive.google.com/uc?export=view&id=1ORZiuxMRqF8TE0tvRFqAK5jSLL3K2dpe')
# logo2.image("https://drive.google.com/uc?export=view&id=1Ocp5yrIFc92NfScro9L7CU-Ctpz5YmUi")
# logo3.image("https://drive.google.com/uc?export=view&id=1ONTfrsFQu0lnuqjbrEw26TLUdUZ_r9rZ")
# logo4.image('https://drive.google.com/uc?export=view&id=1OX6NUw3aghAwKfDH2-a9gHLEvel1Q4A4')
# logo5.image("https://drive.google.com/uc?export=view&id=1OGVCRwfaZTYPx3-LkcVx6qvDnZNjhuAU")
logo1.image("./imgs/4Clogo.jpeg")
logo2.image("./imgs/ONZlogo.png")
logo3.image("./imgs/ECIlogo.jpeg")
logo4.image("./imgs/SoGElogo.png")
logo5.image("./imgs/Oxfordlogo.png")

st.caption("""
---

[^Gütschow et al 2022]: Gütschow, J.; Pflüger, M. (2022): The PRIMAP-hist
national historical emissions time series v2.4 (1750-2021). zenodo.
https://doi.org/10.5281/zenodo.7179775


[^IPCC AR5 WG1 (Myhre et al 2013)]: Myhre, G., D. Shindell, F.-M. Bréon,
W. Collins, J. Fuglestvedt, J. Huang, D. Koch, J.-F. Lamarque, D. Lee,
B. Mendoza, T. Nakajima, A. Robock, G. Stephens, T. Takemura and H. Zhang,
2013: Anthropogenic and Natural Radiative Forcing Supplementary Material.
In: Climate Change 2013: The Physical Science Basis.
Contribution of Working Group I to the Fifth Assessment Report of the
Intergovernmental Panel on Climate Change
[Stocker, T.F., D. Qin, G.-K. Plattner, M. Tignor, S.K. Allen, J. Boschung,
A. Nauels, Y. Xia, V. Bex and P.M. Midgley (eds.)].
Available from
https://www.ipcc.ch/report/ar5/wg1/chapter-8sm-anthropogenic-and-natural-radiative-forcing-supplementary-material/

[^IPCC 2006]: Jim Penman (UK), Michael Gytarsky (Russia),
Taka Hiraishi (Japan), William Irving (USA), and Thelma Krug (Brazil),
(2006) Overview of the IPCC Guidelines for National Greenhouse Gas Inventories.
Available from
https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/0_Overview/V0_1_Overview.pdf

[^ISO 3166-1 alpha-3] : See
https://unstats.un.org/unsd/tradekb/knowledgebase/country-code, and
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3 for more information.

"""
           )
