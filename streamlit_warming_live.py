"""Streamlit web app exploring pre-calculated (static) warming dataset."""
import io
import csv
import datetime as dt
# import numba
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib
# import matplotlib.pyplot as plt
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

st.markdown(
    """
    # Contributions to Historical Warming
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
def ETmod(nyr, a):
    """Create linear operator to convert emissions to warming."""
    Tcal = np.zeros((nyr, nyr))

    # add one to the time array for consistency with AR5 formulae
    time = np.arange(nyr) + 1

    # loop over thermal response times using AR5 formula for AGTP
    for j in [0, 1]:
    # for j in numba.prange(2):
        Tcal[:, 0] = Tcal[:, 0] + a[4] * a[13] * \
            a[0] * a[j+10] * (1 - np.exp(-time / a[j+15]))

        # loop over gas decay terms using AR5 formula for AGTP
        for i in [1, 2, 3]:
        # for i in numba.prange(1, 4, 1):
            Tcal[:, 0] = Tcal[:, 0]+a[4]*a[13]*a[i]*a[i+5]*a[j+10] * \
                (np.exp(-time/a[i+5])-np.exp(-time/a[j+15]))/(a[i+5]-a[j+15])

    # build up the rest of the Toeplitz matrix
    for j in range(1, nyr):
    # for j in numba.prange(1, nyr):
        Tcal[j:nyr, j] = Tcal[0:nyr-j, 0]

    return Tcal


# @numba.jit(nopython=True)  # This didn't help...
# @st.cache(show_spinner=False)
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


def emissions_units(df):
    """Return pandas DataFrame of entities (gases) and unit (units)."""
    entities = np.array(df['entity'])
    units = np.array(df['unit'])
    entity_unit = {}
    for A, B in zip(entities, units):
        entity_unit[A] = B
    emissions_units = pd.DataFrame(data=entity_unit, index=[0])

    return emissions_units.T


@st.cache(show_spinner=False, suppress_st_warning=True)
def load_data(file):
    """Load the dataset, and rename codes with human-friendly terms."""
    # NOTE: function approach allows streamlit caching,
    loading_message = st.sidebar.text('please wait while data initialises')

    df = pd.read_csv(file)

    not_country = ['EARTH', 'ANNEXI', 'NONANNEXI', 'AOSIS',
                   'BASIC', 'EU28', 'LDC', 'UMBRELLA']
    iso_country = list(set(df['country']) - set(not_country) - set(['ANT']))

    country_names = {x: pycountry.countries.get(alpha_3=x).name
                     for x in iso_country}
    country_names['Netherlands Antilles'] = 'ANT'
    country_names.update(
        {'EARTH': 'Aggregated emissions for all countries',
         'ANNEXI': 'Annex I Parties to the Convention',
         'NONANNEXI': 'Non-Annex I Parties to the Convention',
         'AOSIS': 'Alliance of Small Island States',
         'BASIC': 'BASIC countries (Brazil, South Africa, India, and China)',
         'EU28': 'European Union',
         'LDC': 'Least Developed Countries',
         'UMBRELLA': 'Umbrella Group'}
    )

    df['country'] = df['country'].replace(country_names)

    scenario_names = {'HISTCR': 'Prioritise Country-Reported Data',
                      'HISTTP': 'Prioritise Third-Party Data'}
    df['scenario'] = df['scenario'].replace(scenario_names)

    category_names = {
        'IPCM0EL': 'IPCM0EL: National Total excluding LULUCF',
        'IPC1': 'IPC1: Energy',
        'IPC1A': 'IPC1A: Fuel Combustion Activities',
        'IPC1B': 'IPC1B: Fugitive Emissions from Fuels',
        'IPC1B1': 'IPC1B1: Solid Fuels',
        'IPC1B2': 'IPC1B2: Oil and Natural Gas',
        'IPC1B3': 'IPC1B3: Other Emissions from Energy Production',
        'IPC1C': 'IPC1C: Carbon Dioxide Transport and Storage (currently no\
                  data available)',
        'IPC2': 'IPC2: Industrial Processes and Product Use (IPPU)',
        'IPC2A': 'IPC2A: Mineral Industry',
        'IPC2B': 'IPC2B: Chemical Industry',
        'IPC2C': 'IPC2C: Metal Industry',
        'IPC2D': 'IPC2D: Non-Energy Products from Fuels and Solvent Use',
        'IPC2E': 'IPC2E: Electronics Industry (no data available as the\
                  category is only used for fluorinated gases which are only\
                  resolved at the level of category IPC2)',
        'IPC2F': 'IPC2F: Product uses as Substitutes for Ozone Depleting\
                  Substances (no data available as the category is only used\
                  for fluorinated gases which are only resolved at the level\
                  of category IPC2)',
        'IPC2G': 'IPC2G: Other Product Manufacture and Use',
        'IPC2H': 'IPC2H: Other',
        'IPCMAG': 'IPCMAG: Agriculture, sum of IPC3A and IPCMAGELV',
        'IPC3A': 'IPC3A: Livestock',
        'IPCMAGELV': 'IPCMAGELV: Agriculture excluding Livestock',
        'IPC4': 'IPC4: Waste',
        'IPC5': 'IPC5: Other'}
    df['category'] = df['category'].replace(category_names)

    loading_message.empty()

    return df


@st.cache(show_spinner=False, suppress_st_warning=True)
def calc(df, scenarios, countries, categories, entities):
    """Calculate warming impact, and GWP emissions, for given selection."""
    # the GWP_100 factors for [CO2, CH4, N2O] respectively
    gwp = {'CO2': 1., 'CH4': 28., 'N2O': 265.}

    emis_to_calculate = df[
                    (df['scenario'] == scenarios) &
                    (df['country'].isin(countries)) &
                    (df['category'].isin(categories)) &
                    (df['entity'].isin(entities))]

    # Prepare the virtual csv files to output calculated things to
    column_names = ['scenario', 'country', 'category', 'entity', 'unit']
    ny = 2018 - 1850 + 1
    PR_year = np.arange(2018 - 1850 + 1) + 1850
    column_names.extend([str(i) for i in PR_year])
    # Create in memory virtual csv to write temperatures to
    output_T = io.StringIO()
    csv_writer_T = csv.DictWriter(output_T, fieldnames=column_names)
    csv_writer_T.writeheader()
    # Create in memory virtual csv to write GWP to
    output_GWP = io.StringIO()
    csv_writer_GWP = csv.DictWriter(output_GWP, fieldnames=column_names)
    csv_writer_GWP.writeheader()

    t1 = dt.datetime.now()
    # times_calc, times_csv = [], []
    i = 1
    number_of_series = len(countries) * len(categories) * len(entities)
    calc_text = st.sidebar.text('calculating...')
    for country in countries:
        for category in categories:
            for entity in entities:

                #  Visually show how far through the calculation we are
                # name = f'{scenarios}, {country}, {category}, {entity}'
                percentage = int(i/number_of_series*100)
                loading_bar = percentage // 5 * '.' + (20 - percentage // 5) * ' '
                # print(f'\r{percentage}% {loading_bar} {name}', end='')
                calc_text.text(f'calculating {loading_bar} {percentage}% ')
                i += 1

                df_timeseries = emis_to_calculate[
                    (emis_to_calculate['country'] == country) &
                    (emis_to_calculate['category'] == category) &
                    (emis_to_calculate['entity'] == entity)
                    ].transpose().loc['1850':]

                # NOTE: PRIMARP doesn't have emissions timeseries for all
                # combinations of scenario, country, category, entity.
                # Thereofore proceed only `if` this data is present in PRIMAP
                # Crucially, not doing this leads to entries of different data
                # types in the year columns, which prevents pandas doing number
                # operations on them later on down the line!
                if not df_timeseries.empty:
                    # FIRST compute temperatures for warming virtual csv
                    arr_timeseries = df_timeseries.values.squeeze() / 1.e6

                    # Calculate the warming impact from the individual_series
                    
                    # ti = dt.datetime.now()
                    temp = ETmod(ny, a_params(entity)) @ arr_timeseries
                    # tj = dt.datetime.now()
                    # times_calc.append(tj-ti)

                    # Create dictionary with the new temp data in
                    new_row = {'scenario': scenarios,
                               'country': country,
                               'category': category,
                               'entity': entity,
                               'unit': 'K'}
                    new_row.update({str(PR_year[i]): temp[i]
                                    for i in range(len(temp))})
                    # Write this dictionary to the in-memory csv file.
                    csv_writer_T.writerow(new_row)

                    # SECOND compute GWP for GWP virtual csv
                    GWP = df_timeseries.values.squeeze() / 1.e6 * gwp[entity]
                    new_row = {'scenario': scenarios,
                               'country': country,
                               'category': category,
                               'entity': entity,
                               'unit': 'GWP GtC CO2-e yr-1'}
                    new_row.update({str(PR_year[i]): GWP[i]
                                    for i in range(len(GWP))})
                    csv_writer_GWP.writerow(new_row)
                    # tk = dt.datetime.now()
                    # times_csv.append(tk-tj)

    t2 = dt.datetime.now()
    calc_text.text(f'new calculation took: {(t2-t1)}')
    # st.sidebar.text(f'average calc time: {np.mean(times_calc)}')
    # st.sidebar.text(f'average csv time: {np.mean(times_csv)}')

    output_T.seek(0)  # we need to get back to the start of the StringIO
    df_T = pd.read_csv(output_T)
    output_GWP.seek(0)  # we need to get back to the start of the StringIO
    df_GWP = pd.read_csv(output_GWP)

    # Just being paranoid about data leaking between user interactsions (ie 
    # ending up with data duplications, if each subsequent data selection adds
    # on top of the existing StringIO in memory virtual csv)
    output_T = io.StringIO()
    output_GWP = io.StringIO()

    return df_T, df_GWP


def prepare_data(df, scenarios, countries, categories, entities,
                 dis_aggregation, date_range, offset, include_total):
    """Group, time-slice, offset, and calculate sum as required."""
    data = df[(df['scenario'] == scenarios) &
              (df['country'].isin(countries)) &
              (df['category'].isin(categories)) &
              (df['entity'].isin(entities))]

    # Group data
    grouped_data = data.groupby(dis_aggregation).sum()

    # Restrict the data to just that selected by the slider
    grouped_data = grouped_data.loc[:, str(date_range[0]):str(date_range[1])]

    # If offset selected, subtract temperature at beginning of date range from
    # the rest of the timeseries
    if offset:
        start_temps = grouped_data[str(date_range[0])]
        grouped_data = grouped_data.sub(start_temps, axis='index')

    # Add 'SUM' of data to data, if there are multiple lines
    if grouped_data.shape[0] > 1 and include_total is True:
        grouped_data = grouped_data.T
        grouped_data["SUM"] = grouped_data.sum(axis=1)
        grouped_data = grouped_data.T

    return grouped_data


def colour_range(domain, include_total, variable):
    """Create colormap with 'SUM' black."""
    if variable == 'country':
        cm = 'pastel'
        # cm = 'bright'
        # cm = 'winter'
        # cm= 'rainbow'
    elif variable == 'category':
        cm = 'Set1'
        # cm = 'autumn'
        # cm = "pastel"
    elif variable == 'entity':
        # cm = 'flare'
        cm = 'cool'
        # cm = 'viridis'
        # cm = 'muted'

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
    return cols_hex


# st.sidebar.write('## Make a selection')

####
# LOAD DATA
####

st.sidebar.markdown('# Select Data to Explore')

side_expand = st.sidebar.expander('Select Core Data')
d_set = side_expand.selectbox('Choose calculation',
                            #   ['Emissions', 'Warming Impact', 'Live'], 2)
                              ['Live'], 0)

if d_set == 'Emissions':
    # df = load_data("./data/PRIMAP-hist_v2.2_19-Jan-2021.csv")
    df = load_data(
    'https://zenodo.org/record/4479172/files/PRIMAP-hist_v2.2_19-Jan-2021.csv')
elif d_set == 'Warming Impact':
    df = load_data("./data/warming-contributions-data_PRIMAP-format.csv")
# elif d_set == 'Upload Own Data':
#     df = load_data(side_expand.file_uploader('upload emissions'))
elif d_set == 'Live':
    # df = load_data("./data/PRIMAP-hist_v2.2_19-Jan-2021.csv")
    df = load_data(
        'https://zenodo.org/record/4479172/files/PRIMAP-hist_v2.2_19-Jan-2021.csv')

####
# WIDGETS FOR USERS TO SELECT DATA
####
scenarios = side_expand.selectbox(
    "Choose scenario prioritisation",
    list(set(df['scenario'])),
    # index=list(set(df['scenario'])).index('HISTCR')
    # index=list(set(df['scenario'])).index('Prioritise country-reported data')
)

# st.sidebar.write('---')

date_range = st.sidebar.slider(
    "Choose Date Range", min_value=1850, max_value=2018, value=[1990, 2018])

countries = sorted(st.sidebar.multiselect(
    "Choose countries and/or regions",
    list(set(df['country'])),
    # not_country
    ['United Kingdom', 'Italy', 'Germany']
    # list(set(df['country']))
    # ['United Kingdom']
    # ['European Union',
    #  'United States',
    #  'Least Developed Countries',
    #  'Alliance of Small Island States',
    #  'BASIC countries (Brazil, South Africa, India, and China)']
    # ['European Union']
))
categories = sorted(st.sidebar.multiselect(
    "Choose emissions categories",
    list(set(df['category'])),
    # ['IPCM0EL: National Total excluding LULUCF']
    ['IPC1: Energy',
     'IPC2: Industrial Processes and Product Use (IPPU)',
     'IPCMAG: Agriculture, sum of IPC3A and IPCMAGELV',
     'IPC4: Waste',
     'IPC5: Other']

))
entities = sorted(st.sidebar.multiselect(
    "Choose entities (gases)",
    # sorted(list(set(df['entity']))),
    # sorted(list(set(df['entity']))),
    ['CH4', 'CO2', 'N2O'] if d_set == 'Live' else sorted(list(set(df['entity']))),
    ['CH4', 'CO2', 'N2O']
))


####
# IF 'LIVE' DATA SELECTED, CALCULATE TEMPERATURES
####
if d_set == 'Live':
    df_T, df_GWP = calc(df, scenarios, countries, categories, entities)

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
    index=['country', 'category', 'entity'].index('entity')
)
aggregated = sorted(list(set(['country', 'category', 'entity']) -
                         set([dis_aggregation])))

offset = c2.checkbox(
    f"Calculate warming relative to selected start year {date_range[0]}?",
    value=True)

# CREATE EMISSIONS PLOT
include_sum = True
grouped_data_GWP = prepare_data(df_GWP,
                                scenarios, countries, categories, entities,
                                dis_aggregation, date_range, False,
                                include_sum)
# Transform from wide data to long data (altair likes long data)
alt_data = grouped_data_GWP.T.reset_index().melt(id_vars=["index"])
alt_data = alt_data.rename(columns={"index": "year", 'value': 'GWP'})

# Create colour mapping that accounts for a black 'SUM' line if multiple
# lines are present
# Note, sorting this here, means it matches the order returned by the
# (sorted) output_T form the selection widgets; som colours for plots match.
c_domain = sorted(list(grouped_data_GWP.index))
# if include_sum and len(c_domain) > 1:
if 'SUM' in c_domain:
    c_domain.append(c_domain.pop(c_domain.index('SUM')))
c_range = colour_range(c_domain, include_sum, dis_aggregation)

warming_start = date_range[0] if offset else 1850

chart_1a = (
    alt.Chart(alt_data)
       .mark_line(opacity=0.9)
       .encode(x=alt.X("year:T", title=None),
               y=alt.Y("GWP:Q",
                       title=None,
                       # stack=None
                       ),
               color=alt.Color(dis_aggregation,
                               scale=alt.Scale(domain=c_domain, range=c_range))
               )
    #    .properties(height=500)
       .configure_legend(orient='top-left')
       .encode(tooltip=[(dis_aggregation + ':N'), 'GWP:Q'])
)
# c1a.subheader(f'emissions using GWP_100(Gt CO2-e yr-1)')
c1a.subheader('emissions using GWP_100')
c1a.altair_chart(chart_1a, use_container_width=True)
c1a.metric(f'sum of emissions between {date_range[0]}-{date_range[1]} (GWP100)',
           f"{grouped_data_GWP.loc['SUM'].sum():.2E}")


# CREATE WARMING PLOT
include_sum = True
grouped_data = prepare_data(df_T, scenarios, countries, categories, entities,
                            dis_aggregation, date_range, offset, include_sum)

# Transform from wide data to long data (altair likes long data)
alt_data = grouped_data.T.reset_index().melt(id_vars=["index"])
alt_data = alt_data.rename(columns={"index": "year", 'value': 'warming'})

# Create colour mapping that accounts for a black 'SUM' line if multiple
# lines are present
# Note, sorting this here, means it matches the order returned by the
# (sorted) output_T form the selection widgets; som colours for plots match.
c_domain = sorted(list(grouped_data.index))
# if include_sum and len(c_domain) > 1:
if 'SUM' in c_domain:
    c_domain.append(c_domain.pop(c_domain.index('SUM')))
c_range = colour_range(c_domain, include_sum, dis_aggregation)

warming_start = date_range[0] if offset else 1850

chart_1b = (
    alt.Chart(alt_data)
       .mark_line(opacity=0.9)
       .encode(x=alt.X("year:T", title=None),
               y=alt.Y("warming:Q",
                       title=None,
                       # stack=None
                       ),
               color=alt.Color(dis_aggregation,
                               scale=alt.Scale(domain=c_domain, range=c_range))
               )
    #    .properties(height=500)
       .configure_legend(orient='top-left')
       .encode(tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
)
c1b.subheader(f'warming relative to {warming_start} (K)')
c1b.altair_chart(chart_1b, use_container_width=True)
c1b.metric(f'sum of warming between {warming_start}-{date_range[1]} (K)',
           f"{grouped_data.loc['SUM', str(date_range[1])]:.2E}",
           delta_color="normal")

# ####
# Make Sankey Diagram
####
c3, c4 = st.columns([3.5, 1])
c4.subheader(' ')

# NOTE: We force the offset to the start of the selected date range, so this
# plot will always show the warming between the two dates in the range. This
# seems like intuitive behaviour. Therefore, the offset only applies to the
# line chart to change the relateive start date for that...
sankey_cs = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['country', 'category'], date_range, True, False)
sankey_sg = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['category', 'entity'], date_range, True, False)
sankey_gc = prepare_data(df_T, scenarios, countries, categories, entities,
                         ['entity', 'country'], date_range, True, False)

# snky_xpndr = st.expander('sankey data')
# snky_xpndr.write(sankey_cs)
# snky_xpndr.write(sankey_sg)
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
            exceptions.append(f'(country): {c} & (category): {s}')
            values.append(0)

for s in categories:
    for g in entities:
        sources.append(labels.index(s))
        targets.append(labels.index(g))
        try:
            values.append(sankey_sg.loc[(s, g), str(date_range[1])])
        except KeyError:
            exceptions.append(f'(category): {s} & (entity): {g}')
            values.append(0)

for g in entities:
    for c in countries:
        sources.append(labels.index(g))
        targets.append(labels.index(c))
        try:
            values.append(sankey_gc.loc[(g, c), str(date_range[1])])
        except KeyError:
            exceptions.append(f'(entity): {g} & (country): {c}')
            values.append(0)

if len(exceptions) > 1:
    exceptions_expander = st.expander('View Exceptions')
    exceptions_expander.write(
        'This particular selected subset of the dataset contains ' +
        'no data for the following combinations:')
    for exception in exceptions:
        exceptions_expander.write(exception)

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

sankey_title = f'warming between {date_range[0]} and {date_range[1]} (K)'
c3.subheader(sankey_title)
fig.update_layout(
    # font=dict(size=10),
    #   font_color=st.get_option('textColor'),
    height=350,
    margin=dict(l=40, r=40, b=20, t=20, pad=4))
c3.plotly_chart(fig, use_container_width=True,
                config=dict({'displayModeBar': False}))
