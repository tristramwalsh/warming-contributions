"""Hello World implementation of streamlit."""
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
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


def colour_range(domain):
    """Create colormap with 'TOTAL' black."""
    cm = 'bright'
    if len(domain) > 1:
        # domain = list(grouped_data.index)
        # colour_map = plt.get_cmap('viridis')
        # cols = np.array(list(iter(
        #   colour_map(np.linspace(0, 1, len(domain)-1)))))
        cols = np.array(sns.color_palette(cm, len(domain)-1))
        cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
                    for i in range(cols.shape[0])]
        # Final item in index is 'TOTAL'; add 'black' to range for this
        cols_hex.append('#000000')

    else:
        cols = np.array(sns.color_palette(cm, len(domain)))
        cols_hex = [matplotlib.colors.rgb2hex(cols[i, :])
                    for i in range(cols.shape[0])]
    return cols_hex


####
# LOAD DATA
####

@st.cache(show_spinner=False)
def load_data(file):
    """Load the dataset, and rename codes with human-friendly terms."""
    # NOTE: function approach allows streamlit caching,
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
         'BASIC': 'BASIC countries (Brazil, South Africa, India, and China',
         'EU28': 'European Union',
         'LDC': 'Least Developed Countries',
         'UMBRELLA': 'Umbrella Group'}
    )

    df['country'] = df['country'].replace(country_names)

    scenario_names = {'HISTCR': 'Prioritise country-reported data',
                      'HISTTP': 'Prioritise third-party data'}
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

    return df


# st.sidebar.write('## Make a selection')
d_set = st.sidebar.selectbox('Choose dataset',
                             ['Emissions', 'Warming Impact'], 1)
if d_set == 'Emissions':
    df = load_data("./data/PRIMAP-hist_v2.2_19-Jan-2021.csv")
elif d_set == 'Warming Impact':
    df = load_data("./data/warming-contributions-data_PRIMAP-format.csv")


####
# SELECT SUBSETS OF DATA
####

scenarios = st.sidebar.selectbox(
    "Choose scenario prioritisation",
    list(set(df['scenario'])),
    # index=list(set(df['scenario'])).index('HISTCR')
    # index=list(set(df['scenario'])).index('Prioritise country-reported data')
)

st.sidebar.write('---')

countries = st.sidebar.multiselect(
    "Choose countries and/or regions",
    list(set(df['country'])),
    # not_country
    # ['EU28', 'USA', 'IND', 'CHN']
    ['European Union']
)
categories = st.sidebar.multiselect(
    "Choose emissions categories",
    list(set(df['category'])),
    ['IPC1: Energy',
     'IPC2: Industrial Processes and Product Use (IPPU)',
     'IPCMAG: Agriculture, sum of IPC3A and IPCMAGELV',
     'IPC4: Waste',
     'IPC5: Other']

)
entities = st.sidebar.multiselect(
    "Choose entities (gases)",
    sorted(list(set(df['entity']))),
    sorted(list(set(df['entity'])))
)
dis_aggregation = st.sidebar.selectbox(
    "Choose the breakdown to plot",
    ['country', 'category', 'entity'],
    index=['country', 'category', 'entity'].index('entity')
)
aggregated = sorted(list(set(['country', 'category', 'entity']) -
                         set([dis_aggregation])))

# year_expander = st.expander("Year Range Selection")
# with year_expander:
slider_range = st.slider(
    "Plot Date Range", min_value=1850, max_value=2018, value=[1990, 2018])
offset = st.checkbox(
    f"Offset from your selected start year {slider_range[0]}?", value=True)
st.markdown("---")

####
# PREPARE DATA
####

# Select data
data = df[(df['scenario'] == scenarios) &
          (df['country'].isin(countries)) &
          (df['category'].isin(categories)) &
          (df['entity'].isin(entities))]

# Group data
grouped_data = data.groupby(dis_aggregation).sum()

# Restrict the data to just that selected by the slider
grouped_data = grouped_data.loc[:, str(slider_range[0]):str(slider_range[1])]

# If offset selected, subtract temperature at beginning of date range from the
# rest of the timeseries
if offset:
    start_temps = grouped_data[str(slider_range[0])]
    grouped_data = grouped_data.sub(start_temps, axis='index')

# Add 'total' of data to data
if grouped_data.shape[0] > 1:
    grouped_data = grouped_data.T
    grouped_data["TOTAL"] = grouped_data.sum(axis=1)
    grouped_data = grouped_data.T

####
# CREATE ALTAIR PLOTS
####

# Transform from wide data to long data (altair likes long data)
alt_data = grouped_data.T.reset_index().melt(id_vars=["index"])
alt_data = alt_data.rename(columns={"index": "year", 'value': 'warming'})

# Create colour mapping that accounts for a black 'TOTAL' line if multiple
# lines are present
c_domain = list(grouped_data.index)
c_range = colour_range(c_domain)

chart_0 = (
    alt.Chart(alt_data)
       .mark_line(opacity=1)
       .encode(x=alt.X("year:T", title=None),
               y=alt.Y("warming:Q",
                       title=f'warming relative to {slider_range[0]} (K)'),
               color=alt.Color(dis_aggregation,
                               scale=alt.Scale(domain=c_domain, range=c_range))
               )
       .properties(height=500)
       .configure_legend(orient='top-left')
       .encode(tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
)

st.altair_chart(chart_0, use_container_width=True)


# # CREATE MATPLOTLIB PLOTS
# fig, ax = plt.subplots()
# plt.style.use('seaborn-whitegrid')
# # matplotlib.rcParams.update(
# #     {'font.size': 11, 'font.family': 'Roboto', 'font.weight': 'light',
# #      'axes.linewidth': 0.5, 'axes.titleweight': 'regular',
# #      'axes.grid': True, 'grid.linewidth': 0.5,
# #      'grid.color': 'gainsboro',
# #      'figure.dpi': 200, 'figure.figsize': (15, 10),
# #      'figure.titlesize': 17,
# #      'figure.titleweight': 'light',
# #      'legend.frameon': False}
# #                             )

# times = [int(time) for time in grouped_data.columns]

# for x in list(set(data[dis_aggregation])):
#     if x == 'TOTAL':
#         ax.plot(times, grouped_data.loc[x].values.squeeze(),
#                 label=x, color='black')
#     else:
#         ax.plot(times, grouped_data.loc[x].values.squeeze(), label=x)

# ax.legend()
# ax.set_ylabel(f'warming relative to {slider_range[0]} (K)')
# ax.set_xlim(slider_range)

# st.pyplot(fig)
# plt.close()


# selected_data_expander = st.expander("View Full Selected Data")
# selected_data_expander.write(data)
# grouped_data_expander = st.expander("View grouped data")
# grouped_data_expander.write(grouped_data)
