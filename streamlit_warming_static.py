"""Streamlit web app exploring pre-calculated (static) warming dataset."""
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

    return df


def prepare_data(df, scenarios, countries, categories, entities,
                 dis_aggregation, date_range, offset, include_total):
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
d_set = side_expand.selectbox('Choose dataset',
                             ['Emissions', 'Warming Impact'], 1)

if d_set == 'Emissions':
    df = load_data("./data/PRIMAP-hist_v2.2_19-Jan-2021.csv")
    # df = load_data(
    # 'https://zenodo.org/record/4479172/files/PRIMAP-hist_v2.2_19-Jan-2021.csv')
elif d_set == 'Warming Impact':
    df = load_data("./data/warming-contributions-data_PRIMAP-format.csv")
# elif d_set == 'Upload Own Data':
#     df = load_data(side_expand.file_uploader('upload emissions'))


####
# SELECT SUBSETS OF DATA
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
    sorted(list(set(df['entity']))),
    sorted(list(set(df['entity'])))
))

# year_expander = st.expander("Year Range Selection")
# with year_expander:
c1, c2 = st.columns([3, 1])
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

# st.markdown("---")

####
# CREATE ALTAIR PLOTS
####

# Select data
include_sum = True
grouped_data = prepare_data(df, scenarios, countries, categories, entities,
                            dis_aggregation, date_range, offset, include_sum)

# Transform from wide data to long data (altair likes long data)
alt_data = grouped_data.T.reset_index().melt(id_vars=["index"])
alt_data = alt_data.rename(columns={"index": "year", 'value': 'warming'})

# Create colour mapping that accounts for a black 'SUM' line if multiple
# lines are present
# Note, sorting this here, means it matches the order returned by the
# (sorted) output form the selection widgets; som colours for plots match.
c_domain = sorted(list(grouped_data.index))
# if include_sum and len(c_domain) > 1:
if 'SUM' in c_domain:
    c_domain.append(c_domain.pop(c_domain.index('SUM')))
c_range = colour_range(c_domain, include_sum, dis_aggregation)

warming_start = date_range[0] if offset else 1850

c1.subheader(f'warming relative to {warming_start} (K)')
chart_0 = (
    alt.Chart(alt_data)
       .mark_line(opacity=0.9)
       .encode(x=alt.X("year:T", title=None),
               y=alt.Y("warming:Q",
                       title=None,
                    #    stack=None
                       ),
               color=alt.Color(dis_aggregation,
                               scale=alt.Scale(domain=c_domain, range=c_range))
               )
    #    .properties(height=500)
       .configure_legend(orient='top-left')
       .encode(tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
)

c1.altair_chart(chart_0, use_container_width=True)


# # CREATE MATPLOTLIB PLOTS
# # fig, ax = plt.subplots()
# # plt.style.use('seaborn-whitegrid')
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

# # times = [int(time) for time in grouped_data.columns]

# # for x in list(set(data[dis_aggregation])):
# #     if x == 'SUM':
# #         ax.plot(times, grouped_data.loc[x].values.squeeze(),
# #                 label=x, color='black')
# #     else:
# #         ax.plot(times, grouped_data.loc[x].values.squeeze(), label=x)

# # ax.legend()
# # ax.set_ylabel(f'warming relative to {date_range[0]} (K)')
# # ax.set_xlim(date_range)

# # st.pyplot(fig)
# # plt.close()


# # selected_data_expander = st.expander("View Full Selected Data")
# # selected_data_expander.write(data)
# # grouped_data_expander = st.expander("View grouped data")
# # grouped_data_expander.write(grouped_data)


# ####
# Make Sankey Diagram
####
c3, c4 = st.columns([3, 1])
c4.subheader(' ')

# NOTE: We force the offset to the start of the selected date range, so this
# plot will always show the warming between the two dates in the range. This
# seems like intuitive behaviour. Therefore, the offset only applies to the
# line chart to change the relateive start date for that...
sankey_cs = prepare_data(df, scenarios, countries, categories, entities,
                         ['country', 'category'], date_range, True, False)
sankey_sg = prepare_data(df, scenarios, countries, categories, entities,
                         ['category', 'entity'], date_range, True, False)
sankey_gc = prepare_data(df, scenarios, countries, categories, entities,
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
