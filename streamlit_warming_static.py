"""Hello World implementation of streamlit."""
import streamlit as st
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import matplotlib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Warming Contributions Explorer",
    page_icon="https://images.emojiterra.com/twitter/512px/1f321.png",
    layout="wide"
)

st.markdown(
    """
    # Contributions to Historical Warming
    *across scenarios, countries, sectors, and the main gases*
    """
)


@st.cache(show_spinner=False)
def load_data(file):
    """Load the dataset; function approach allows streamlit caching."""
    return pd.read_csv(file)


# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
df = load_data("./data/warming-contributions-data_PRIMAP-format.csv")
# Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')

st.markdown('## Select from country, sector, and gas')

left_filters, right_filters = st.columns(2)

scenarios = left_filters.selectbox(
    "Choose scenario",
    list(set(df['scenario'])),
    index=list(set(df['scenario'])).index('HISTCR')
)

countries = left_filters.multiselect(
    "Choose countries and/or regions",
    list(set(df['country'])),
    ['EU28', 'USA', 'AOSIS']
    # ['EU28']
)

not_country = ['EARTH', 'ANNEXI', 'NONANNEXI', 'AOSIS',
               'BASIC', 'EU28', 'LDC', 'UMBRELLA']
iso_country = list(set(df['country']) - set(not_country))

categories = right_filters.multiselect(
    "Choose sectors",
    list(set(df['category'])),
    ['IPC1', 'IPC2', 'IPCMAG', 'IPC4']
)

entities = right_filters.multiselect(
    "Which gases would you like to consider?",
    sorted(list(set(df['entity']))),
    sorted(list(set(df['entity'])))
)


dis_aggregation = st.selectbox(
    "Choose the disaggregation that you'd like to view",
    ['country', 'category', 'entity'],
    index=['country', 'category', 'entity'].index('country')
)
aggregated = sorted(list(set(['country', 'category', 'entity']) -
                         set([dis_aggregation])))


"""
## Explore Warming
"""

data = df[(df['scenario'] == scenarios) &
          (df['country'].isin(countries)) &
          (df['category'].isin(categories)) &
          (df['entity'].isin(entities))
          ]

# Get time range for plots
year_expander = st.expander("Year Range Selection")
with year_expander:
    slider_range = st.slider(
        "Plot Date Range", value=[1850, 2018], min_value=1850, max_value=2018)
    offset = st.checkbox(
        f"Offset from your selected start year {slider_range[0]}?", value=False)


# Select data grouping to use
grouped_data = data.groupby(dis_aggregation).sum()


# alt.renderers.set_embed_options(actions=False)
# Without this following text it isn't possible to see hover tooltips in the
# fullscreen version of the plot
# https://discuss.streamlit.io/t/tool-tips-in-fullscreen-mode-for-charts/6800/10
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
             unsafe_allow_html=True)

# PREPARE DATA

# Restrict the data to just that selected by the slider
grouped_data = (grouped_data.loc[:, str(slider_range[0]):str(slider_range[1])])

# Add 'total' of data to data.
grouped_data = grouped_data.T
grouped_data["TOTAL"] = grouped_data.sum(axis=1)
grouped_data = grouped_data.T

# If offset selected, subtract temperature at beginning of date range from the
# rest of the timeseries
if offset:
    start_temps = grouped_data[str(slider_range[0])]
    grouped_data = grouped_data.sub(start_temps, axis='index')


# CREATE ALTAIR PLOTS

# Transform from wide data to long data (altair likes long data)
alt_data = grouped_data.T.reset_index().melt(id_vars=["index"])
alt_data = alt_data.rename(columns={"index": "year", 'value': 'warming'})

chart_1 = (
    alt.Chart(alt_data[alt_data[dis_aggregation] != 'TOTAL'])
       .mark_line(opacity=1)
       .encode(
           x=alt.X("year:T", title=None),
           y=alt.Y("warming:Q",
                   title=f'warming relative to {slider_range[0]} (K)'),
           color=dis_aggregation,
    )
)

chart_2 = (
    alt.Chart(alt_data[alt_data[dis_aggregation] == 'TOTAL'])
       .mark_line(opacity=1, color='black')
       .encode(
           x=alt.X("year:T", title=None),
           y=alt.Y("warming:Q",
                   title=f'warming relative to {slider_range[0]} (K)'),
           opacity=alt.Opacity(dis_aggregation, legend=alt.Legend(title="\n"))
    )
)

chart = (alt.layer(chart_1, chart_2)
            .properties(height=500)
            .configure_legend(orient='top-left')
            .encode(tooltip=[(dis_aggregation + ':N'), 'warming:Q'])
         )

st.altair_chart(chart, use_container_width=True)


# CREATE MATPLOTLIB PLOTS
fig, ax = plt.subplots()
plt.style.use('seaborn-whitegrid')
# matplotlib.rcParams.update(
#     {'font.size': 11, 'font.family': 'Roboto', 'font.weight': 'light',
#      'axes.linewidth': 0.5, 'axes.titleweight': 'regular',
#      'axes.grid': True, 'grid.linewidth': 0.5,
#      'grid.color': 'gainsboro',
#      'figure.dpi': 200, 'figure.figsize': (15, 10),
#      'figure.titlesize': 17,
#      'figure.titleweight': 'light',
#      'legend.frameon': False}
#                             )

times = [int(time) for time in grouped_data.columns]

for x in list(set(data[dis_aggregation])):
    line = grouped_data.loc[x].values.squeeze()
    ax.plot(times, line, label=x)

ax.plot(times, grouped_data.loc['TOTAL'].values.squeeze(),
        label='Total', color='black')

ax.legend()
ax.set_ylabel(f'warming relative to {slider_range[0]} (K)')
ax.set_xlim(slider_range)

st.pyplot(fig)
plt.close()




# selected_data_expander = st.expander("View Full Selected Data")
# selected_data_expander.write(data)
# grouped_data_expander = st.expander("View grouped data")
# grouped_data_expander.write(grouped_data)
