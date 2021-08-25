"""Hello World implementation of streamlit."""
import streamlit as st
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
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

st.markdown('## Pick & Mix')

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


st.markdown('## Warming Dataset')

data = df[(df['scenario'] == scenarios) &
          (df['country'].isin(countries)) &
          (df['category'].isin(categories)) &
          (df['entity'].isin(entities))
          ]

expander = st.expander("View Full Selected Data")
expander.write(data)

# Select data grouping to use
grouped_data = data.groupby(dis_aggregation).sum()

# Get time range for plots
times = np.arange(2018-1850+1)+1850
year_expander = st.expander("Year Range Selection")
with year_expander:
    slider_range = st.slider(
        "Plot Date Range", value=[1990, 2018], min_value=1850, max_value=2018)
    offset = st.checkbox(
        f"Offset from your selected start year {slider_range[0]}?", value=True)
start_index = int(np.where(times == slider_range[0])[0])
end_index = int(np.where(times == slider_range[1])[0])
total = np.zeros_like(times, dtype='float64')[start_index:end_index]
times = times[start_index:end_index]

data_expander = st.expander("View grouped data")
data_expander.write(grouped_data)

# st.write(list(set(data[dis_aggregation])))
for x in list(set(data[dis_aggregation])):
    test_gas = grouped_data.loc[x].values.squeeze()

    if offset:
        plt.plot(times,
                 (test_gas-test_gas[start_index])[start_index:end_index],
                 label=x)
        total = total + (test_gas-test_gas[start_index])[start_index:end_index]
        # st.write(total)
    else:
        plt.plot(times, test_gas[start_index:end_index], label=x)
        total = total + (test_gas)[start_index:end_index]


plt.plot(times, total, label='Total', color='black')
plt.legend()

plt.xlim(slider_range)
# plt.ylim(-0.02, 0.06)
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('seaborn-whitegrid')

# plt.title(f'Contribution to warming\nfrom {aggregated[1]} sectors\nin {aggregated[0]} regions')
st.pyplot()
plt.close()
