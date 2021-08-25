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

# left_filters, right_filters = st.columns(2)

# countries = left_filters.multiselect(
#     "Choose countries and/or regions",
#     list(set(df['country'])),
#     ['EU28', 'USA', 'AOSIS']
#     # ['EU28']
# )

# scenarios = left_filters.selectbox(
#     "Choose scenario",
#     list(set(df['scenario'])),
#     index=list(set(df['scenario'])).index('HISTCR')
# )

# categories = right_filters.multiselect(
#     "Choose sectors",
#     list(set(df['category'])),
#     ['IPC1', 'IPC2', 'IPCMAG', 'IPC4']
# )


# entities = right_filters.multiselect(
#     "Which gases would you like to consider?",
#     sorted(list(set(df['entity']))),
#     sorted(list(set(df['entity'])))
# )


# aggregation = st.multiselect(
#     "Choose the grouping/s that you'd like",
#     ['country', 'category', 'entity'],
#     ['country']
# )

left_filters, right_filters = st.columns(2)

countries = left_filters.selectbox(
    "Choose countries and/or regions",
    list(set(df['country'])),
    index=list(set(df['country'])).index('EU28')
)

scenarios = left_filters.selectbox(
    "Choose scenario",
    list(set(df['scenario'])),
    index=list(set(df['scenario'])).index('HISTCR')
)

categories = right_filters.selectbox(
    "Choose sectors",
    list(set(df['category'])),
    index=list(set(df['category'])).index('IPCM0EL')
)


entities = right_filters.multiselect(
    "Which gases would you like to consider?",
    sorted(list(set(df['entity']))),
    sorted(list(set(df['entity'])))
)


st.markdown('## Warming Dataset')
# expander = st.expander("View Data")
# expander.write(df.head())
data = df[
            (df['scenario'] == scenarios) &
            (df['country'] == countries) &
            (df['category'] == categories) &
            (df['entity'].isin(entities))
            ]

st.write(data)
# st.table(data)
# if not aggregation:
    
# else:
#     data = data.groupby(aggregation).sum()
#     st.write(data)
# st.write(data.index)


# st.write('## Test for EU28 emissions')
test_select = df[
    (df['scenario'] == scenarios) &
    (df['country'] == countries) &
    (df['category'] == categories)
    ]
# st.write(test_select)
times = np.arange(2018-1850+1)+1850
slider_range = st.slider("Plot Date Range", value=[1990, 2018], min_value=1850, max_value=2018)
offset = st.checkbox(f"Offset from your selected start year {slider_range[0]}?", value=True)

index = int(np.where(times == slider_range[0])[0])
total = np.zeros_like(times, dtype='float64')[index:]
times = times[index:]

for entity in entities:
    test_gas = test_select[test_select['entity'] == entity].loc[:, '1850':].values.squeeze()
    if offset:
        plt.plot(times, (test_gas-test_gas[index])[index:],
                 label=entity)
        total = total + (test_gas-test_gas[index])[index:]
        # st.write(total)
    else:
        plt.plot(times, test_gas[index:], label=entity)
        total = total + (test_gas)[index:]

plt.plot(times, total, label='Total', color='black')
plt.legend()

plt.xlim(slider_range)
# plt.ylim(-0.02, 0.06)
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('seaborn-whitegrid')
plt.title(f'{countries} contribution to warming in {categories} sector')
st.pyplot()
plt.close()
