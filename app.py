from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CarValue | Used car insights",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "car_price_regression_data (1).csv"
MODEL_FILE = BASE_DIR / "cars_model.pkl"
FEATURES = [
    "Brand",
    "Fuel_Type",
    "Transmission",
    "Engine_CC",
    "Owner_Type",
    "Safety_Rating",
]
CATEGORICAL_FEATURES = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"]
REQUIRED_COLUMNS = {
    *FEATURES,
    "Model",
    "Year",
    "Kilometers_Driven",
    "Price_Lakh",
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


if not DATA_FILE.is_file():
    st.error(f"Dataset file not found: {DATA_FILE.name}")
    st.stop()
if not MODEL_FILE.is_file():
    st.error(f"Model file not found: {MODEL_FILE.name}")
    st.stop()

cars = load_data(DATA_FILE)
missing_columns = REQUIRED_COLUMNS.difference(cars.columns)
if missing_columns:
    st.error(f"The dataset is missing required columns: {', '.join(sorted(missing_columns))}.")
    st.stop()
if cars.empty:
    st.error("The dataset contains no cars to display.")
    st.stop()

model = load_model(MODEL_FILE)
model_features = list(getattr(model, "feature_names_in_", FEATURES))
if model_features != FEATURES:
    st.error("The saved model's input features do not match the app's expected feature order.")
    st.stop()


st.markdown(
    """
    <style>
    .stApp { background: #f5f4ef; color: #1c2923; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMetric"] {
        background: #fffefa; border: 1px solid #e5e3d9;
        padding: 1rem 1.1rem; border-radius: 14px;
    }
    .hero {
        background: linear-gradient(115deg, #153d32 0%, #245b49 68%, #38745b 100%);
        color: #fffaf0; padding: 2.2rem 2.5rem; border-radius: 20px;
        margin: .5rem 0 1.4rem;
    }
    .hero h1 {
        font-family: Georgia, serif; font-size: clamp(2.3rem, 5vw, 3.4rem);
        letter-spacing: -.04em; margin: 0;
    }
    .hero p { color: #d5e4d8; font-size: 1.05rem; margin: .55rem 0 0; }
    h2, h3 { font-family: Georgia, serif; letter-spacing: -.02em; }
    </style>
    <section class="hero">
      <h1>CarValue</h1>
      <p>A clearer view of the used-car market. Explore listings and get a data-backed estimate.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

average_price = cars["Price_Lakh"].mean()
median_price = cars["Price_Lakh"].median()
most_common_brand = cars["Brand"].mode().iat[0]
latest_year = int(cars["Year"].max())

metrics = st.columns(4)
metrics[0].metric("Cars in the market", f"{len(cars):,}")
metrics[1].metric("Average asking price", f"₹{average_price:.1f} lakh")
metrics[2].metric("Most listed brand", most_common_brand)
metrics[3].metric("Newest model year", str(latest_year))

market_tab, browse_tab, estimate_tab = st.tabs(
    ["Market overview", "Browse cars", "Estimate a price"]
)

with market_tab:
    st.subheader("The market, at a glance")
    st.caption(
        f"Prices span ₹{cars['Price_Lakh'].min():.1f}–₹{cars['Price_Lakh'].max():.1f} lakh; "
        f"the median listing is ₹{median_price:.1f} lakh."
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Average price by brand")
        brand_prices = (
            cars.groupby("Brand")["Price_Lakh"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(brand_prices, color="#d87941", y_label="Price (₹ lakh)")
    with chart_right:
        st.markdown("#### Average price by model year")
        year_prices = cars.groupby("Year")["Price_Lakh"].mean().sort_index()
        st.line_chart(year_prices, color="#2f7762", y_label="Price (₹ lakh)")

    fuel_mix = cars["Fuel_Type"].value_counts().rename_axis("Fuel type").to_frame("Listings")
    st.markdown("#### Listings by fuel type")
    st.bar_chart(fuel_mix, color="#73977a", horizontal=True)

with browse_tab:
    st.subheader("Find your next car")
    st.caption("Combine filters to narrow the listings. Leave a category empty to include all options.")

    with st.container(border=True):
        filter_columns = st.columns([1.15, 1, 1, 1.15])
        with filter_columns[0]:
            search = st.text_input("Search make or model", placeholder="e.g. Honda, Creta")
        with filter_columns[1]:
            brands = st.multiselect("Brand", sorted(cars["Brand"].unique()))
        with filter_columns[2]:
            fuels = st.multiselect("Fuel type", sorted(cars["Fuel_Type"].unique()))
        with filter_columns[3]:
            transmissions = st.multiselect(
                "Transmission", sorted(cars["Transmission"].unique())
            )

        range_columns = st.columns(2)
        min_year, max_year = int(cars["Year"].min()), int(cars["Year"].max())
        min_price, max_price = float(cars["Price_Lakh"].min()), float(
            cars["Price_Lakh"].max()
        )
        with range_columns[0]:
            year_range = st.slider(
                "Model year",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        with range_columns[1]:
            price_range = st.slider(
                "Price range (₹ lakh)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=1.0,
            )

    filtered_cars = cars.copy()
    if search.strip():
        query = search.strip()
        matches = (
            filtered_cars["Brand"].str.contains(query, case=False, na=False)
            | filtered_cars["Model"].str.contains(query, case=False, na=False)
        )
        filtered_cars = filtered_cars[matches]
    if brands:
        filtered_cars = filtered_cars[filtered_cars["Brand"].isin(brands)]
    if fuels:
        filtered_cars = filtered_cars[filtered_cars["Fuel_Type"].isin(fuels)]
    if transmissions:
        filtered_cars = filtered_cars[
            filtered_cars["Transmission"].isin(transmissions)
        ]
    filtered_cars = filtered_cars[
        filtered_cars["Year"].between(*year_range)
        & filtered_cars["Price_Lakh"].between(*price_range)
    ]

    result_left, result_right = st.columns([1, 4])
    result_left.metric("Matching cars", f"{len(filtered_cars):,}")
    display_columns = [
        "Brand",
        "Model",
        "Year",
        "Fuel_Type",
        "Transmission",
        "Engine_CC",
        "Kilometers_Driven",
        "Owner_Type",
        "Price_Lakh",
    ]
    st.dataframe(
        filtered_cars[display_columns].sort_values("Price_Lakh", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "Brand": st.column_config.TextColumn("Brand"),
            "Model": st.column_config.TextColumn("Model"),
            "Year": st.column_config.NumberColumn("Year", format="%d"),
            "Engine_CC": st.column_config.NumberColumn("Engine (CC)", format="%d"),
            "Kilometers_Driven": st.column_config.NumberColumn(
                "Kilometers driven", format="%d"
            ),
            "Price_Lakh": st.column_config.NumberColumn(
                "Price (₹ lakh)", format="₹%.2f"
            ),
        },
    )
    st.download_button(
        "Download matching cars",
        data=filtered_cars[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="carvalue_listings.csv",
        mime="text/csv",
        disabled=filtered_cars.empty,
    )

with estimate_tab:
    st.subheader("What could this car be worth?")
    st.caption(
        "Choose the car's details for an instant estimate. The model was trained on "
        "the six characteristics shown below."
    )

    categories = {
        column: sorted(cars[column].dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    with st.form("price_estimate_form"):
        input_columns = st.columns(3)
        with input_columns[0]:
            prediction_brand = st.selectbox("Brand", categories["Brand"])
            prediction_fuel = st.selectbox("Fuel type", categories["Fuel_Type"])
        with input_columns[1]:
            prediction_transmission = st.selectbox(
                "Transmission", categories["Transmission"]
            )
            prediction_engine = st.number_input(
                "Engine size (CC)",
                min_value=int(cars["Engine_CC"].min()),
                max_value=int(cars["Engine_CC"].max()),
                value=int(cars["Engine_CC"].median()),
                step=50,
            )
        with input_columns[2]:
            prediction_owner = st.selectbox("Owner type", categories["Owner_Type"])
            prediction_safety = st.slider(
                "Safety rating",
                min_value=float(cars["Safety_Rating"].min()),
                max_value=float(cars["Safety_Rating"].max()),
                value=float(cars["Safety_Rating"].median()),
                step=0.1,
            )
        submitted = st.form_submit_button(
            "Estimate car value", type="primary", width="stretch"
        )

    if submitted:
        feature_values = {
            "Brand": prediction_brand,
            "Fuel_Type": prediction_fuel,
            "Transmission": prediction_transmission,
            "Engine_CC": prediction_engine,
            "Owner_Type": prediction_owner,
            "Safety_Rating": prediction_safety,
        }
        encoded_values = {
            column: categories[column].index(feature_values[column])
            if column in CATEGORICAL_FEATURES
            else feature_values[column]
            for column in FEATURES
        }
        input_frame = pd.DataFrame([encoded_values], columns=FEATURES)
        estimate = max(float(model.predict(input_frame)[0]), 0.0)
        brand_median = cars.loc[
            cars["Brand"] == prediction_brand, "Price_Lakh"
        ].median()

        result_columns = st.columns(2)
        result_columns[0].metric("Estimated value", f"₹{estimate:.2f} lakh")
        result_columns[1].metric(
            f"{prediction_brand} listing median",
            f"₹{brand_median:.2f} lakh",
        )
        st.info(
            "This is a model-based estimate, not a formal valuation. Actual prices depend "
            "on condition, location, service history, and other factors not represented "
            "in the training data."
        )
