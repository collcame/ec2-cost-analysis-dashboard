import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="EC2 Cost Analysis Dashboard",
    page_icon="☁️",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("☁️ Amazon EC2 Cost Analysis & Prediction Dashboard")

st.write(
    "Analyze Amazon EC2 pricing data and predict On-Demand "
    "instance costs using linear regression."
)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("ec2dataset.csv")


data = load_data()

# Store original dataset dimensions before feature engineering
original_row_count = len(data)
original_column_count = len(data.columns)


# ---------------------------------------------------------
# Clean Pricing Data
# ---------------------------------------------------------

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace("[$, hourly]", "", regex=True),
        errors="coerce"
    )


# ---------------------------------------------------------
# Prepare Numeric Features
# ---------------------------------------------------------

data["Instance Memory Numeric"] = pd.to_numeric(
    data["Instance Memory"].str.replace(" GiB", ""),
    errors="coerce"
)

data["vCPUs Numeric"] = pd.to_numeric(
    data["vCPUs"].str.extract(r"(\d+)", expand=False),
    errors="coerce"
)


# ---------------------------------------------------------
# IQR Outlier Detection
# ---------------------------------------------------------

Q1 = data["On Demand"].quantile(0.25)
Q3 = data["On Demand"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_on_demand = data[
    (data["On Demand"] < lower_bound)
    | (data["On Demand"] > upper_bound)
]


# ---------------------------------------------------------
# Prepare Data for Machine Learning
# Remove On-Demand Outliers Before Model Training
# ---------------------------------------------------------

# First remove rows that cannot be used by the regression model
model_data_before_outliers = data.dropna(
    subset=[
        "On Demand",
        "Instance Memory Numeric",
        "vCPUs Numeric"
    ]
).copy()


# Remove On-Demand price outliers using the IQR bounds
model_data = model_data_before_outliers[
    (model_data_before_outliers["On Demand"] >= lower_bound)
    & (model_data_before_outliers["On Demand"] <= upper_bound)
].copy()


# Number of regression rows removed as outliers
outliers_removed_from_model = (
    len(model_data_before_outliers)
    - len(model_data)
)


X = model_data[
    ["Instance Memory Numeric", "vCPUs Numeric"]
]

y = model_data["On Demand"]


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5


# ---------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "EC2 Instances",
    original_row_count
)

col2.metric(
    "Dataset Columns",
    original_column_count
)

col3.metric(
    "Pricing Models",
    len(cost_columns)
)


# ---------------------------------------------------------
# Dataset Table
# ---------------------------------------------------------

st.subheader("EC2 Dataset")

display_columns = [
    "Name",
    "API Name",
    "Instance Memory",
    "vCPUs",
    "Instance Storage",
    "Network Performance",
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

st.dataframe(
    data[display_columns],
    width="stretch"
)


# ---------------------------------------------------------
# Pricing Analysis
# ---------------------------------------------------------

st.divider()

st.header("EC2 Pricing Analysis")

st.write(
    "Compare hourly EC2 costs across On-Demand, Reserved, "
    "Spot, and Windows pricing options."
)


# ---------------------------------------------------------
# Pricing Summary
# ---------------------------------------------------------

st.subheader("Pricing Summary")

summary = (
    data[cost_columns]
    .describe()
    .round(4)
)

st.dataframe(
    summary,
    width="stretch"
)


# ---------------------------------------------------------
# Overall Cost Distribution
# ---------------------------------------------------------

st.subheader("Cost Distribution")

fig, ax = plt.subplots(
    figsize=(12, 6)
)

sns.boxplot(
    data=data[cost_columns],
    ax=ax
)

ax.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)"
)

ax.set_ylabel(
    "Cost (USD)"
)

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# IQR Outlier Analysis
# ---------------------------------------------------------

st.subheader("On-Demand Cost Outliers")

st.write(
    "Potential On-Demand pricing outliers are identified "
    "using the Interquartile Range (IQR) method. These outliers "
    "remain visible in the exploratory analysis but are excluded "
    "from regression model training."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "On-Demand Outliers",
    len(outliers_on_demand)
)

col2.metric(
    "Lower IQR Bound",
    f"${lower_bound:.4f}"
)

col3.metric(
    "Upper IQR Bound",
    f"${upper_bound:.4f}"
)


with st.expander("View On-Demand Cost Outliers"):

    st.dataframe(
        outliers_on_demand[
            [
                "Name",
                "API Name",
                "Instance Memory",
                "vCPUs",
                "On Demand"
            ]
        ].sort_values(
            "On Demand",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )


# ---------------------------------------------------------
# Lowest-Cost EC2 Instances
# ---------------------------------------------------------

st.subheader("10 Lowest-Cost EC2 Instances")

lowest_cost_instances = (
    data[
        [
            "Name",
            "API Name",
            "On Demand",
            "Linux Reserved cost"
        ]
    ]
    .dropna()
    .sort_values("On Demand")
    .head(10)
)

st.dataframe(
    lowest_cost_instances,
    width="stretch",
    hide_index=True
)


# ---------------------------------------------------------
# Reserved vs On-Demand Savings
# ---------------------------------------------------------

st.subheader("On-Demand vs Reserved Savings")

savings_data = data[
    [
        "Name",
        "API Name",
        "On Demand",
        "Linux Reserved cost"
    ]
].dropna().copy()

savings_data["Savings"] = (
    savings_data["On Demand"]
    - savings_data["Linux Reserved cost"]
)

savings_data["Savings %"] = (
    savings_data["Savings"]
    / savings_data["On Demand"]
) * 100

savings_data = savings_data.sort_values(
    "On Demand"
)

st.dataframe(
    savings_data.head(10).round(4),
    width="stretch",
    hide_index=True
)


# ---------------------------------------------------------
# Instance Family Analysis
# ---------------------------------------------------------

st.divider()

st.header("Instance Family Analysis")

st.write(
    "Select an EC2 instance family to explore its pricing "
    "distribution and compare pricing options."
)


# Extract the EC2 family from the API name
data["Instance Family"] = (
    data["API Name"]
    .str.split(".")
    .str[0]
    .str.upper()
)

families = sorted(
    data["Instance Family"]
    .dropna()
    .unique()
)


# Default to T3 if available
default_family_index = (
    families.index("T3")
    if "T3" in families
    else 0
)

selected_family = st.selectbox(
    "Select Instance Family",
    families,
    index=default_family_index
)


# ---------------------------------------------------------
# Filter Selected Family
# ---------------------------------------------------------

family_data = data[
    data["Instance Family"]
    == selected_family
]

st.write(
    f"**{len(family_data)} instances found in the "
    f"{selected_family} family.**"
)


# ---------------------------------------------------------
# Family Pricing Summary
# ---------------------------------------------------------

st.subheader(
    f"{selected_family} Pricing Summary"
)

family_summary = (
    family_data[cost_columns]
    .describe()
    .round(4)
)

st.dataframe(
    family_summary,
    width="stretch"
)


# ---------------------------------------------------------
# Family Cost Distribution
# ---------------------------------------------------------

st.subheader(
    f"{selected_family} Cost Distribution"
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

sns.boxplot(
    data=family_data[cost_columns],
    showmeans=True,
    ax=ax
)

ax.set_title(
    f"Cost Distribution for {selected_family} Instances"
)

ax.set_ylabel(
    "Cost (USD)"
)

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# Lowest-Cost Instances in Selected Family
# ---------------------------------------------------------

st.subheader(
    f"Lowest-Cost {selected_family} Instances"
)

family_comparison = (
    family_data[
        [
            "Name",
            "API Name",
            "On Demand",
            "Linux Reserved cost"
        ]
    ]
    .dropna()
    .sort_values("On Demand")
)

st.dataframe(
    family_comparison.head(10),
    width="stretch",
    hide_index=True
)


# ---------------------------------------------------------
# Regression Model Performance
# ---------------------------------------------------------

st.divider()

st.header("Regression Model Performance")

st.write(
    "The linear regression model predicts EC2 On-Demand cost "
    "using instance memory and number of vCPUs. On-Demand cost "
    "outliers identified using the IQR method are excluded from "
    "model training to reduce the influence of extreme prices."
)


# ---------------------------------------------------------
# Training Information
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Samples Before Filtering",
    len(model_data_before_outliers)
)

col2.metric(
    "Outliers Removed",
    outliers_removed_from_model
)

col3.metric(
    "Training Samples",
    len(X_train)
)

col4.metric(
    "Testing Samples",
    len(X_test)
)


# ---------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Absolute Error (MAE)",
    f"${mae:.2f}"
)

col2.metric(
    "Mean Squared Error (MSE)",
    f"{mse:.2f}"
)

col3.metric(
    "Root Mean Squared Error (RMSE)",
    f"${rmse:.2f}"
)


# ---------------------------------------------------------
# Model Coefficients
# ---------------------------------------------------------

with st.expander("View Model Details"):

    st.write(
        f"**Intercept:** {model.intercept_:.6f}"
    )

    st.write(
        "**Instance Memory Coefficient:** "
        f"{model.coef_[0]:.6f}"
    )

    st.write(
        "**vCPU Coefficient:** "
        f"{model.coef_[1]:.6f}"
    )

    st.write(
        "**IQR Filtering Range Used for Training:** "
        f"${lower_bound:.4f} to ${upper_bound:.4f} per hour"
    )

    st.write(
        "**Regression Samples After Outlier Removal:** "
        f"{len(model_data)}"
    )

    st.write(
        "The model uses Instance Memory and vCPU count "
        "as predictors of hourly EC2 On-Demand cost."
    )


# ---------------------------------------------------------
# Actual vs Predicted Cost Plot
# ---------------------------------------------------------

st.subheader(
    "Actual vs Predicted On-Demand Costs"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

ax.plot(
    [
        y_test.min(),
        y_test.max()
    ],
    [
        y_test.min(),
        y_test.max()
    ],
    linestyle="--"
)

ax.set_title(
    "Actual vs Predicted On-Demand Costs After Outlier Removal"
)

ax.set_xlabel(
    "Actual On-Demand Cost ($/hour)"
)

ax.set_ylabel(
    "Predicted On-Demand Cost ($/hour)"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# Model Limitations
# ---------------------------------------------------------

with st.expander("Model Interpretation and Limitations"):

    st.write(
        "On-Demand price outliers are removed from the regression "
        "training data using the IQR method. This reduces the "
        "influence of extremely expensive EC2 instances on the "
        "regression line."
    )

    st.write(
        "The full dataset, including outliers, is still retained "
        "for exploratory analysis elsewhere in the dashboard."
    )

    st.write(
        "The regression model still uses only two predictors: "
        "instance memory and number of vCPUs."
    )

    st.write(
        "EC2 pricing is influenced by additional factors such "
        "as instance family, processor architecture, storage, "
        "network performance, operating system, and other "
        "instance characteristics."
    )

    st.write(
        "Therefore, outlier removal can improve the model without "
        "making it a complete representation of EC2 pricing."
    )

    st.write(
        f"The filtered model's MAE is ${mae:.2f} per hour and its "
        f"RMSE is ${rmse:.2f} per hour."
    )


# ---------------------------------------------------------
# EC2 Cost Predictor
# ---------------------------------------------------------

st.divider()

st.header("EC2 Cost Predictor")

st.write(
    "Enter an EC2 configuration below to estimate its "
    "On-Demand hourly cost using the linear regression model "
    "trained after IQR outlier removal."
)

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Predictor Inputs
# ---------------------------------------------------------

with col1:

    memory_input = st.number_input(
        "Instance Memory (GiB)",
        min_value=0.5,
        max_value=32768.0,
        value=4.0,
        step=0.5
    )


with col2:

    vcpu_input = st.number_input(
        "vCPUs",
        min_value=1,
        max_value=448,
        value=2,
        step=1
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "Predict On-Demand Cost",
    type="primary"
):

    prediction_input = pd.DataFrame(
        [
            [
                memory_input,
                vcpu_input
            ]
        ],
        columns=[
            "Instance Memory Numeric",
            "vCPUs Numeric"
        ]
    )

    predicted_cost = model.predict(
        prediction_input
    )[0]


    # -----------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------

    st.subheader(
        "Prediction Result"
    )


    if predicted_cost >= 0:

        monthly_cost = (
            predicted_cost * 730
        )

        annual_cost = (
            predicted_cost * 8760
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Hourly Cost",
            f"${predicted_cost:.4f}"
        )

        col2.metric(
            "Estimated Monthly Cost",
            f"${monthly_cost:,.2f}"
        )

        col3.metric(
            "Estimated Annual Cost",
            f"${annual_cost:,.2f}"
        )


    else:

        st.metric(
            "Raw Model Prediction",
            f"${predicted_cost:.4f} / hour"
        )

        st.warning(
            "The model produced a negative price, which is not "
            "realistic for an EC2 instance. Although IQR outliers "
            "were removed before training, the model still uses "
            "only memory and vCPUs and may be inaccurate for some "
            "configurations."
        )


    # -----------------------------------------------------
    # Comparable Real EC2 Instances
    # -----------------------------------------------------

    comparable_instances = data[
        (data["Instance Memory Numeric"] == memory_input)
        & (data["vCPUs Numeric"] == vcpu_input)
        & (data["On Demand"].notna())
    ][
        [
            "Name",
            "API Name",
            "Instance Memory",
            "vCPUs",
            "On Demand"
        ]
    ].sort_values(
        "On Demand"
    )


    st.subheader(
        "Comparable Instances in Dataset"
    )


    if not comparable_instances.empty:

        st.write(
            "These EC2 instances in the original dataset have "
            "the same memory and vCPU configuration:"
        )

        st.dataframe(
            comparable_instances,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No EC2 instances in the dataset have this exact "
            "memory and vCPU configuration."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "EC2 Cost Analysis Dashboard | "
    "Cloud Economics | Fall 2026"
)