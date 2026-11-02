import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="P.F. Chang's Menu Analysis", page_icon="🍜", layout="wide")

DEFAULT_DATA = [
    {"Category": "Dessert", "Item Name": "The Great Wall of Chocolate", "Description": "Six rich layers of chocolate cake, chocolate frosting, semi-sweet chocolate chips"},
    {"Category": "Dessert", "Item Name": "Banana Spring Rolls", "Description": "Crispy bites of banana, caramel vanilla drizzle, coconut-pineapple ice cream"},
    {"Category": "Dessert", "Item Name": "New York-Style Cheesecake", "Description": "Graham cracker crust, fresh berries"},
    {"Category": "Beverage", "Item Name": "Chang's Coconut Cooler", "Description": "Coconut milk, coconut water, pure cane sugar, nutmeg"},
    {"Category": "Beverage", "Item Name": "Strawberry Cucumber Limonada", "Description": "Strawberry puree, fresh cucumber, house-made lemonade"},
    {"Category": "Beverage", "Item Name": "Organic Green Tea", "Description": "Hot brewed organic green tea"},
    {"Category": "Beverage", "Item Name": "Asian Pear Mojito", "Description": "Bacardi Limon Rum, Dekuyper Apple Pucker, fresh mint, lime juice, pineapple juice"},
    {"Category": "Beverage", "Item Name": "Red Sangria", "Description": "Red wine, Merlot, fresh fruit, orange liqueur"},
]

TEST_DATA = [
    {"Item Name": "Asian Pear Mojito", "Description": "Bacardi Limon Rum, Dekuyper Apple Pucker, fresh mint, lime juice, pineapple juice", "Actual": "Alcoholic"},
    {"Item Name": "Chang's Coconut Cooler", "Description": "Coconut milk, coconut water, pure cane sugar, nutmeg", "Actual": "Non-Alcoholic"},
    {"Item Name": "Strawberry Cucumber Limonada", "Description": "Strawberry puree, fresh cucumber, house-made lemonade", "Actual": "Non-Alcoholic"},
    {"Item Name": "Red Sangria", "Description": "Red wine, Merlot, fresh fruit, orange liqueur", "Actual": "Alcoholic"},
    {"Item Name": "Organic Green Tea", "Description": "Hot brewed organic green tea", "Actual": "Non-Alcoholic"},
    {"Item Name": "Organic Agave Margarita", "Description": "Patron Silver Tequila, organic agave nectar, fresh lime juice", "Actual": "Alcoholic"},
    {"Item Name": "Peach Boba Breeze", "Description": "Black tea, peach puree, popping boba pearls", "Actual": "Non-Alcoholic"},
    {"Item Name": "Japanese Old Fashioned", "Description": "Suntory Toki Whisky, orange bitters, cherry", "Actual": "Alcoholic"},
    {"Item Name": "House-Made Ginger Beer", "Description": "Fresh juiced ginger, pure cane sugar, club soda", "Actual": "Non-Alcoholic"},
    {"Item Name": "Spicy Passionfruit Margarita", "Description": "Cazadores Reposado Tequila, passionfruit puree, jalapeno", "Actual": "Alcoholic"},
]

ALCOHOL_KEYWORDS = [
    "rum", "pucker", "wine", "merlot", "liqueur", "tequila", "whisky",
    "patron", "cazadores", "vodka", "gin", "sake", "bacardi"
]

def predict_alcoholic(description):
    text = str(description).lower()
    return "Alcoholic" if any(word in text for word in ALCOHOL_KEYWORDS) else "Non-Alcoholic"

st.title("🍜 P.F. Chang's Menu Analysis")
st.caption("Desserts & Beverages • Data Analysis • Drink Classification")

with st.sidebar:
    st.header("Project Controls")
    uploaded = st.file_uploader("Upload your own menu CSV (optional)", type=["csv"])
    st.info("Without an uploaded file, the app uses the sample dataset from your notebook.")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        required = {"Category", "Item Name", "Description"}
        if not required.issubset(df.columns):
            st.error("CSV must contain these columns: Category, Item Name, Description.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = pd.DataFrame(DEFAULT_DATA)

df = df.copy()
df["Description"] = df["Description"].fillna("").astype(str)
df["Description_Length"] = df["Description"].str.len()

st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total menu items", len(df))
c2.metric("Categories", df["Category"].nunique())
c3.metric("Average description length", f"{df['Description_Length'].mean():.1f}" if len(df) else "0")

st.subheader("Explore Menu Dataset")
categories = sorted(df["Category"].dropna().astype(str).unique().tolist())
selected = st.multiselect("Filter by category", categories, default=categories)
filtered = df[df["Category"].astype(str).isin(selected)]
search_text = st.text_input("Search item name or description")
if search_text:
    mask = (
        filtered["Item Name"].astype(str).str.contains(search_text, case=False, na=False)
        | filtered["Description"].str.contains(search_text, case=False, na=False)
    )
    filtered = filtered[mask]
st.dataframe(filtered, use_container_width=True, hide_index=True)

csv_data = df.drop(columns=["Description_Length"]).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download menu dataset CSV", csv_data, "pf_changs_desserts_beverages.csv", "text/csv")

st.subheader("Exploratory Data Analysis")
left, right = st.columns(2)
with left:
    st.markdown("**Number of items by category**")
    counts = df["Category"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of items")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Description length by category**")
    fig, ax = plt.subplots()
    categories_for_plot = [g["Description_Length"].values for _, g in df.groupby("Category")]
    labels = [str(name) for name, _ in df.groupby("Category")]
    if categories_for_plot:
        ax.boxplot(categories_for_plot, labels=labels)
    ax.set_xlabel("Category")
    ax.set_ylabel("Description length (characters)")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with st.expander("View descriptive statistics"):
    st.dataframe(df["Description_Length"].describe().to_frame("Description_Length"))

st.subheader("Drink Classification")
st.write("Enter a drink description to classify it using the same keyword-based rule as your notebook.")
description = st.text_area(
    "Drink description",
    placeholder="Example: Tequila, fresh lime juice and agave nectar"
)
if st.button("Predict drink type", type="primary"):
    if not description.strip():
        st.warning("Please enter a drink description first.")
    else:
        result = predict_alcoholic(description)
        if result == "Alcoholic":
            st.warning(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
        st.caption("This is a simple keyword-based demonstration, not a trained machine-learning model. Results may be incorrect if ingredients are missing or described differently.")

st.subheader("Notebook Test Results")
test_df = pd.DataFrame(TEST_DATA)
test_df["Predicted"] = test_df["Description"].apply(predict_alcoholic)
accuracy = (test_df["Actual"] == test_df["Predicted"]).mean() * 100
m1, m2 = st.columns(2)
m1.metric("Test examples", len(test_df))
m2.metric("Keyword-rule accuracy on notebook test set", f"{accuracy:.1f}%")
st.dataframe(test_df, use_container_width=True, hide_index=True)

st.caption("Dataset note: the sample rows are the fallback examples included in the notebook. Website scraping may fail if the website blocks automated requests or changes its page structure.")
