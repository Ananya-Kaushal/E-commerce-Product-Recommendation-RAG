import streamlit as st
import pandas as pd
from rag.ingest import build_or_load_db
from rag.retriever import Retriever
from rag.recommender import rerank, template_generate
from rag.compare import compare_products
from rag.sentiment import sentiment_score

st.set_page_config(page_title="E-commerce RAG Recommender", layout="wide")

st.title("🛒 E-commerce Product Recommendation RAG (Demo)")

# Sidebar for user preferences
with st.sidebar:
    st.header("Preferences")
    category = st.selectbox("Preferred category", ["Any","Smartphones","Laptops","Headphones","Camera","Consoles","Smart Home","Wearables"])
    min_price = st.number_input("Min price (Rs)", value=0, step=50)
    max_price = st.number_input("Max price (Rs)", value=2000, step=50)
    keywords = st.text_input("Keywords (comma-separated)", value="battery, camera")

    st.caption("These preferences influence re-ranking.")
    if st.button("Rebuild Vector DB"):
        build_or_load_db()
        st.success("Vector DB rebuilt successfully ✅")

# Query input
st.info("First run? Click **Rebuild Vector DB** in the sidebar to create embeddings.")
query = st.text_input("Describe what you want", "noise cancelling headphones for travel")
k = st.slider("Number of results", 1, 10, 5)

# Recommendation button
if st.button("Search & Recommend"):
    build_or_load_db()
    r = Retriever()
    results = r.search(query, k=k)

    prefs = {
        "category": None if category == "Any" else category,
        "min_price": min_price,
        "max_price": max_price,
        "keywords": [w.strip() for w in keywords.split(",") if w.strip()],
    }

    ranked = rerank(results, prefs, top_k=k)

    # Show recommendations
    st.subheader("Top Recommendations")
    for item in ranked:
        md = item["metadata"]
        st.markdown(f"**{md['title']}** — {md['category']} — Rs {md['price']}")
        st.caption(item["document"][:300] + ("..." if len(item["document"])>300 else ""))

    # Generated summary
    st.subheader("Generated Summary")
    st.code(template_generate(query, ranked))

    # Product comparison
    st.subheader("Comparison")
    prod_ids = [i["metadata"]["product_id"] for i in ranked]
    prods = pd.read_csv("data/products.csv")
    specs = pd.read_csv("data/specs.csv")
    comp = compare_products(prods, specs, prod_ids[:3])
    st.dataframe(comp, use_container_width=True)

    # Review sentiment
    st.subheader("Review Sentiment (sample)")
    revs = pd.read_csv("data/reviews.csv")
    sample = revs[revs['product_id'].isin(prod_ids[:3])].copy()
    sample['sentiment'] = sample['review_text'].apply(sentiment_score)
    st.dataframe(sample[['product_id','stars','review_text','sentiment']], use_container_width=True)
