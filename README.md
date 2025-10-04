import streamlit as st
import pandas as pd
import re, difflib

# LangChain / HuggingFace
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -------------------- Page Config --------------------
st.set_page_config(page_title="Restaurant Chatbot", page_icon="🍽️", layout="wide")
st.title("🍽️ Restaurant  chatbot")
st.markdown("Upload a restaurant CSV and ask about **address, rating, positives, negatives**")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("📁 Upload Feedback CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    st.markdown("---")
    st.markdown("### Example Queries")
    st.code("""MTR address
CTR rating
Veena Stores positives
Brahmins negatives
best south indian""")

# -------------------- Helpers --------------------
def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower().strip())

def find_restaurant_name(question, df):
    q = normalize(question)
    names = df["Name"].dropna().tolist()
    normalized_names = [normalize(n) for n in names]

    for i, n in enumerate(normalized_names):
        if q == n or q in n or n in q:
            return names[i]

    match = difflib.get_close_matches(q, normalized_names, n=1, cutoff=0.6)
    if match:
        return names[normalized_names.index(match[0])]
    return None

def get_precise_answer(question, df):
    restaurant = find_restaurant_name(question, df)
    q = question.lower()

    if restaurant:
        row = df[df["Name"] == restaurant].iloc[0]

        if "address" in q or "location" in q:
            return f"📍 {restaurant} Address: {row['Address']}"
        if "rating" in q or "star" in q or "score" in q:
            return f"⭐ {restaurant} Rating: {row['Rating']}"
        if "positive" in q or "good" in q or "pros" in q or "best" in q:
            return f"✅ Positive about {restaurant}: {row['Positive Feedback']}"
        if "negative" in q or "bad" in q or "cons" in q or "complaint" in q:
            return f"⚠️ Negative about {restaurant}: {row['Negative Feedback']}"

        return f"❓ Please ask about **address, rating, positives, or negatives**."

    if "south indian" in q:
        top = df[df["Cuisine"].str.contains("South Indian", case=False, na=False)]
        if not top.empty:
            best = top.sort_values("Rating", ascending=False).iloc[0]
            return f"🍛 Best South Indian: **{best['Name']}** ({best['Rating']}⭐)"

    return "❌ No specific information found"


# -------------------- LLM Setup --------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"   # ✅ you can replace with your downloaded model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

# -------------------- Main App --------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.session_state.df = df

        with st.expander("📊 Preview Data"):
            st.dataframe(df.head(10))
            st.write(f"Total entries: {len(df)}")

        st.subheader("💬 Ask a Question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about a restaurant..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Finding answer..."):
                    # ✅ Use rule-based filter first
                    answer = get_precise_answer(prompt, df)

                    # If no direct match, fallback to embeddings + LLM
                    if "❌" in answer:
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        docs = []
                        for _, row in df.iterrows():
                            docs.append(f"Restaurant: {row['Name']}, Address: {row['Address']}, Rating: {row['Rating']}, Cuisine: {row['Cuisine']}, Positives: {row['Positive Feedback']}, Negatives: {row['Negative Feedback']}")

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                        splits = text_splitter.create_documents(docs)
                        vectorstore = FAISS.from_documents(splits, embeddings)

                        retriever = vectorstore.as_retriever()
                        llm = load_llm()

                        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                        answer = qa.run(prompt)

                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("CSV must have columns: Name, Address, Rating, Cuisine, Positive Feedback, Negative Feedback")
else:
    st.info("👈 Upload a CSV file to start.")

# -------------------- Clear Chat --------------------
if st.session_state.get("messages"):
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

