import streamlit as st
import pickle
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #f8f9fb; }
  .hero { text-align: center; padding: 2rem 0 1rem; }
  .badge { display: inline-block; background: #e6f1fb; color: #185FA5; font-size: 12px; font-weight: 500; padding: 4px 14px; border-radius: 20px; margin-bottom: 1rem; }
  .stat-row { display: flex; gap: 12px; margin: 1.5rem 0; }
  .stat-card { flex: 1; background: #f1f3f5; border-radius: 10px; padding: 1rem; text-align: center; }
  .result-real { background: #eaf3de; border: 1px solid #639922; border-radius: 12px; padding: 1.2rem; }
  .result-fake { background: #fcebeb; border: 1px solid #E24B4A; border-radius: 12px; padding: 1.2rem; }
  .stButton > button { width: 100%; background: #185FA5 !important; color: white !important; font-size: 15px !important; font-weight: 500 !important; border: none !important; border-radius: 8px !important; padding: 0.75rem !important; }
  .stButton > button:hover { background: #0C447C !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><span class="badge">AI-powered detection</span><h1>📰 Fake News Detector</h1><p style="color:#666;">Paste any news article and our ML model will analyze it instantly.</p></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "99.7%")
col2.metric("Articles Trained", "44,000+")
col3.metric("Best Algorithm", "LinearSVC")

st.divider()

examples = {
    "Real — Reuters politics": ("Senate passes bipartisan infrastructure bill", "WASHINGTON (Reuters) — The United States Senate passed a sweeping $1.2 trillion infrastructure bill on Tuesday with bipartisan support. The bill includes funding for roads, bridges, broadband internet, and clean water systems, passed 69-30."),
    "Real — Tech news": ("Apple unveils new AI chip", "SAN FRANCISCO — Apple Inc on Monday introduced a new generation of its custom silicon processor designed to accelerate artificial intelligence tasks. The chip delivers three times the performance of its predecessor."),
    "Fake — Health claim": ("SHOCKING: Doctors confirm miracle cure", "A group of whistleblowers reveal what Big Pharma has been hiding. A simple household remedy can eliminate any virus. The mainstream media refuses to cover this. Share before it gets deleted! Government agents are suppressing this information."),
    "Fake — Political hoax": ("CONFIRMED: Secret executive order cancels elections", "According to unnamed sources close to the White House, the President secretly signed a classified order suspending all federal elections. Patriots must spread this before the globalist media buries it forever.")
}

st.markdown("**Try an example:**")
ex_cols = st.columns(4)
selected = None
for i, (label, _) in enumerate(examples.items()):
    if ex_cols[i].button(label, use_container_width=True):
        selected = label

title_val = ""
text_val = ""
if selected:
    title_val, text_val = examples[selected]

title_input = st.text_input("Article title (optional)", value=title_val, placeholder="e.g. Breaking: Scientists discover...")
news_input = st.text_area("Article text", value=text_val, height=180, placeholder="Paste the full article content here...")

if st.button("Analyze article"):
    if not news_input.strip():
        st.warning("Please enter some article text.")
    else:
        with st.spinner("Analyzing..."):
            combined = title_input + " " + news_input
            vec = vectorizer.transform([combined])
            prediction = model.predict(vec)[0]

        if prediction == 1:
            st.markdown('<div class="result-real"><h3 style="color:#3B6D11;">✅ This article appears to be REAL</h3><p style="color:#27500A;">The article contains patterns consistent with credible news reporting.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake"><h3 style="color:#A32D2D;">🚨 This article appears to be FAKE</h3><p style="color:#791F1F;">Linguistic patterns associated with misinformation were detected.</p></div>', unsafe_allow_html=True)

st.divider()
st.markdown("### How it works")
h1, h2, h3, h4 = st.columns(4)
h1.info("**Step 1**\nTitle + text combined")
h2.info("**Step 2**\nTF-IDF vectorization\n50,000 features")
h3.info("**Step 3**\nLinearSVC model\npredicts label")
h4.info("**Step 4**\nResult shown\ninstantly")

st.divider()
with st.expander("What dataset was used?"):
    st.write("Trained on Kaggle's Fake and Real News Dataset with 44,000+ articles. Real news from Reuters, fake news from flagged unreliable websites.")
with st.expander("How accurate is it?"):
    st.write("LinearSVC achieves 99.73% accuracy. We compared 7 algorithms — Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, Passive Aggressive, and Linear SVM.")
with st.expander("Can it detect all fake news?"):
    st.write("Best on English political/world news. May be less accurate on satire, opinion, or very short headlines. Always verify from trusted sources.")
with st.expander("What technologies power this?"):
    st.write("Python · Scikit-learn (LinearSVC + TF-IDF) · Streamlit · Pickle for model saving.")

st.caption("Built with Scikit-learn + Streamlit · Dataset: Kaggle Fake & Real News · Accuracy: 99.7%")
