import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

examples = {
    "✅ Real — Reuters": {
        "title": "Senate passes bipartisan infrastructure bill",
        "text": "WASHINGTON (Reuters) — The United States Senate passed a sweeping $1.2 trillion infrastructure bill on Tuesday with bipartisan support. The bill includes funding for roads, bridges, broadband internet, and clean water systems, passed 69-30. Senate Majority Leader Chuck Schumer called it a generational investment in America's future. Republican Senator Rob Portman said the package would create millions of good-paying jobs."
    },
    "✅ Real — Tech": {
        "title": "Apple unveils new AI chip for iPhones",
        "text": "SAN FRANCISCO (Reuters) — Apple Inc on Monday introduced a new generation of its custom silicon processor designed to accelerate artificial intelligence tasks on device. The chip delivers three times the performance of its predecessor according to company officials. The processor will power the next generation of iPhones and MacBooks. Analysts said the move puts Apple ahead of rivals in the mobile AI race."
    },
    "🚨 Fake — Health": {
        "title": "SHOCKING: Doctors confirm miracle cure suppressed by Big Pharma",
        "text": "A group of whistleblowers have finally revealed what Big Pharma has been hiding for decades. A simple household remedy can eliminate any virus within 24 hours. The mainstream media refuses to cover this because it would destroy the billion-dollar pharmaceutical industry. Share this before it gets deleted! Government agents are already trying to suppress this information. Thousands of people have tried this miracle cure with amazing results. The deep state does not want you to know this secret remedy."
    },
    "🚨 Fake — Political": {
        "title": "CONFIRMED: President secretly cancels all elections",
        "text": "According to unnamed sources close to the White House that cannot be named for their safety, the President has quietly signed a classified executive order suspending all upcoming federal elections indefinitely. Patriots are urged to spread this information before the globalist media buries it forever. The deep state is planning a complete takeover of the democratic process. Multiple insider sources have confirmed this shocking development that mainstream media is hiding from the public."
    },
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.result-real { background: #eaf3de; border: 2px solid #639922; border-radius: 12px; padding: 1.2rem; margin-top: 1rem; }
.result-fake { background: #fcebeb; border: 2px solid #E24B4A; border-radius: 12px; padding: 1.2rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666'>Paste any news article and our ML model will analyze it instantly.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "99.7%")
col2.metric("Trained On", "44,000+")
col3.metric("Algorithm", "LinearSVC")

st.divider()
st.markdown("### Try an example or paste your own news:")

selected_example = st.selectbox(
    "📌 Load an example:",
    ["-- Select an example --"] + list(examples.keys())
)

if selected_example != "-- Select an example --":
    title_default = examples[selected_example]["title"]
    text_default = examples[selected_example]["text"]
else:
    title_default = ""
    text_default = ""

title_input = st.text_input("Article title (optional)", value=title_default, placeholder="e.g. Breaking: Scientists discover...")
news_input = st.text_area("Article text", value=text_default, height=200, placeholder="Paste the full article content here...")

if st.button("🔍 Analyze article", use_container_width=True):
    if not news_input.strip():
        st.warning("⚠️ Please enter some article text.")
    else:
        with st.spinner("Analyzing..."):
            combined = title_input + " " + news_input
            vec = vectorizer.transform([combined])
            prediction = model.predict(vec)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-real">
                <h3 style="color:#3B6D11;">✅ This article appears to be REAL</h3>
                <p style="color:#27500A;">The article contains patterns consistent with credible news reporting.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-fake">
                <h3 style="color:#A32D2D;">🚨 This article appears to be FAKE</h3>
                <p style="color:#791F1F;">Linguistic patterns associated with misinformation were detected.</p>
            </div>""", unsafe_allow_html=True)

st.divider()
with st.expander("ℹ️ How does it work?"):
    st.write("Text is converted to TF-IDF features → LinearSVC model predicts Real or Fake → Result shown instantly.")
with st.expander("📊 How accurate is it?"):
    st.write("99.73% accuracy. Compared 7 algorithms — LinearSVC performed best.")
with st.expander("⚠️ Limitations"):
    st.write("Best on English political/world news. Always verify from trusted sources.")

st.caption("Built with Scikit-learn + Streamlit · Kaggle Fake & Real News Dataset · 99.7% Accuracy")
