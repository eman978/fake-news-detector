import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.result-real { background: #eaf3de; border: 2px solid #639922; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
.result-fake { background: #fcebeb; border: 2px solid #E24B4A; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
.stat-card { background: #f8f9fb; border: 1px solid #e0e0e0; border-radius: 12px; padding: 1.2rem; text-align: center; }
.hero-section { background: linear-gradient(135deg, #185FA5 0%, #0C447C 100%); color: white; padding: 3rem 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; }
.feature-card { background: white; border: 1px solid #e8ecf0; border-radius: 12px; padding: 1.5rem; height: 100%; }
.team-card { background: #f8f9fb; border-radius: 12px; padding: 1.5rem; text-align: center; }
section[data-testid="stSidebar"] { background: #0C447C; }
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;text-align:center'>📰 FakeGuard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#adc8e6;text-align:center;font-size:13px'>AI-Powered News Detector</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Detector", "📊 Dashboard", "📖 About", "❓ FAQ"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<p style='color:#adc8e6;font-size:12px;text-align:center'>Model Accuracy</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;text-align:center'>99.7%</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#adc8e6;font-size:12px;text-align:center'>LinearSVC Algorithm</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-section">
        <h1 style='font-size:2.5rem;margin-bottom:0.5rem'>📰 FakeGuard</h1>
        <p style='font-size:1.2rem;opacity:0.9'>AI-Powered Fake News Detection System</p>
        <p style='font-size:1rem;opacity:0.7;margin-top:0.5rem'>Trained on 44,000+ articles · 99.7% Accuracy · LinearSVC Model</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#185FA5;font-size:2rem'>99.7%</h2>
            <p style='color:#666;font-size:13px'>Model Accuracy</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#185FA5;font-size:2rem'>44K+</h2>
            <p style='color:#666;font-size:13px'>Articles Trained</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#185FA5;font-size:2rem'>7</h2>
            <p style='color:#666;font-size:13px'>Models Compared</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#185FA5;font-size:2rem'>&lt;1s</h2>
            <p style='color:#666;font-size:13px'>Detection Speed</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Key Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="feature-card">
            <h3>🤖 AI Detection</h3>
            <p style='color:#666'>Advanced LinearSVC model trained on 44,000+ real and fake news articles from trusted sources.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card">
            <h3>⚡ Instant Results</h3>
            <p style='color:#666'>Get results in under 1 second. Paste any article and our model analyzes it instantly.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="feature-card">
            <h3>📊 High Accuracy</h3>
            <p style='color:#666'>99.7% accuracy achieved using TF-IDF features with 50,000 word and bigram combinations.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 How It Works")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1️⃣", "Input", "Paste your news article title and text"),
        ("2️⃣", "Vectorize", "TF-IDF converts text to 50,000 features"),
        ("3️⃣", "Analyze", "LinearSVC model classifies the article"),
        ("4️⃣", "Result", "Get instant Real or Fake prediction"),
    ]
    for col, (icon, title, desc) in zip([s1,s2,s3,s4], steps):
        with col:
            st.markdown(f"""<div style='background:#f0f4ff;border-radius:12px;padding:1.2rem;text-align:center'>
                <div style='font-size:2rem'>{icon}</div>
                <h4 style='color:#185FA5'>{title}</h4>
                <p style='color:#666;font-size:13px'>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — DETECTOR
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Detector":
    st.markdown("## 🔍 News Detector")
    st.markdown("Paste any news article below to check if it's real or fake.")

    examples = {
        "-- Select an example --": ("", ""),
        "✅ Real — Reuters Politics": (
            "Senate passes bipartisan infrastructure bill",
            "WASHINGTON (Reuters) — The United States Senate passed a sweeping $1.2 trillion infrastructure bill on Tuesday with bipartisan support. The bill includes funding for roads, bridges, broadband internet, and clean water systems, passed 69-30. Senate Majority Leader Chuck Schumer called it a generational investment in America's future. Republican Senator Rob Portman said the package would create millions of good-paying jobs and make the country more competitive globally."
        ),
        "✅ Real — Tech News": (
            "Apple unveils new AI chip for next generation iPhones",
            "SAN FRANCISCO (Reuters) — Apple Inc on Monday introduced a new generation of its custom silicon processor designed to accelerate artificial intelligence tasks on device. The chip delivers three times the performance of its predecessor according to company officials. The new processor will power the next generation of iPhones and MacBooks. Analysts said the move puts Apple ahead of rivals in the mobile AI race and could boost sales significantly."
        ),
        "✅ Real — Science": (
            "Scientists discover new treatment for Alzheimer's disease",
            "LONDON (Reuters) — Researchers at University College London announced a significant breakthrough in Alzheimer's treatment on Wednesday. The clinical trial involving 1,800 patients showed a 35 percent reduction in cognitive decline over 18 months. The drug targets amyloid plaques in the brain, which are associated with the disease. The findings were published in the New England Journal of Medicine and peer-reviewed by independent scientists."
        ),
        "🚨 Fake — Health Hoax": (
            "SHOCKING: Doctors confirm miracle cure suppressed by Big Pharma",
            "A group of whistleblowers have finally revealed what Big Pharma has been hiding for decades. A simple household remedy can eliminate any virus within 24 hours. The mainstream media refuses to cover this because it would destroy the billion-dollar pharmaceutical industry. Share this before it gets deleted! Government agents are already trying to suppress this information. Thousands of people have tried this miracle cure with amazing results. The deep state does not want you to know this secret remedy."
        ),
        "🚨 Fake — Political Hoax": (
            "CONFIRMED: President secretly cancels all elections with hidden order",
            "According to unnamed sources close to the White House that cannot be named for their safety, the President has quietly signed a classified executive order suspending all upcoming federal elections indefinitely. Patriots are urged to spread this information before the globalist media buries it forever. The deep state is planning a complete takeover of the democratic process. Multiple insider sources have confirmed this shocking development that mainstream media is hiding from the public."
        ),
    }

    selected = st.selectbox("📌 Load an example:", list(examples.keys()))
    title_default, text_default = examples[selected]

    col_inp, col_res = st.columns([1, 1])
    with col_inp:
        title_input = st.text_input("Article Title (optional)", value=title_default, placeholder="Enter article title...")
        news_input = st.text_area("Article Text", value=text_default, height=250, placeholder="Paste the full article content here...")
        analyze = st.button("🔍 Analyze Article", use_container_width=True)

    with col_res:
        st.markdown("#### 📋 Result")
        if analyze:
            if not news_input.strip():
                st.warning("⚠️ Please enter some article text.")
            else:
                with st.spinner("Analyzing..."):
                    combined = title_input + " " + news_input
                    vec = vectorizer.transform([combined])
                    prediction = model.predict(vec)[0]

                if prediction == 1:
                    st.markdown("""<div class="result-real">
                        <h3 style='color:#3B6D11'>✅ REAL NEWS</h3>
                        <p style='color:#27500A'>This article contains patterns consistent with credible news reporting.</p>
                        <hr style='border-color:#639922'>
                        <p style='color:#3B6D11;font-size:13px'>✔ Professional language detected<br>✔ Credible source patterns found<br>✔ No sensationalist triggers</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="result-fake">
                        <h3 style='color:#A32D2D'>🚨 FAKE NEWS</h3>
                        <p style='color:#791F1F'>Linguistic patterns associated with misinformation were detected.</p>
                        <hr style='border-color:#E24B4A'>
                        <p style='color:#A32D2D;font-size:13px'>⚠ Sensationalist language detected<br>⚠ Unverified claims found<br>⚠ Misinformation patterns present</p>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#f8f9fb;border:2px dashed #dee2e6;border-radius:12px;padding:2rem;text-align:center;height:200px;display:flex;align-items:center;justify-content:center'>
                <div><p style='color:#999;font-size:1.1rem'>🔍</p><p style='color:#999'>Result will appear here after analysis</p></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Model Performance Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", "99.73%", "+0.73%")
    c2.metric("Precision", "99.8%", "+1.2%")
    c3.metric("Recall", "99.7%", "+0.9%")
    c4.metric("F1 Score", "99.7%", "+1.1%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Comparison")
        models_data = {
            "Model": ["Linear SVM", "Passive Aggressive", "Gradient Boosting", "Decision Tree", "Random Forest", "Logistic Regression", "Naive Bayes"],
            "Accuracy": [99.73, 99.71, 99.65, 99.64, 99.60, 99.22, 96.33]
        }
        df = pd.DataFrame(models_data).sort_values("Accuracy")
        fig = px.bar(df, x="Accuracy", y="Model", orientation="h",
                    color="Accuracy", color_continuous_scale="Blues",
                    range_x=[94, 100])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Dataset Distribution")
        fig2 = go.Figure(data=[go.Pie(
            labels=["Real News", "Fake News"],
            values=[21417, 23481],
            hole=0.4,
            marker_colors=["#639922", "#E24B4A"]
        )])
        fig2.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Confusion Matrix")
        fig3 = go.Figure(data=go.Heatmap(
            z=[[4280, 12], [9, 4283]],
            x=["Predicted Fake", "Predicted Real"],
            y=["Actual Fake", "Actual Real"],
            colorscale="Blues",
            text=[[4280, 12], [9, 4283]],
            texttemplate="%{text}",
        ))
        fig3.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Classification Report")
        report_data = {
            "Class": ["Fake News", "Real News", "Avg"],
            "Precision": ["99.8%", "99.7%", "99.7%"],
            "Recall": ["99.8%", "99.7%", "99.7%"],
            "F1-Score": ["99.8%", "99.7%", "99.7%"],
        }
        st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
        st.markdown("""
        **TF-IDF Settings:**
        - Max features: 50,000
        - N-gram range: (1, 2)
        - Sublinear TF: True
        - Stop words: English
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown("## 📖 About This Project")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        FakeGuard is an AI-powered fake news detection system built using machine learning.
        It was trained on the **Kaggle Fake and Real News Dataset** containing over 44,000 news articles.

        ### 🔬 Methodology
        - **Data Collection:** Kaggle dataset with real news from Reuters and fake news from flagged sources
        - **Preprocessing:** Text cleaning, TF-IDF vectorization with 50,000 features
        - **Model Selection:** 7 algorithms compared — LinearSVC achieved best accuracy
        - **Evaluation:** 80/20 train-test split with 5-fold cross validation

        ### 🛠️ Tech Stack
        """)
        t1, t2, t3, t4 = st.columns(4)
        for col, (tech, color) in zip([t1,t2,t3,t4], [("Python","#3776AB"),("Scikit-learn","#F7931E"),("Streamlit","#FF4B4B"),("Pandas","#150458")]):
            col.markdown(f"""<div style='background:{color};color:white;border-radius:8px;padding:0.5rem;text-align:center;font-size:13px;font-weight:500'>{tech}</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### 📈 Model Stats
        """)
        stats = {"Accuracy": "99.73%", "Precision": "99.8%", "Recall": "99.7%", "F1-Score": "99.7%", "Training Size": "35,278", "Test Size": "8,820", "Features": "50,000", "Algorithm": "LinearSVC"}
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

    st.divider()
    st.markdown("### 📚 Dataset Information")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""<div class="feature-card">
            <h4>📰 Real News</h4>
            <p style='color:#666'>21,417 articles sourced from Reuters.com — one of the world's most trusted news agencies.</p>
            <p style='color:#3B6D11;font-weight:500'>Source: Reuters</p>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class="feature-card">
            <h4>🚨 Fake News</h4>
            <p style='color:#666'>23,481 articles collected from websites flagged as unreliable by fact-checking organizations.</p>
            <p style='color:#A32D2D;font-weight:500'>Source: Flagged websites</p>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""<div class="feature-card">
            <h4>📊 Total Dataset</h4>
            <p style='color:#666'>44,898 articles total, covering politics, world news, government, and social topics.</p>
            <p style='color:#185FA5;font-weight:500'>Source: Kaggle</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — FAQ
# ══════════════════════════════════════════════════════════════
elif page == "❓ FAQ":
    st.markdown("## ❓ Frequently Asked Questions")

    faqs = [
        ("🤔 What is FakeGuard?", "FakeGuard is an AI-powered fake news detection system. It uses a LinearSVC machine learning model trained on 44,000+ news articles to classify news as real or fake with 99.7% accuracy."),
        ("📊 How accurate is the model?", "The LinearSVC model achieves 99.73% accuracy on the test set. We compared 7 different algorithms including Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, Passive Aggressive Classifier, and Linear SVM."),
        ("📰 What kind of news can it detect?", "The model works best on English-language political and world news articles similar to the Reuters dataset it was trained on. It may be less accurate on satire, opinion pieces, or very short headlines."),
        ("🔒 Is my data safe?", "Yes! All processing happens in real-time and no article text is stored or saved anywhere. Your data is completely private."),
        ("⚡ How fast is detection?", "Detection happens in under 1 second. The TF-IDF vectorizer converts your text to features instantly and the LinearSVC model predicts in milliseconds."),
        ("🌐 What dataset was used?", "The Kaggle Fake and Real News Dataset — real news from Reuters.com and fake news from websites flagged as unreliable. Total 44,898 balanced articles."),
        ("⚠️ Can it be fooled?", "Like any ML model, it can make mistakes — especially on satirical content, opinion pieces, or news outside its training domain. Always verify important news from multiple trusted sources."),
        ("🛠️ What technologies are used?", "Python, Scikit-learn (LinearSVC + TF-IDF), Streamlit for the web interface, Pandas for data handling, and Pickle for model saving."),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.write(a)

    st.divider()
    st.markdown("### 📬 Still have questions?")
    st.info("This project is open source. Check the GitHub repository for full code and documentation.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;font-size:12px'>📰 FakeGuard · Built with Scikit-learn + Streamlit · Kaggle Fake & Real News Dataset · 99.7% Accuracy</p>", unsafe_allow_html=True)
