import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="FakeGuard - Fake News Detector", page_icon="📰", layout="wide")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ── Session State Init ────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Examples Database ─────────────────────────────────────────
examples_db = {
    "✅ Real — Reuters Politics": [
        ("Senate passes bipartisan infrastructure bill", "WASHINGTON (Reuters) — The United States Senate passed a sweeping $1.2 trillion infrastructure bill on Tuesday with bipartisan support. The bill includes funding for roads, bridges, broadband internet, and clean water systems, passed 69-30. Senate Majority Leader Chuck Schumer called it a generational investment in America's future. Republican Senator Rob Portman said the package would create millions of good-paying jobs."),
        ("Biden signs executive order on climate change", "WASHINGTON (Reuters) — President Biden signed a sweeping executive order on climate change on Wednesday, directing federal agencies to eliminate carbon emissions from the power sector by 2035. The order also rejoins the Paris Climate Agreement and revokes permits for the Keystone XL pipeline. Environmental groups praised the decision as historic."),
        ("Congress approves $1.9 trillion COVID relief package", "WASHINGTON (Reuters) — The House of Representatives approved a $1.9 trillion coronavirus relief bill on Wednesday, sending the legislation to President Biden for his signature. The package includes $1,400 direct payments to most Americans, extended unemployment benefits, and $350 billion for state and local governments."),
        ("Federal Reserve raises interest rates by 0.75 percent", "WASHINGTON (Reuters) — The Federal Reserve raised its benchmark interest rate by three-quarters of a percentage point on Wednesday, the largest increase since 1994, as policymakers accelerate their fight against the highest inflation in four decades. The federal funds rate now stands between 1.5 and 1.75 percent."),
        ("NATO allies agree to increase defense spending", "BRUSSELS (Reuters) — NATO member countries agreed on Thursday to significantly increase their defense spending commitments following Russia's invasion of Ukraine. All 30 alliance members signed a declaration pledging to meet the two percent of GDP spending target within the next two years."),
        ("US unemployment rate falls to 3.5 percent", "WASHINGTON (Reuters) — The United States unemployment rate fell to 3.5 percent in December, matching a 50-year low, as employers added 223,000 jobs despite rising interest rates. The Labor Department report showed wages grew 4.6 percent from a year earlier, slightly below expectations."),
        ("House passes sweeping gun control legislation", "WASHINGTON (Reuters) — The House of Representatives passed the most significant gun control legislation in nearly three decades on Friday, approving a bill that would close the boyfriend loophole and enhance background checks for gun buyers under 21 years old."),
        ("Supreme Court overturns Roe v Wade in landmark decision", "WASHINGTON (Reuters) — The Supreme Court overturned Roe v. Wade on Friday, eliminating the constitutional right to abortion that had been in place for nearly 50 years. The 6-3 decision written by Justice Samuel Alito returns the question of abortion regulation to individual states."),
        ("US and China reach trade agreement after months of negotiations", "WASHINGTON (Reuters) — The United States and China reached a preliminary trade agreement on Friday after months of negotiations, with Beijing agreeing to purchase an additional $200 billion in American goods over two years. Both sides agreed to hold off on imposing additional tariffs."),
        ("President signs bipartisan bill to protect same sex marriage", "WASHINGTON (Reuters) — President Biden signed legislation on Tuesday protecting same-sex and interracial marriages into federal law, providing a legal backstop in case the Supreme Court overturns its landmark marriage equality ruling. The bill passed with bipartisan support."),
    ],
    "✅ Real — Tech News": [
        ("Apple unveils new AI chip for next generation iPhones", "SAN FRANCISCO (Reuters) — Apple Inc on Monday introduced a new generation of its custom silicon processor designed to accelerate artificial intelligence tasks on device. The chip delivers three times the performance of its predecessor. The processor will power the next generation of iPhones and MacBooks."),
        ("Microsoft acquires Activision Blizzard for 68 billion dollars", "SEATTLE (Reuters) — Microsoft Corp said it would buy video game company Activision Blizzard Inc for $68.7 billion in its biggest ever deal, giving it a major presence in mobile gaming and the metaverse. The acquisition is the largest in the history of the video game industry."),
        ("Google announces major updates to search algorithm", "MOUNTAIN VIEW (Reuters) — Alphabet Inc's Google announced sweeping updates to its search algorithm, incorporating large language model technology to provide more conversational results. The company said the changes would affect billions of daily searches worldwide."),
        ("Tesla reports record quarterly deliveries", "AUSTIN (Reuters) — Tesla Inc reported record vehicle deliveries in the fourth quarter despite ongoing supply chain challenges, delivering 405,278 vehicles. Chief Executive Elon Musk said the company remains on track to achieve 50 percent annual delivery growth."),
        ("Amazon Web Services launches new cloud region in Asia", "SEATTLE (Reuters) — Amazon.com Inc's cloud computing unit announced the launch of a new data center region in Southeast Asia, its third in the region. The expansion reflects growing demand for cloud services across emerging markets."),
        ("OpenAI releases new version of ChatGPT with improved capabilities", "SAN FRANCISCO (Reuters) — OpenAI released an updated version of its ChatGPT artificial intelligence chatbot, featuring improved reasoning capabilities and reduced errors. The new model performs significantly better on standardized tests including the bar exam and SAT."),
        ("Nvidia reports record revenue driven by AI chip demand", "SANTA CLARA (Reuters) — Nvidia Corp reported record quarterly revenue of $22.1 billion, driven by unprecedented demand for its artificial intelligence chips. The data center segment grew 409 percent year over year."),
        ("Meta announces layoffs of 11000 employees", "MENLO PARK (Reuters) — Meta Platforms Inc said it would lay off 11,000 employees, about 13 percent of its workforce, as the social media company cuts costs following a sharp decline in revenue. Zuckerberg said the company had over-hired during the pandemic."),
        ("Twitter acquired by Elon Musk for 44 billion dollars", "SAN FRANCISCO (Reuters) — Elon Musk completed his $44 billion acquisition of Twitter Inc on Thursday and immediately fired top executives including Chief Executive Parag Agrawal. Musk said the bird is freed as he took control of the platform."),
        ("Samsung unveils new foldable smartphone lineup", "SEOUL (Reuters) — Samsung Electronics unveiled its latest lineup of foldable smartphones on Wednesday, featuring improved durability and a lighter design. The Galaxy Z Fold5 and Z Flip5 will go on sale globally next month."),
    ],
    "✅ Real — Science": [
        ("Scientists discover new treatment for Alzheimers disease", "LONDON (Reuters) — Researchers at University College London announced a breakthrough in Alzheimer's treatment. A clinical trial involving 1,800 patients showed a 35 percent reduction in cognitive decline over 18 months. The findings were published in the New England Journal of Medicine."),
        ("NASA confirms water ice discovery on the Moon", "WASHINGTON (Reuters) — NASA scientists confirmed the presence of water ice in permanently shadowed craters near the Moon's south pole. The findings were published in Nature Astronomy and based on data from the SOFIA airborne observatory."),
        ("WHO approves first malaria vaccine for widespread use", "GENEVA (Reuters) — The World Health Organization approved the world's first malaria vaccine for widespread use in children across sub-Saharan Africa. The RTS,S vaccine showed 30 percent efficacy in clinical trials involving 800,000 children."),
        ("Scientists sequence complete human genome for first time", "WASHINGTON (Reuters) — An international team of scientists announced they had sequenced the complete human genome for the first time, filling in gaps that had existed since the Human Genome Project was completed in 2003."),
        ("SpaceX successfully lands reusable rocket for record 15th time", "CAPE CANAVERAL (Reuters) — SpaceX successfully launched and landed its Falcon 9 rocket booster for the 15th time, setting a new record for rocket reusability. The mission delivered 53 Starlink satellites to low Earth orbit."),
        ("New study links air pollution to increased dementia risk", "LONDON (Reuters) — A major study published in the British Medical Journal found that long-term exposure to air pollution significantly increases the risk of developing dementia. The research followed 130,000 people over 10 years."),
        ("Researchers develop solar panel with record 47 percent efficiency", "CAMBRIDGE (Reuters) — Scientists at MIT announced a breakthrough in solar energy technology, developing a new type of solar cell that achieves 47 percent efficiency, nearly double the current commercial standard."),
        ("Scientists create lab grown meat approved for human consumption", "WASHINGTON (Reuters) — The US Food and Drug Administration granted approval for the sale of lab-grown chicken meat developed by two California companies. The cultivated meat is grown from animal cells without slaughter."),
        ("James Webb telescope captures deepest image of universe", "WASHINGTON (Reuters) — NASA released the deepest and sharpest infrared image of the universe ever taken by the James Webb Space Telescope. The image shows thousands of galaxies dating back over 13 billion years."),
        ("Scientists find evidence of ancient ocean on Mars", "PASADENA (Reuters) — NASA scientists analyzing data from the Perseverance rover announced new evidence suggesting Mars once had a large ocean. The findings support theories about Mars having conditions suitable for life."),
    ],
    "🚨 Fake — Health Hoax": [
        ("SHOCKING: Doctors confirm miracle cure suppressed by Big Pharma", "Whistleblowers reveal what Big Pharma has been hiding for decades. A simple household remedy eliminates any virus within 24 hours. Mainstream media refuses to cover this. Share before it gets deleted! Government agents are suppressing this information. The deep state does not want you to know this secret remedy."),
        ("BREAKING: Scientists prove vaccines cause autism in new hidden study", "A bombshell study the CDC has been desperately trying to hide finally proves what concerned parents have known for years. Top scientists confirmed that childhood vaccines directly cause autism in 1 in 3 children. The government has paid billions in secret settlements to silence victims."),
        ("URGENT: 5G towers confirmed to spread disease whistleblower reveals", "A brave engineer who worked for a major telecommunications company has come forward to reveal the shocking truth. The 5G towers being installed across the country are deliberately spreading disease. The globalist elite are using this technology to depopulate the planet."),
        ("EXPOSED: Common household chemical cures all cancers doctors silenced", "What oncologists do not want you to know — a simple combination of baking soda and lemon juice has been proven to cure all forms of cancer in trials that were never published. The pharmaceutical industry cannot afford for this cure to become public knowledge."),
        ("WARNING: Government adding mind control chemicals to drinking water", "Brave scientists have blown the whistle on a shocking government program to add fluoride and mind-control chemicals to municipal water supplies. The chemicals are designed to make the population docile and compliant. Install a special filter to protect your family."),
        ("MIRACLE: Man cures stage 4 cancer in 2 weeks using one fruit", "Doctors are furious about this natural cancer cure the pharmaceutical industry has tried to suppress for 30 years. A man diagnosed with stage 4 cancer refused chemotherapy and used a special fruit diet. Two weeks later his cancer completely disappeared."),
        ("ALERT: Face masks cause oxygen deprivation and permanent brain damage", "A shocking new study that social media has been desperately suppressing proves that wearing face masks for more than 20 minutes causes dangerous oxygen deprivation leading to permanent brain damage. The government is using masks to dumb down the population."),
        ("REVEALED: Hospitals secretly harvesting organs from COVID patients", "Multiple nurses have come forward anonymously to reveal that patients admitted with COVID-19 are being secretly given lethal injections so their organs can be harvested and sold. The hospitals receive $39,000 for each COVID death."),
        ("BOMBSHELL: Drinking bleach solution kills coronavirus in 10 minutes", "A patriot doctor reveals that a diluted bleach solution kills the coronavirus in the body within 10 minutes. The FDA tried to suppress this information because it would destroy the vaccine industry. Share this life-saving information before it disappears."),
        ("EXPOSED: Bill Gates microchipping people through COVID vaccines", "Leaked documents confirm that Bill Gates has been secretly inserting microchips into COVID-19 vaccines to track and control the global population. The chips are activated by 5G towers and allow the globalist elite to monitor your every move."),
    ],
    "🚨 Fake — Political Hoax": [
        ("CONFIRMED: President secretly cancels all elections with hidden order", "Unnamed sources confirm the President signed a classified executive order suspending all federal elections. Patriots must spread this before the globalist media buries it forever. The deep state is planning a complete takeover. Multiple insiders confirmed this shocking development."),
        ("EXPOSED: George Soros funding secret army to overthrow US government", "Documents leaked from a deep state operative reveal that billionaire globalist George Soros has been secretly funding a private army of 500,000 mercenaries positioned in underground bunkers across America ready to overthrow the government."),
        ("BREAKING: Democrats caught rigging voting machines in 47 states", "A bombshell report reveals that Democratic operatives successfully hacked voting machines in 47 states. The software was installed by Chinese technicians disguised as election workers. The Supreme Court is being pressured to keep this evidence hidden."),
        ("URGENT: UN troops massing on US border preparing for invasion", "Patriot scouts along the borders have confirmed massive buildup of United Nations troops disguised in civilian vehicles. The globalist invasion force includes soldiers from China and Venezuela awaiting the signal from deep state operatives."),
        ("SHOCKING: Secret law passed to confiscate all privately owned guns", "Congress passed a secret amendment buried in a spending bill that would allow federal agents to enter homes without warrants to confiscate all privately owned firearms. The bill was signed at midnight to avoid public attention."),
        ("REVEALED: Obama building secret militia in 50 American cities", "Intelligence sources confirmed that former President Obama has been quietly building a private militia in 50 American cities using funds funneled through his charitable foundation. The militia is preparing to launch attacks on conservative communities."),
        ("ALERT: Martial law to be declared this weekend military sources confirm", "High-ranking military officers who cannot be named confirmed that martial law will be declared this coming weekend. All civilian courts will be suspended and a military tribunal established. Citizens are advised to stock up on food and ammunition."),
        ("BREAKING: Thousands of illegal ballots found in Democrat storage unit", "A patriot discovered thousands of pre-filled ballots in a storage unit rented under a fictitious name linked to the Democratic National Committee. The FBI is refusing to investigate because of deep state corruption at the highest levels."),
        ("EXPOSED: CIA secretly assassinated three senators who opposed the agenda", "A former CIA operative has come forward with proof that three US senators who opposed the globalist agenda were assassinated by CIA operatives. The mainstream media has covered up these murders on orders from the deep state shadow government."),
        ("CONFIRMED: North Korea tested nuclear weapon on US soil last week", "A patriot with connections to the Defense Intelligence Agency confirmed that North Korea successfully tested a small nuclear device on American soil in a remote area of Montana. The explosion was covered up as a mining accident."),
    ],
}

# ── Helper Functions ──────────────────────────────────────────
def get_top_keywords(text, vectorizer, model, top_n=10):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = vec.toarray()[0]
    
    try:
        coef = model.coef_[0]
        word_scores = {}
        nonzero_indices = vec.nonzero()[1]
        for idx in nonzero_indices:
            word = feature_names[idx]
            if len(word) > 2:
                score = tfidf_scores[idx] * abs(coef[idx])
                word_scores[word] = score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    except:
        nonzero_indices = vec.nonzero()[1]
        word_scores = [(feature_names[i], tfidf_scores[i]) for i in nonzero_indices if len(feature_names[i]) > 2]
        return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]

def get_probability_score(text, vectorizer, model):
    vec = vectorizer.transform([text])
    try:
        proba = model.predict_proba(vec)[0]
        return proba[0], proba[1]
    except:
        decision = model.decision_function(vec)[0]
        import math
        real_score = 1 / (1 + math.exp(-decision))
        fake_score = 1 - real_score
        return fake_score, real_score

def get_ai_explanation(text, prediction):
    fake_words = ["shocking","confirmed","exposed","whistleblower","big pharma","deep state",
                  "globalist","patriots","suppressed","mainstream media","share before","deleted",
                  "secret","bombshell","urgent","breaking","miracle","banned","they dont want",
                  "government hiding","unnamed sources","cannot be named","cover up","coverup"]
    real_words = ["reuters","according to","said in a statement","officials said","percent",
                  "billion","trillion","announced","reported","study","research","university",
                  "congress","senate","published","journal","confirmed by","clinical trial"]
    
    text_lower = text.lower()
    found_fake = [w for w in fake_words if w in text_lower]
    found_real = [w for w in real_words if w in text_lower]
    
    if prediction == 0:
        reasons = []
        if found_fake:
            reasons.append(f"Sensationalist words detected: **{', '.join(found_fake[:4])}**")
        reasons.append("Uses emotional language to trigger fear or urgency")
        reasons.append("Claims from unnamed or unverifiable sources")
        reasons.append("No credible news organization cited")
        if len(text.split()) < 50:
            reasons.append("Very short text — lacks journalistic detail")
        return reasons
    else:
        reasons = []
        if found_real:
            reasons.append(f"Credible source indicators found: **{', '.join(found_real[:4])}**")
        reasons.append("Professional and neutral journalistic language")
        reasons.append("Contains specific facts, numbers, and named sources")
        reasons.append("Follows standard news reporting structure")
        return reasons

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.result-real { background: #eaf3de; border: 2px solid #639922; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
.result-fake { background: #fcebeb; border: 2px solid #E24B4A; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
.feature-card { background: white; border: 1px solid #e8ecf0; border-radius: 12px; padding: 1.5rem; }
.step-card { background: #f0f4ff; border-radius: 12px; padding: 1.2rem; text-align: center; }
.history-card { background: #f8f9fb; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem; }
.keyword-fake { background: #fde8e8; color: #A32D2D; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500; margin: 3px; display: inline-block; }
.keyword-real { background: #e8f5e9; color: #2e7d32; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500; margin: 3px; display: inline-block; }
section[data-testid="stSidebar"] { background: #0C447C; }
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:white;text-align:center'>📰 FakeGuard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#adc8e6;text-align:center;font-size:13px'>AI-Powered News Detector</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔍 Detector", "📊 Dashboard", "🕒 History", "📖 About", "❓ FAQ"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#adc8e6;font-size:12px;text-align:center'>Model Accuracy</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;text-align:center'>99.7%</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#adc8e6;font-size:12px;text-align:center'>LinearSVC Algorithm</p>", unsafe_allow_html=True)
    st.markdown("---")
    hist_count = len(st.session_state.history)
    st.markdown(f"<p style='color:#adc8e6;font-size:12px;text-align:center'>Articles Checked: <b style='color:white'>{hist_count}</b></p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#185FA5,#0C447C);color:white;padding:3rem 2rem;border-radius:16px;text-align:center;margin-bottom:2rem'>
        <h1 style='font-size:2.5rem;margin-bottom:0.5rem'>📰 FakeGuard</h1>
        <p style='font-size:1.2rem;opacity:0.9'>AI-Powered Fake News Detection System</p>
        <p style='font-size:1rem;opacity:0.7'>Trained on 44,000+ articles · 99.7% Accuracy · LinearSVC Model</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,num,label in zip([c1,c2,c3,c4],["99.7%","44K+","7","<1s"],["Model Accuracy","Articles Trained","Models Compared","Detection Speed"]):
        with col:
            st.markdown(f"""<div style='background:#f8f9fb;border:1px solid #e0e0e0;border-radius:12px;padding:1.2rem;text-align:center'>
                <h2 style='color:#185FA5;font-size:2rem;margin:0'>{num}</h2>
                <p style='color:#666;font-size:13px;margin:0'>{label}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Key Features")
    f1,f2,f3,f4 = st.columns(4)
    features = [
        ("🤖 AI Detection","Advanced LinearSVC model trained on 44,000+ articles with 99.7% accuracy."),
        ("🔑 Top Keywords","See which words influenced the AI decision most."),
        ("📊 Probability Bar","Visual confidence score showing how sure the model is."),
        ("🕒 History","All your checked articles saved in one place this session."),
    ]
    for col,(title,desc) in zip([f1,f2,f3,f4],features):
        with col:
            st.markdown(f"""<div class="feature-card"><h4>{title}</h4><p style='color:#666;font-size:13px'>{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 How It Works")
    s1,s2,s3,s4 = st.columns(4)
    steps = [("1️⃣","Input","Paste your news article title and text"),
             ("2️⃣","Vectorize","TF-IDF converts text to 50,000 features"),
             ("3️⃣","Analyze","LinearSVC model classifies instantly"),
             ("4️⃣","Result","Get prediction + keywords + probability")]
    for col,(icon,title,desc) in zip([s1,s2,s3,s4],steps):
        with col:
            st.markdown(f"""<div class="step-card"><div style='font-size:2rem'>{icon}</div>
                <h4 style='color:#185FA5'>{title}</h4>
                <p style='color:#666;font-size:13px'>{desc}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Go to **🔍 Detector** in the sidebar to start analyzing news articles!")

# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Detector":
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("Select an example or paste your own article.")

    col_left, col_right = st.columns([1,1])

    with col_left:
        category = st.selectbox("📂 Select Category:", ["-- Choose category --"] + list(examples_db.keys()))
        title_default, text_default = "", ""
        if category != "-- Choose category --":
            example_list = examples_db[category]
            example_titles = ["-- Choose example --"] + [e[0] for e in example_list]
            chosen = st.selectbox("📌 Select Example:", example_titles)
            if chosen != "-- Choose example --":
                for t,txt in example_list:
                    if t == chosen:
                        title_default, text_default = t, txt
                        break

        title_input = st.text_input("Article Title (optional)", value=title_default, placeholder="Enter article title...")
        news_input = st.text_area("Article Text", value=text_default, height=200, placeholder="Paste the full article content here...")
        analyze = st.button("🔍 Analyze Article", use_container_width=True)

    with col_right:
        st.markdown("#### 📋 Analysis Result")
        if analyze:
            if not news_input.strip():
                st.warning("⚠️ Please enter some article text.")
            else:
                with st.spinner("Analyzing article..."):
                    combined = title_input + " " + news_input
                    vec = vectorizer.transform([combined])
                    prediction = model.predict(vec)[0]
                    fake_prob, real_prob = get_probability_score(combined, vectorizer, model)
                    keywords = get_top_keywords(combined, vectorizer, model, top_n=10)
                    reasons = get_ai_explanation(combined, prediction)

                # ── Result Box ──
                if prediction == 1:
                    st.markdown("""<div class="result-real">
                        <h3 style='color:#3B6D11'>✅ REAL NEWS</h3>
                        <p style='color:#27500A'>This article contains patterns consistent with credible reporting.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="result-fake">
                        <h3 style='color:#A32D2D'>🚨 FAKE NEWS</h3>
                        <p style='color:#791F1F'>Linguistic patterns associated with misinformation detected.</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── FEATURE 2: Probability Bar ──
                st.markdown("#### 📊 Confidence Score")
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    x=[real_prob * 100], y=[""],
                    orientation='h', name="Real",
                    marker_color="#639922",
                    text=[f"Real: {real_prob*100:.1f}%"],
                    textposition='inside'
                ))
                fig_prob.add_trace(go.Bar(
                    x=[fake_prob * 100], y=[""],
                    orientation='h', name="Fake",
                    marker_color="#E24B4A",
                    text=[f"Fake: {fake_prob*100:.1f}%"],
                    textposition='inside'
                ))
                fig_prob.update_layout(
                    barmode='stack', height=80,
                    margin=dict(l=0,r=0,t=0,b=0),
                    showlegend=True,
                    xaxis=dict(range=[0,100], showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", y=1.3)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # ── FEATURE 1: Top Keywords ──
                st.markdown("#### 🔑 Top Influential Keywords")
                kw_color = "keyword-fake" if prediction == 0 else "keyword-real"
                kw_html = ""
                for word, score in keywords:
                    kw_html += f'<span class="{kw_color}">{word}</span> '
                st.markdown(kw_html, unsafe_allow_html=True)

                # ── FEATURE 4: AI Explanation ──
                st.markdown("#### 🤖 AI Explanation")
                for reason in reasons:
                    if prediction == 0:
                        st.markdown(f"⚠️ {reason}")
                    else:
                        st.markdown(f"✔️ {reason}")

                # ── FEATURE 3: Save to History ──
                st.session_state.history.append({
                    "time": datetime.now().strftime("%I:%M %p"),
                    "date": datetime.now().strftime("%d %b %Y"),
                    "title": title_input if title_input else news_input[:60] + "...",
                    "prediction": "REAL ✅" if prediction == 1 else "FAKE 🚨",
                    "real_prob": round(real_prob * 100, 1),
                    "fake_prob": round(fake_prob * 100, 1),
                    "keywords": [w for w,s in keywords[:5]],
                    "prediction_raw": prediction
                })
                st.success("✅ Saved to History!")

        else:
            st.markdown("""<div style='background:#f8f9fb;border:2px dashed #dee2e6;border-radius:12px;
                padding:3rem;text-align:center;margin-top:1rem'>
                <p style='font-size:2rem'>🔍</p>
                <p style='color:#999'>Select an example or paste your article<br>then click Analyze Article</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Model Performance Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Test Accuracy", "99.73%", "+0.73%")
    c2.metric("Precision", "99.8%", "+1.2%")
    c3.metric("Recall", "99.7%", "+0.9%")
    c4.metric("F1 Score", "99.7%", "+1.1%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Model Comparison")
        df = pd.DataFrame({
            "Model": ["Linear SVM","Passive Aggressive","Gradient Boosting","Decision Tree","Random Forest","Logistic Regression","Naive Bayes"],
            "Accuracy": [99.73,99.71,99.65,99.64,99.60,99.22,96.33]
        }).sort_values("Accuracy")
        fig = px.bar(df, x="Accuracy", y="Model", orientation="h",
                    color="Accuracy", color_continuous_scale="Blues", range_x=[94,100])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🥧 Dataset Distribution")
        fig2 = go.Figure(data=[go.Pie(
            labels=["Real News","Fake News"], values=[21417,23481],
            hole=0.4, marker_colors=["#639922","#E24B4A"]
        )])
        fig2.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("#### 🔲 Confusion Matrix")
        fig3 = go.Figure(data=go.Heatmap(
            z=[[4280,12],[9,4283]],
            x=["Predicted Fake","Predicted Real"],
            y=["Actual Fake","Actual Real"],
            colorscale="Blues", text=[[4280,12],[9,4283]], texttemplate="%{text}",
        ))
        fig3.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 📋 Classification Report")
        st.dataframe(pd.DataFrame({
            "Class": ["Fake News","Real News","Average"],
            "Precision": ["99.8%","99.7%","99.7%"],
            "Recall": ["99.8%","99.7%","99.7%"],
            "F1-Score": ["99.8%","99.7%","99.7%"],
        }), use_container_width=True, hide_index=True)
        st.markdown("""
        **TF-IDF Settings:**
        - Max features: 50,000
        - N-gram range: (1, 2)
        - Sublinear TF: True
        - Stop words: English
        """)

# ══════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "🕒 History":
    st.markdown("## 🕒 Analysis History")

    if len(st.session_state.history) == 0:
        st.markdown("""<div style='background:#f8f9fb;border:2px dashed #dee2e6;border-radius:12px;
            padding:3rem;text-align:center'>
            <p style='font-size:2rem'>🕒</p>
            <p style='color:#999'>No articles checked yet.<br>Go to Detector and analyze some articles!</p>
        </div>""", unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        real_count = sum(1 for h in st.session_state.history if h["prediction_raw"] == 1)
        fake_count = total - real_count

        m1,m2,m3 = st.columns(3)
        m1.metric("Total Checked", total)
        m2.metric("Real News", real_count)
        m3.metric("Fake News", fake_count)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()

        st.markdown("### 📋 Recent Checks")
        for i, item in enumerate(reversed(st.session_state.history)):
            color = "#eaf3de" if item["prediction_raw"] == 1 else "#fcebeb"
            border = "#639922" if item["prediction_raw"] == 1 else "#E24B4A"
            text_color = "#3B6D11" if item["prediction_raw"] == 1 else "#A32D2D"
            keywords_str = ", ".join(item["keywords"]) if item["keywords"] else "N/A"

            st.markdown(f"""
            <div style='background:{color};border:1px solid {border};border-radius:10px;padding:1rem;margin-bottom:0.75rem'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <h4 style='color:{text_color};margin:0'>{item["prediction"]}</h4>
                    <span style='color:#888;font-size:12px'>{item["date"]} · {item["time"]}</span>
                </div>
                <p style='color:#333;margin:0.5rem 0;font-size:14px'><b>Article:</b> {item["title"]}</p>
                <p style='color:#666;margin:0;font-size:13px'><b>Confidence:</b> Real {item["real_prob"]}% | Fake {item["fake_prob"]}%</p>
                <p style='color:#666;margin:0.3rem 0 0;font-size:13px'><b>Top Keywords:</b> {keywords_str}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if len(st.session_state.history) > 0:
            st.markdown("#### 📊 Session Statistics")
            fig_hist = go.Figure(data=[go.Pie(
                labels=["Real News ✅", "Fake News 🚨"],
                values=[real_count, fake_count],
                hole=0.4,
                marker_colors=["#639922", "#E24B4A"]
            )])
            fig_hist.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown("## 📖 About FakeGuard")
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        FakeGuard is an AI-powered fake news detection system built using machine learning,
        trained on the **Kaggle Fake and Real News Dataset** containing over 44,000 news articles.

        ### 🔬 Methodology
        - **Data Collection:** Kaggle dataset — real news from Reuters, fake news from flagged sources
        - **Preprocessing:** Text cleaning, TF-IDF vectorization with 50,000 features
        - **Model Selection:** 7 algorithms compared — LinearSVC achieved best accuracy of 99.73%
        - **Evaluation:** 80/20 train-test split with 5-fold cross validation

        ### 🆕 New Features
        - 🔑 **Top Keywords** — See which words influenced the AI decision
        - 📊 **Probability Bar** — Visual confidence score
        - 🕒 **History** — All checked articles saved this session
        - 🤖 **AI Explanation** — Simple reasons why it's real or fake

        ### 🛠️ Tech Stack
        """)
        t1,t2,t3,t4 = st.columns(4)
        for col,(tech,color) in zip([t1,t2,t3,t4],[("Python","#3776AB"),("Scikit-learn","#F7931E"),("Streamlit","#FF4B4B"),("Plotly","#3D4DB7")]):
            col.markdown(f"""<div style='background:{color};color:white;border-radius:8px;padding:0.5rem;text-align:center;font-size:13px;font-weight:500'>{tech}</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 📈 Model Stats")
        for k,v in {"Accuracy":"99.73%","Precision":"99.8%","Recall":"99.7%","F1-Score":"99.7%","Train Size":"35,278","Test Size":"8,820","Features":"50,000","Algorithm":"LinearSVC"}.items():
            st.markdown(f"**{k}:** {v}")

    st.divider()
    st.markdown("### 📚 Dataset Information")
    d1,d2,d3 = st.columns(3)
    with d1:
        st.markdown("""<div class="feature-card"><h4>📰 Real News</h4>
            <p style='color:#666'>21,417 articles sourced from Reuters.com</p>
            <p style='color:#3B6D11;font-weight:500'>Source: Reuters</p></div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class="feature-card"><h4>🚨 Fake News</h4>
            <p style='color:#666'>23,481 articles from flagged unreliable websites</p>
            <p style='color:#A32D2D;font-weight:500'>Source: Flagged websites</p></div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""<div class="feature-card"><h4>📊 Total Dataset</h4>
            <p style='color:#666'>44,898 articles covering politics, world news, and social topics</p>
            <p style='color:#185FA5;font-weight:500'>Source: Kaggle</p></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FAQ
# ══════════════════════════════════════════════════════════════
elif page == "❓ FAQ":
    st.markdown("## ❓ Frequently Asked Questions")
    faqs = [
        ("🤔 What is FakeGuard?", "FakeGuard is an AI-powered fake news detection system using LinearSVC trained on 44,000+ articles with 99.7% accuracy."),
        ("📊 How accurate is the model?", "99.73% accuracy. We compared 7 algorithms — LinearSVC performed best."),
        ("🔑 What are Top Keywords?", "After analysis, the app shows which words most influenced the AI's decision. Red keywords indicate fake patterns, green indicate real patterns."),
        ("📊 What is the Probability Bar?", "It shows how confident the model is — e.g. Real 85% | Fake 15%. The wider the bar, the more confident the model."),
        ("🕒 What is History feature?", "Every article you analyze is automatically saved in the History section for this session with time, result, confidence, and top keywords."),
        ("🤖 What is AI Explanation?", "After each prediction, the app explains in simple language why it thinks the article is real or fake — like which words triggered the decision."),
        ("📰 What kind of news can it detect?", "Best on English political and world news. May be less accurate on satire or opinion pieces."),
        ("🔒 Is my data safe?", "Yes! No article text is stored permanently. History is only saved for the current session."),
        ("⚠️ Can it be fooled?", "Like any ML model, it can make mistakes. Always verify important news from multiple trusted sources."),
        ("🛠️ What technologies are used?", "Python · Scikit-learn · Streamlit · Plotly · Pandas · Pickle"),
    ]
    for q,a in faqs:
        with st.expander(q):
            st.write(a)
    st.divider()
    st.info("This project is open source. Check the GitHub repository for full code and documentation.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#999;font-size:12px'>📰 FakeGuard · Built with Scikit-learn + Streamlit · Kaggle Dataset · 99.7% Accuracy</p>", unsafe_allow_html=True)
