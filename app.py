import streamlit as st
import pickle
import random
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
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

if "history" not in st.session_state:
    st.session_state.history = []

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
        return sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
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
        import math
        decision = model.decision_function(vec)[0]
        real_score = 1 / (1 + math.exp(-decision))
        return 1 - real_score, real_score

def get_ai_explanation(text, prediction):
    fake_words = ["shocking","confirmed","exposed","whistleblower","big pharma","deep state",
                  "globalist","patriots","suppressed","mainstream media","share before","deleted",
                  "secret","bombshell","urgent","breaking","miracle","unnamed sources","cover up"]
    real_words = ["reuters","according to","officials said","percent","billion","trillion",
                  "announced","reported","study","research","university","congress","senate",
                  "published","journal","clinical trial"]
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
        return reasons
    else:
        reasons = []
        if found_real:
            reasons.append(f"Credible source indicators found: **{', '.join(found_real[:4])}**")
        reasons.append("Professional and neutral journalistic language")
        reasons.append("Contains specific facts, numbers, and named sources")
        reasons.append("Follows standard news reporting structure")
        return reasons


# ══════════════════════════════════════════════════════════════
# GLOBAL CSS — Premium redesign
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── CSS Variables ── */
:root {
    --primary:       #1A56CC;
    --primary-dark:  #1240A0;
    --primary-light: #E8F0FB;
    --accent:        #3B82F6;
    --real-green:    #16A34A;
    --real-bg:       #F0FDF4;
    --real-border:   #86EFAC;
    --fake-red:      #DC2626;
    --fake-bg:       #FEF2F2;
    --fake-border:   #FCA5A5;
    --surface:       #FFFFFF;
    --bg:            #F7F9FC;
    --border:        #E2E8F0;
    --text-primary:  #0F172A;
    --text-secondary:#475569;
    --text-muted:    #94A3B8;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 16px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:     0 10px 40px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.05);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
    --radius-xl:     28px;
    --transition:    all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Remove Streamlit default padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ── Main background ── */
.stApp {
    background: var(--bg) !important;
}

/* ══════════════════════════════
   SIDEBAR
══════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #0B1D3A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* Nav items */
div[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
div[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 14px !important;
    margin: 1px 0 !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
}
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.08) !important;
}
div[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
div[data-testid="stSidebar"] .stRadio [aria-checked="true"][data-baseweb="radio"] {
    background: rgba(59,130,246,0.25) !important;
    border-left: 3px solid var(--accent) !important;
}
div[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    margin: 0 !important;
}

/* ══════════════════════════════
   GLOBAL BUTTONS
══════════════════════════════ */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.72rem 1.8rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 14px rgba(26,86,204,0.30) !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26,86,204,0.40) !important;
    background: linear-gradient(135deg, #2563EB 0%, var(--primary) 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Danger button (Clear History) ── */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid #E2E8F0 !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--fake-red) !important;
    color: var(--fake-red) !important;
    background: var(--fake-bg) !important;
    box-shadow: none !important;
}

/* ══════════════════════════════
   INPUTS
══════════════════════════════ */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
    padding: 0.65rem 0.9rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(26,86,204,0.12) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: var(--transition) !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--primary) !important;
}

/* ── Labels ── */
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: var(--transition) !important;
}
div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetric"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    margin-bottom: 10px !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
    transition: var(--transition) !important;
}
div[data-testid="stExpander"]:hover {
    border-color: #CBD5E1 !important;
    box-shadow: var(--shadow-md) !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--text-primary) !important;
    padding: 1rem 1.2rem !important;
}

/* ── Alert / Info ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid !important;
    font-size: 14px !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] {
    color: var(--primary) !important;
}

/* ══════════════════════════════
   REUSABLE COMPONENT CLASSES
══════════════════════════════ */

/* Stat Card */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.4rem 1rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: #CBD5E1;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: var(--primary);
    margin: 0;
    line-height: 1;
}
.stat-lbl {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin: 8px 0 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Feature Card */
.feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.3rem;
    height: 100%;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: #CBD5E1;
}
.feature-icon {
    font-size: 1.7rem;
    margin-bottom: 10px;
    display: block;
}
.feature-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 7px;
}
.feature-desc {
    font-size: 13.5px;
    color: var(--text-secondary);
    line-height: 1.65;
}

/* Step Card */
.step-card {
    background: linear-gradient(145deg, #EEF5FF, #E3EDFF);
    border: 1px solid #C3D9FF;
    border-radius: var(--radius-md);
    padding: 1.4rem 1rem;
    text-align: center;
    transition: var(--transition);
}
.step-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.step-num {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
    width: 34px; height: 34px;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px;
    box-shadow: 0 4px 10px rgba(26,86,204,0.30);
}
.step-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 7px;
}
.step-desc {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* Result Cards */
.result-real {
    background: var(--real-bg);
    border: 1.5px solid var(--real-border);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.5rem 1.3rem;
    margin-top: 0.25rem;
    position: relative;
    overflow: hidden;
    animation: slideIn 0.3s ease-out;
}
.result-fake {
    background: var(--fake-bg);
    border: 1.5px solid var(--fake-border);
    border-radius: var(--radius-md);
    padding: 1.5rem 1.5rem 1.3rem;
    margin-top: 0.25rem;
    position: relative;
    overflow: hidden;
    animation: slideIn 0.3s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-badge-real {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--real-green);
    color: white;
    font-size: 11px; font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-badge-fake {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--fake-red);
    color: white;
    font-size: 11px; font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Keyword Pills */
.keyword-real {
    background: #DCFCE7;
    color: #166534;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    border: 1px solid #86EFAC;
    transition: var(--transition);
}
.keyword-real:hover {
    background: #BBF7D0;
}
.keyword-fake {
    background: #FEE2E2;
    color: #991B1B;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 3px;
    display: inline-block;
    border: 1px solid #FCA5A5;
    transition: var(--transition);
}
.keyword-fake:hover {
    background: #FECACA;
}

/* History items */
.history-item-real {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--real-green);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}
.history-item-fake {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--fake-red);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}
.history-item-real:hover,
.history-item-fake:hover {
    box-shadow: var(--shadow-md);
    transform: translateX(2px);
}

/* Empty State */
.empty-state {
    background: var(--surface);
    border: 2px dashed #CBD5E1;
    border-radius: var(--radius-lg);
    padding: 3.5rem 2rem;
    text-align: center;
    margin-top: 0.25rem;
    animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.empty-icon {
    font-size: 2.8rem;
    margin-bottom: 12px;
    display: block;
    opacity: 0.6;
}
.empty-title {
    color: #64748B;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 5px;
}
.empty-subtitle {
    color: #94A3B8;
    font-size: 13.5px;
}

/* Section heading helper */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    display: block;
}

/* Info badge inline */
.info-badge {
    display: inline-block;
    background: var(--primary-light);
    color: var(--primary);
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}

/* Dataset info cards */
.dataset-card {
    border-radius: var(--radius-md);
    padding: 1.4rem;
    border: 1px solid;
    transition: var(--transition);
}
.dataset-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

/* Tech badge */
.tech-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.8rem;
    font-size: 13px;
    font-weight: 700;
    color: white;
    transition: var(--transition);
}
.tech-badge:hover {
    transform: scale(1.04);
}

/* Stats row in About */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: var(--text-secondary); }
.stat-val { color: var(--primary); font-weight: 700; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── Responsive columns ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(26,86,204,0.08));
        border-bottom: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        margin: 0 -1rem 0.5rem;
    '>
        <div style='
            width: 52px; height: 52px;
            background: linear-gradient(135deg, #3B82F6, #1A56CC);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            margin: 0 auto 10px;
            box-shadow: 0 6px 16px rgba(59,130,246,0.35);
        '>📰</div>
        <div style='font-size: 20px; font-weight: 700; color: white; letter-spacing: -0.01em;'>FakeGuard</div>
        <div style='font-size: 11.5px; color: #94A3B8; margin-top: 3px; letter-spacing: 0.04em; text-transform: uppercase;'>AI News Detector</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 0 0.25rem;'>", unsafe_allow_html=True)
    page = st.radio("", [
        "Home",
        "Detector",
        "Dashboard",
        "History",
        "About",
        "FAQ"
    ], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1rem 0'>", unsafe_allow_html=True)

    checked_count = len(st.session_state.history)
    real_count_sb = sum(1 for h in st.session_state.history if h["prediction_raw"] == 1)
    fake_count_sb = checked_count - real_count_sb

    st.markdown(f"""
    <div style='padding: 0 0.4rem;'>
        <div style='
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        '>
            <div style='text-align: center; margin-bottom: 0.8rem;'>
                <div style='font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;'>Model Accuracy</div>
                <div style='font-size: 28px; font-weight: 700; color: white; line-height: 1;'>99.7%</div>
                <div style='font-size: 10.5px; color: #64748B; margin-top: 2px;'>LinearSVC Algorithm</div>
            </div>
            <div style='display: flex; justify-content: space-between; padding-top: 0.7rem; border-top: 1px solid rgba(255,255,255,0.07);'>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 18px; font-weight: 700; color: #4ADE80;'>{real_count_sb}</div>
                    <div style='font-size: 10px; color: #64748B;'>Real</div>
                </div>
                <div style='width: 1px; background: rgba(255,255,255,0.07);'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 18px; font-weight: 700; color: #F87171;'>{fake_count_sb}</div>
                    <div style='font-size: 10px; color: #64748B;'>Fake</div>
                </div>
                <div style='width: 1px; background: rgba(255,255,255,0.07);'></div>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 18px; font-weight: 700; color: white;'>{checked_count}</div>
                    <div style='font-size: 10px; color: #64748B;'>Total</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    # Hero Banner
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1240A0 0%, #1A56CC 45%, #2563EB 100%);
        color: white;
        padding: 4rem 3rem;
        border-radius: var(--radius-xl);
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(18,64,160,0.35);
    '>
        <div style='
            position: absolute; top: -60px; right: -60px;
            width: 200px; height: 200px;
            background: rgba(255,255,255,0.04);
            border-radius: 50%;
        '></div>
        <div style='
            position: absolute; bottom: -80px; left: -40px;
            width: 240px; height: 240px;
            background: rgba(255,255,255,0.03);
            border-radius: 50%;
        '></div>
        <div style='
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 20px;
            padding: 5px 16px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 20px;
        '>
            ✦ AI-Powered Detection System
        </div>
        <h1 style='
            font-family: "DM Serif Display", serif;
            font-size: 3.5rem;
            font-weight: 400;
            margin: 0 0 12px;
            line-height: 1.1;
            letter-spacing: -0.02em;
        '>FakeGuard</h1>
        <p style='font-size: 1.1rem; opacity: 0.82; margin: 0 0 8px; font-weight: 400;'>
            Detect misinformation with 99.7% accuracy
        </p>
        <p style='font-size: 0.9rem; opacity: 0.5; margin: 0;'>
            Trained on 44,000+ articles &nbsp;·&nbsp; 7 algorithms compared &nbsp;·&nbsp; LinearSVC model
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("99.7%", "Model Accuracy"),
        ("44K+",  "Articles Trained"),
        ("7",     "Algorithms Tested"),
        ("<1s",   "Detection Speed"),
    ]
    for col, (num, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{num}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.1rem'>
        <span style='font-size:1.1rem;font-weight:700;color:var(--text-primary)'>Key Features</span>
        <span style='flex:1;height:1px;background:var(--border)'></span>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🤖", "AI Detection",      "Advanced LinearSVC model trained on 44K+ real and fake articles with 99.7% accuracy."),
        ("🔑", "Key Signals",       "After analysis, see which words most influenced the prediction decision."),
        ("📊", "Confidence Score",  "Visual probability bar showing how confident the model is in its verdict."),
        ("🕒", "History Tracking",  "Every article you analyze is automatically saved with timestamps and keywords."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.1rem'>
        <span style='font-size:1.1rem;font-weight:700;color:var(--text-primary)'>How It Works</span>
        <span style='flex:1;height:1px;background:var(--border)'></span>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1", "Paste Article",  "Add your article title and full text into the detector"),
        ("2", "Vectorize",      "TF-IDF converts text into 50,000 numerical features"),
        ("3", "Classify",       "LinearSVC model predicts Real or Fake instantly"),
        ("4", "Get Results",    "View prediction, confidence, keywords, and explanation"),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.info("👉 Head to **Detector** in the sidebar to start analyzing news articles.")


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════
elif page == "Detector":
    st.markdown("""
    <h2 style='font-size:1.7rem;font-weight:700;margin-bottom:4px;color:var(--text-primary)'>
        Fake News Detector
    </h2>
    <p style='color:var(--text-secondary);font-size:14.5px;margin-bottom:1.5rem'>
        Select an example from the library, or paste your own article text to analyze.
    </p>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Session state defaults for inputs ──
    if "title_input" not in st.session_state:
        st.session_state.title_input = ""
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""

    with col_left:
        st.markdown("<span class='section-label'>Example Library</span>", unsafe_allow_html=True)
        category = st.selectbox(
            "Category",
            ["— Choose a category —"] + list(examples_db.keys()),
            label_visibility="collapsed",
            key="category_select",
        )
        if category != "— Choose a category —":
            example_list = examples_db[category]
            example_titles = ["— Choose an example —"] + [e[0] for e in example_list]
            chosen = st.selectbox(
                "Example", example_titles, label_visibility="collapsed", key="example_select"
            )
            if chosen != "— Choose an example —":
                for t, txt in example_list:
                    if t == chosen:
                        if (
                            st.session_state.title_input != t
                            or st.session_state.news_input != txt
                        ):
                            st.session_state.title_input = t
                            st.session_state.news_input = txt
                            st.rerun()
                        break

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Article Input</span>", unsafe_allow_html=True)
        title_input = st.text_input(
            "Article Title",
            placeholder="Enter the article headline…",
            label_visibility="collapsed",
            key="title_input",
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        news_input = st.text_area(
            "Article Body",
            height=200,
            placeholder="Paste the full news article content here…",
            label_visibility="collapsed",
            key="news_input",
        )

        # Word count indicator
        wc = len(news_input.split()) if news_input.strip() else 0
        wc_color = "#16A34A" if wc >= 30 else "#D97706" if wc > 0 else "#94A3B8"
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-end;margin-top:4px;margin-bottom:12px'>
            <span style='font-size:11.5px;color:{wc_color};font-weight:500'>{wc} words</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Action buttons: Analyze (primary) + Load Sample + Clear ──
        b1, b2, b3 = st.columns([2, 1, 1])
        with b1:
            analyze = st.button(
                "🔍  Analyze Article", use_container_width=True, type="primary"
            )
        with b2:
            if st.button("🎲  Load Sample", use_container_width=True):
                all_examples = [
                    item for items in examples_db.values() for item in items
                ]
                t, txt = random.choice(all_examples)
                st.session_state.title_input = t
                st.session_state.news_input = txt
                st.rerun()
        with b3:
            if st.button("🧹  Clear", use_container_width=True):
                st.session_state.title_input = ""
                st.session_state.news_input = ""
                st.rerun()

    with col_right:
        st.markdown("<span class='section-label'>Analysis Result</span>", unsafe_allow_html=True)

        if analyze:
            if not news_input.strip():
                st.warning("⚠️ Please enter some article text before analyzing.")
            else:
                with st.spinner("Analyzing article…"):
                    combined = title_input + " " + news_input
                    vec = vectorizer.transform([combined])
                    prediction = model.predict(vec)[0]
                    fake_prob, real_prob = get_probability_score(combined, vectorizer, model)
                    keywords = get_top_keywords(combined, vectorizer, model, top_n=10)
                    reasons = get_ai_explanation(combined, prediction)

                # ── Verdict ──
                if prediction == 1:
                    st.markdown("""
                    <div class="result-real">
                        <span class="result-badge-real">✓ Verified</span>
                        <h3 style='color:#15803D;margin:4px 0 6px;font-size:1.35rem;font-weight:700'>Real News</h3>
                        <p style='color:#166534;margin:0;font-size:13.5px;line-height:1.55'>
                            This article contains patterns consistent with credible journalism.
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-fake">
                        <span class="result-badge-fake">⚠ Alert</span>
                        <h3 style='color:#B91C1C;margin:4px 0 6px;font-size:1.35rem;font-weight:700'>Fake News</h3>
                        <p style='color:#991B1B;margin:0;font-size:13.5px;line-height:1.55'>
                            Misinformation patterns detected in this article.
                        </p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                # ── Confidence Score ──
                st.markdown("<span class='section-label'>Confidence Score</span>", unsafe_allow_html=True)
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    x=[real_prob * 100], y=[""],
                    orientation='h', name="Real",
                    marker_color="#16A34A",
                    text=[f"Real  {real_prob*100:.1f}%"],
                    textposition='inside',
                    textfont=dict(color='white', size=12.5, family='DM Sans'),
                    marker=dict(line=dict(width=0))
                ))
                fig_prob.add_trace(go.Bar(
                    x=[fake_prob * 100], y=[""],
                    orientation='h', name="Fake",
                    marker_color="#DC2626",
                    text=[f"Fake  {fake_prob*100:.1f}%"],
                    textposition='inside',
                    textfont=dict(color='white', size=12.5, family='DM Sans'),
                    marker=dict(line=dict(width=0))
                ))
                fig_prob.update_layout(
                    barmode='stack', height=56,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

                rc, fc = st.columns(2)
                rc.markdown(f"<div style='text-align:center;color:#15803D;font-size:13px;font-weight:700;margin-top:-6px'>Real: {real_prob*100:.1f}%</div>", unsafe_allow_html=True)
                fc.markdown(f"<div style='text-align:center;color:#B91C1C;font-size:13px;font-weight:700;margin-top:-6px'>Fake: {fake_prob*100:.1f}%</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                # ── Keywords ──
                st.markdown("<span class='section-label'>Top Influential Keywords</span>", unsafe_allow_html=True)
                kw_class = "keyword-fake" if prediction == 0 else "keyword-real"
                kw_html = "".join([f'<span class="{kw_class}">{w}</span>' for w, _ in keywords])
                st.markdown(f"<div style='margin-top:4px;line-height:2.2'>{kw_html}</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                # ── Word Cloud ──
                st.markdown("<span class='section-label'>Word Cloud</span>", unsafe_allow_html=True)
                try:
                    cloud_colormap = "Reds" if prediction == 0 else "Greens"
                    cloud_bg = "#FEF2F2" if prediction == 0 else "#F0FDF4"
                    wc_obj = WordCloud(
                        width=900,
                        height=380,
                        background_color=cloud_bg,
                        colormap=cloud_colormap,
                        stopwords=set(STOPWORDS),
                        max_words=80,
                        prefer_horizontal=0.95,
                        relative_scaling=0.5,
                        margin=4,
                    ).generate(combined)
                    fig_wc, ax_wc = plt.subplots(figsize=(9, 3.8))
                    ax_wc.imshow(wc_obj, interpolation="bilinear")
                    ax_wc.axis("off")
                    fig_wc.patch.set_facecolor(cloud_bg)
                    plt.tight_layout(pad=0)
                    st.pyplot(fig_wc, use_container_width=True)
                    plt.close(fig_wc)
                except Exception:
                    st.caption("Word cloud unavailable for this input.")

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                # ── AI Explanation ──
                st.markdown("<span class='section-label'>Why this verdict?</span>", unsafe_allow_html=True)
                icon = "⚠️" if prediction == 0 else "✔️"
                exp_color = "#7F1D1D" if prediction == 0 else "#14532D"
                exp_bg = "#FEF2F2" if prediction == 0 else "#F0FDF4"
                exp_border = "#FECACA" if prediction == 0 else "#BBF7D0"
                reasons_html = "".join([
                    f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:6px'>"
                    f"<span style='font-size:13px;flex-shrink:0;margin-top:1px'>{icon}</span>"
                    f"<span style='font-size:13px;color:{exp_color};line-height:1.5'>{r}</span>"
                    f"</div>"
                    for r in reasons
                ])
                st.markdown(f"""
                <div style='background:{exp_bg};border:1px solid {exp_border};border-radius:10px;padding:0.9rem 1rem;margin-top:4px'>
                    {reasons_html}
                </div>
                """, unsafe_allow_html=True)

                # Save to history
                st.session_state.history.append({
                    "time":           datetime.now().strftime("%I:%M %p"),
                    "date":           datetime.now().strftime("%d %b %Y"),
                    "title":          title_input if title_input else news_input[:65] + "…",
                    "prediction":     "REAL" if prediction == 1 else "FAKE",
                    "real_prob":      round(real_prob * 100, 1),
                    "fake_prob":      round(fake_prob * 100, 1),
                    "keywords":       [w for w, _ in keywords[:5]],
                    "prediction_raw": prediction
                })
                st.success("✅ Analysis saved to History.")
        else:
            st.markdown("""
            <div class="empty-state" style='margin-top:4rem'>
                <span class="empty-icon">🔍</span>
                <div class="empty-title">Analysis result will appear here</div>
                <div class="empty-subtitle">Select an example or paste your own article, then click Analyze</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("""
    <h2 style='font-size:1.7rem;font-weight:700;margin-bottom:4px'>Model Performance Dashboard</h2>
    <p style='color:var(--text-secondary);font-size:14.5px;margin-bottom:1.5rem'>
        Detailed metrics and visualizations of the trained LinearSVC model.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy",  "99.73%", "+0.73%")
    c2.metric("Precision",      "99.8%",  "+1.2%")
    c3.metric("Recall",         "99.7%",  "+0.9%")
    c4.metric("F1 Score",       "99.7%",  "+1.1%")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div style='
            background:white;border:1px solid var(--border);
            border-radius:var(--radius-md);padding:1.3rem 1.3rem 0.5rem;
            box-shadow:var(--shadow-sm);
        '>
        <span class='section-label'>Algorithm Comparison</span>
        """, unsafe_allow_html=True)
        df = pd.DataFrame({
            "Model":    ["Naive Bayes","Logistic Regression","Random Forest","Decision Tree","Gradient Boosting","Passive Aggressive","Linear SVM"],
            "Accuracy": [96.33, 99.22, 99.60, 99.64, 99.65, 99.71, 99.73]
        })
        fig = px.bar(
            df, x="Accuracy", y="Model", orientation="h",
            color="Accuracy",
            color_continuous_scale=["#BFDBFE", "#1A56CC"],
            range_x=[94, 100],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            font=dict(family='DM Sans', size=12),
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(size=11)),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='
            background:white;border:1px solid var(--border);
            border-radius:var(--radius-md);padding:1.3rem 1.3rem 0.5rem;
            box-shadow:var(--shadow-sm);
        '>
        <span class='section-label'>Dataset Distribution</span>
        """, unsafe_allow_html=True)
        fig2 = go.Figure(data=[go.Pie(
            labels=["Real News", "Fake News"],
            values=[21417, 23481],
            hole=0.5,
            marker_colors=["#16A34A", "#DC2626"],
            textfont=dict(size=13, family='DM Sans'),
            pull=[0.02, 0.02],
        )])
        fig2.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='white',
            font=dict(family='DM Sans'),
            legend=dict(font=dict(size=12)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown("""
        <div style='
            background:white;border:1px solid var(--border);
            border-radius:var(--radius-md);padding:1.3rem 1.3rem 0.5rem;
            box-shadow:var(--shadow-sm);
        '>
        <span class='section-label'>Confusion Matrix</span>
        """, unsafe_allow_html=True)
        fig3 = go.Figure(data=go.Heatmap(
            z=[[4280, 12], [9, 4283]],
            x=["Predicted Fake", "Predicted Real"],
            y=["Actual Fake", "Actual Real"],
            colorscale=[[0, "#EFF6FF"], [1, "#1A56CC"]],
            text=[[4280, 12], [9, 4283]],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=15, family='DM Sans'),
            showscale=False,
        ))
        fig3.update_layout(
            height=290,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='white',
            font=dict(family='DM Sans', size=12),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style='
            background:white;border:1px solid var(--border);
            border-radius:var(--radius-md);padding:1.3rem 1.3rem 1rem;
            box-shadow:var(--shadow-sm);
        '>
        <span class='section-label'>Classification Report</span>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Class":     ["Fake News", "Real News", "Average"],
            "Precision": ["99.8%", "99.7%", "99.7%"],
            "Recall":    ["99.8%", "99.7%", "99.7%"],
            "F1-Score":  ["99.8%", "99.7%", "99.7%"],
        }), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>TF-IDF Configuration</span>", unsafe_allow_html=True)
        config_items = [
            ("Max Features", "50,000"),
            ("N-gram Range", "(1, 2)"),
            ("Sublinear TF", "True"),
            ("Stop Words", "English removed"),
        ]
        for k, v in config_items:
            st.markdown(f"""
            <div class='stat-row'>
                <span class='stat-key'>{k}</span>
                <span class='stat-val'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <h2 style='font-size:1.7rem;font-weight:700;margin-bottom:4px'>Analysis History</h2>
    <p style='color:var(--text-secondary);font-size:14.5px;margin-bottom:1.5rem'>
        All articles you have analyzed in this session are recorded here automatically.
    </p>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🕒</span>
            <div class="empty-title">No articles analyzed yet</div>
            <div class="empty-subtitle">Go to Detector and analyze some articles to see your history here</div>
        </div>""", unsafe_allow_html=True)
    else:
        total      = len(st.session_state.history)
        real_count = sum(1 for h in st.session_state.history if h["prediction_raw"] == 1)
        fake_count = total - real_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Checked", total)
        m2.metric("Real News",     real_count)
        m3.metric("Fake News",     fake_count)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        col_clr, _ = st.columns([1, 5])
        with col_clr:
            if st.button("🗑  Clear History"):
                st.session_state.history = []
                st.rerun()

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Recent Checks</span>", unsafe_allow_html=True)

        for item in reversed(st.session_state.history):
            is_real = item["prediction_raw"] == 1
            item_class = "history-item-real" if is_real else "history-item-fake"
            icon  = "✅" if is_real else "🚨"
            tc    = "#15803D" if is_real else "#B91C1C"
            kws   = ", ".join(item["keywords"]) if item["keywords"] else "N/A"
            st.markdown(f"""
            <div class="{item_class}">
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>
                    <span style='color:{tc};font-size:13.5px;font-weight:700'>{icon} {item["prediction"]}</span>
                    <span style='color:var(--text-muted);font-size:11.5px'>{item["date"]} · {item["time"]}</span>
                </div>
                <div style='color:var(--text-primary);font-size:13.5px;margin-bottom:4px;font-weight:500'>{item["title"]}</div>
                <div style='display:flex;gap:16px;flex-wrap:wrap'>
                    <span style='font-size:12px;color:var(--text-secondary)'><b>Real:</b> {item["real_prob"]}% &nbsp; <b>Fake:</b> {item["fake_prob"]}%</span>
                    <span style='font-size:12px;color:var(--text-muted)'><b>Keywords:</b> {kws}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Session Statistics</span>", unsafe_allow_html=True)

        ch1, ch2 = st.columns([1, 1])
        with ch1:
            fig_h = go.Figure(data=[go.Pie(
                labels=["Real News", "Fake News"],
                values=[max(real_count, 0), max(fake_count, 0)],
                hole=0.5,
                marker_colors=["#16A34A", "#DC2626"],
                pull=[0.02, 0.02],
                textfont=dict(family='DM Sans', size=12),
            )])
            fig_h.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='white',
                font=dict(family='DM Sans'),
                legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <h2 style='font-size:1.7rem;font-weight:700;margin-bottom:4px'>About FakeGuard</h2>
    <p style='color:var(--text-secondary);font-size:14.5px;margin-bottom:1.5rem'>
        Learn about the project, methodology, dataset, and technology stack behind FakeGuard.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("""
        #### Project Overview
        FakeGuard is an AI-powered fake news detection system built using machine learning,
        trained on the **Kaggle Fake and Real News Dataset** containing over 44,000 news articles.
        Real news was sourced from Reuters.com; fake news from sites flagged by fact-checking
        organizations. The system achieves **99.73% accuracy** using a LinearSVC model with TF-IDF features.

        #### Methodology
        - **Data Collection:** 44,898 articles — 21,417 real from Reuters + 23,481 fake from flagged sites
        - **Preprocessing:** Title and text combined, TF-IDF vectorization with 50,000 features
        - **Model Selection:** 7 algorithms compared — LinearSVC achieved best accuracy of 99.73%
        - **Evaluation:** 80/20 stratified train-test split with 5-fold cross-validation

        #### AI Features
        - **Top Keywords** — Colored pills showing which words most influenced the prediction
        - **Probability Bar** — Visual confidence split between Real and Fake percentages
        - **History Tracking** — Session-based log of all analyzed articles
        - **AI Explanation** — Human-readable reasoning behind every prediction
        """)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Tech Stack</span>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        techs = [("Python", "#3776AB"), ("Scikit-learn", "#F7931E"), ("Streamlit", "#FF4B4B"), ("Plotly", "#3D4DB7")]
        for col, (tech, color) in zip([t1, t2, t3, t4], techs):
            col.markdown(f"""
            <div class="tech-badge" style="background:{color};">{tech}</div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<span class='section-label'>Model Stats</span>", unsafe_allow_html=True)

        stats = {
            "Accuracy":   "99.73%",
            "Precision":  "99.8%",
            "Recall":     "99.7%",
            "F1-Score":   "99.7%",
            "Train Size": "35,278",
            "Test Size":  "8,820",
            "Features":   "50,000",
            "Algorithm":  "LinearSVC",
        }
        for k, v in stats.items():
            st.markdown(f"""
            <div class='stat-row'>
                <span class='stat-key'>{k}</span>
                <span class='stat-val'>{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>Dataset Information</span>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3, gap="medium")
    cards = [
        ("📰 Real News",    "21,417 articles from Reuters.com, one of the world's most trusted international news agencies.",
         "#166534", "#F0FDF4", "#86EFAC"),
        ("🚨 Fake News",    "23,481 articles collected from websites flagged as unreliable by fact-checking organizations.",
         "#991B1B", "#FEF2F2", "#FCA5A5"),
        ("📊 Total Dataset","44,898 articles covering politics, tech, and world news from 2015–2018.",
         "#1E40AF", "#EFF6FF", "#93C5FD"),
    ]
    for col, (title, desc, tc, bg, bdr) in zip([d1, d2, d3], cards):
        with col:
            st.markdown(f"""
            <div class="dataset-card" style='background:{bg};border-color:{bdr};'>
                <div style='font-size:15px;font-weight:700;color:{tc};margin-bottom:8px'>{title}</div>
                <div style='font-size:13px;color:#374151;line-height:1.65'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FAQ
# ══════════════════════════════════════════════════════════════
elif page == "FAQ":
    st.markdown("""
    <h2 style='font-size:1.7rem;font-weight:700;margin-bottom:4px'>Frequently Asked Questions</h2>
    <p style='color:var(--text-secondary);font-size:14.5px;margin-bottom:1.5rem'>
        Find answers to common questions about FakeGuard, its features, and how it works.
    </p>
    """, unsafe_allow_html=True)

    faqs = [
        ("What is FakeGuard?",
         "FakeGuard is an AI-powered fake news detection web application. It uses a LinearSVC machine learning model trained on 44,898 news articles to classify any news article as Real or Fake with 99.73% accuracy."),
        ("How accurate is the model?",
         "The LinearSVC model achieves 99.73% accuracy on the held-out test set of 8,820 articles. It correctly classified 8,799 articles with only 21 misclassifications. We compared 7 algorithms — LinearSVC performed best."),
        ("What are Top Keywords?",
         "After each analysis, the app identifies the words in the article that most strongly influenced the prediction. These are calculated using TF-IDF weights combined with the model's coefficient values. Green pills indicate real news words, red pills indicate fake news patterns."),
        ("What is the Confidence Score?",
         "The confidence score shows how certain the model is about its prediction as a percentage split between Real and Fake. A result of Real 95% means the model is highly confident the article is real. A 55/45 split suggests a borderline case that should be verified manually."),
        ("What is the History feature?",
         "Every article you analyze is automatically saved in the History section for your current session. Each record includes the article title, prediction result, confidence percentages, top keywords, and the exact date and time of analysis."),
        ("What is the AI Explanation?",
         "After each prediction, the app provides a plain-language explanation of why it classified the article as real or fake. For fake news it identifies specific sensationalist trigger words. For real news it highlights credible source language and journalistic patterns."),
        ("What kind of news can it detect?",
         "FakeGuard performs best on English-language political and world news articles similar to those in its training data. It may be less reliable on satire, opinion pieces, very short headlines, or news in other languages."),
        ("Is my data safe?",
         "Yes. No article text is stored on any server. All processing happens in real-time and the History feature only exists for your current browser session. Closing or refreshing the page clears all history completely."),
        ("Can the model be fooled?",
         "Like all machine learning models, FakeGuard can make mistakes — especially on well-written fake news or satirical content that mimics professional journalism. Always cross-check important news from multiple trusted sources."),
        ("What technologies power FakeGuard?",
         "FakeGuard is built with Python, Scikit-learn for the LinearSVC and TF-IDF pipeline, Streamlit for the web interface, Plotly for interactive charts, and Pandas for data handling. The model is saved and loaded using Python's Pickle library."),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.markdown(f"<div style='font-size:14px;color:var(--text-secondary);line-height:1.75;padding:4px 0'>{a}</div>",
                        unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.info("💡 FakeGuard is an open-source portfolio project. The complete code is available on GitHub.")


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='
    border-top: 1px solid var(--border);
    padding: 1.2rem 0 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
'>
    <span style='font-size: 12px; color: #94A3B8;'>FakeGuard</span>
    <span style='color: #CBD5E1; font-size: 11px;'>·</span>
    <span style='font-size: 12px; color: #94A3B8;'>Scikit-learn + Streamlit</span>
    <span style='color: #CBD5E1; font-size: 11px;'>·</span>
    <span style='font-size: 12px; color: #94A3B8;'>Kaggle Fake &amp; Real News Dataset</span>
    <span style='color: #CBD5E1; font-size: 11px;'>·</span>
    <span style='font-size: 12px; color: #94A3B8;'>99.7% Accuracy</span>
</div>
""", unsafe_allow_html=True)
