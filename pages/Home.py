import streamlit as st

# ==========================================
# 🎨 UTILS: Giao diện Box đẹp (Glassmorphism)
# ==========================================
def info_box(title, content, icon="📌", color="#e6d784", bg_color="#fff7cc"):
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid {color};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            font-size: 16px;
            line-height: 1.6;
            color: #333;">
            <h4 style="color: {color}; margin-top: 0; display: flex; align-items: center;">
                <span style="margin-right: 10px;">{icon}</span> {title}
            </h4>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def keyword_badge(keyword):
    return f"<span style='background-color:#eee; padding:4px 10px; border-radius:15px; border:1px solid #ccc; font-size:14px; margin-right:5px;'>#{keyword}</span>"

# ==========================================
# 🏠 HOME PAGE CONTENT
# ==========================================
def show():
    # --- HEADER ---
    st.markdown("<h1 style='text-align: center; color: #b30000;'>PROJECT FINAL REPORT</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #2b6f3e;'>Developing a Sentiment Analysis Application</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Course: Application Development of Artificial Intelligence</h4>", unsafe_allow_html=True)
    st.write("---")

    # --- ADMINISTRATIVE INFO ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://th.bing.com/th/id/OIP.dmj07RiYpBLY3bGMvKOuCgHaGE?w=213&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3", width=120)
        st.markdown("""
        **University of Economics & Finance (UEF)** *Faculty of Information Technology*
        
        **Class ID:** 251.ITE1174E.B01E  
        **Year:** 2025
        """)

    with col2:
        st.markdown("### 👨‍🏫 Instructor & Team")
        # Sử dụng HTML để chèn icon ORCID và link đẹp mắt
        st.markdown("""
        **Supervising Lecturer:** 🎓 **MSc. Bùi Tiến Đức**
        
        <a href="https://orcid.org/0000-0001-5174-3558" target="_blank" style="text-decoration: none; color: #333; font-size: 14px; display: flex; align-items: center;">
            <img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="margin-right: 8px; vertical-align: middle;">
            https://orcid.org/0000-0001-5174-3558
        </a>
        
        **Student Team:**
        1. **Huỳnh Ngọc Minh Quan** (ID: 235052863)
        2. **Bùi Đức Nguyên** (ID: 235053154)
        """, unsafe_allow_html=True)

    st.write("---")

    # --- SECTION 1: ABSTRACT & KEYWORDS ---
    st.markdown("### 📝 Abstract & Keywords")
    
    abstract_content = """
    <b>Context:</b> The rapid growth of e-commerce has led to a massive volume of customer reviews, making manual analysis inefficient and time-consuming.<br><br>
    <b>Objective:</b> This project aims to develop a bilingual <b>Sentiment Analysis Application</b> capable of automatically classifying product reviews into <b>Positive, Negative, or Neutral</b> categories.<br><br>
    <b>Methodology:</b> The system utilizes a hybrid approach: 
    <b>Logistic Regression with TF-IDF</b> for English text and a <b>Rule-based Dictionary approach</b> for Vietnamese text. 
    The application is deployed using the <b>Streamlit</b> framework for real-time interaction.<br><br>
    <b>Results:</b> The proposed English model achieves an accuracy of approximately <b>86%</b>, providing businesses with actionable insights into customer satisfaction.
    """
    
    info_box("Abstract", abstract_content, icon="📄", color="#b30000", bg_color="#ffe6e6")

    # Keywords Badge Display
    keywords = ["Sentiment Analysis", "Natural Language Processing (NLP)", "Machine Learning", "Streamlit", "E-commerce", "Vietnamese Text Processing"]
    st.markdown("<b>Keywords:</b> " + " ".join([keyword_badge(k) for k in keywords]), unsafe_allow_html=True)
    
    st.write("") # Spacer

    # --- SECTION 2: PROPOSED MODELS ---
    st.markdown("### 🧠 Proposed Models & Techniques")
    
    model_content = """
    The application integrates two distinct approaches to handle bilingual data effectively:
    
    <ul>
        <li><b>🇺🇸 English Processing (Machine Learning):</b>
            <ul>
                <li><b>Algorithm:</b> <b style='color:#b30000'>Logistic Regression</b> (Selected for high efficiency and interpretability).</li>
                <li><b>Feature Engineering:</b> TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization.</li>
                <li><b>Dataset:</b> 50,000+ reviews from Amazon/Kaggle datasets.</li>
            </ul>
        </li>
        <br>
        <li><b>🇻🇳 Vietnamese Processing (Rule-Based):</b>
            <ul>
                <li><b>Algorithm:</b> <b style='color:#2b6f3e'>Dictionary/Rule-Based Approach</b>.</li>
                <li><b>Logic:</b> Keyword counting using a curated dictionary of positive/negative Vietnamese terms (e.g., "tốt", "tệ", "hài lòng").</li>
                <li><b>Handling:</b> Text normalization and negation handling (e.g., "không tốt" -> negative).</li>
            </ul>
        </li>
    </ul>
    """
    info_box("Model Architecture", model_content, icon="⚙️", color="#1a73e8", bg_color="#e6f2ff")

    # --- SECTION 3: SYSTEM WORKFLOW ---
    with st.expander("🔎 System Workflow (Click to expand)"):
        st.markdown("""
        1. **Input:** User enters text or uploads a CSV file.
        2. **Language Detection:** System identifies Vietnamese or English.
        3. **Processing:**
           - If English: TF-IDF -> Logistic Regression Model.
           - If Vietnamese: Tokenization -> Dictionary Lookup.
        4. **Output:** Sentiment Label (Pos/Neg/Neu) + Confidence Score.
        5. **Visualization:** Charts & Reports.
        """)

if __name__ == "__main__":
    show()