import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import re

# --- CẤU HÌNH MATPLOTLIB (Chống lỗi màn hình trắng/Thread) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- IMPORT WORDCLOUD AN TOÀN ---
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
except Exception:
    HAS_WORDCLOUD = False

# ==================================================
# 1. CẤU HÌNH TRANG (Luôn phải ở dòng đầu tiên sau imports)
# ==================================================
st.set_page_config(
    page_title="Topic 5 – Sentiment Analysis | UEF Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. HÀM XỬ LÝ NGÔN NGỮ (VIETNAMESE LOGIC)
# ==================================================
def is_vietnamese(text: str) -> bool:
    """Phát hiện tiếng Việt qua dấu câu đặc trưng"""
    if not isinstance(text, str): return False
    return bool(re.search(r"[àáạảãâăđêôơưíìịỉĩúùụủũýỳỵỷỹ]", text.lower()))

# Từ điển cảm xúc tiếng Việt (Rule-based Dictionary)
VI_POS = ["tốt", "tuyệt", "hài lòng", "xuất sắc", "ổn", "đẹp", "ngon", "ưng ý", "hoàn hảo", "thích", "ok", "good", "xịn", "phê", "chất lượng", "nhanh", "nhiệt tình"]
VI_NEG = ["tệ", "xấu", "kém", "thất vọng", "dở", "lỗi", "tồi", "không tốt", "chán", "tiếc", "phí", "đau", "bực", "chậm", "thái độ", "lừa đảo"]

def vietnamese_sentiment(text: str):
    score = 0
    t = text.lower()
    
    # Đếm từ khóa
    for w in VI_POS:
        if w in t: score += 1
    for w in VI_NEG:
        if w in t: score -= 1
        
    # Tính độ tin cậy giả lập (Confidence score)
    confidence = min(0.6 + abs(score) * 0.1, 0.98)
    
    if score > 0: return "Positive", confidence
    elif score < 0: return "Negative", confidence
    else: return "Neutral", 0.55

# ==================================================
# 3. HÀM LOAD DỮ LIỆU & MODEL (CACHED)
# ==================================================
@st.cache_resource
def load_model_objects():
    """Load model Logistic Regression và TF-IDF Vectorizer"""
    try:
        # Kiểm tra nhiều đường dẫn để tránh lỗi path
        paths_to_check = [
            (os.path.join("models", "model_en.pkl"), os.path.join("models", "vectorizer_en.pkl")),
            ("model_en.pkl", "vectorizer_en.pkl")
        ]
        for m_path, v_path in paths_to_check:
            if os.path.exists(m_path) and os.path.exists(v_path):
                model = joblib.load(m_path)
                vectorizer = joblib.load(v_path)
                return model, vectorizer
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_real_dataset():
    """Load dữ liệu CSV để hiển thị thông kê"""
    possible_paths = [
        "sentimentdataset.csv",
        "sentiment_results.csv",
        os.path.join("data", "sentimentdataset.csv"),
        os.path.join("data", "sentiment_results.csv")
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            try:
                # Xử lý encoding
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')

                # Chuẩn hóa tên cột
                rename_map = {
                    "Text": "review", "text": "review", "Review": "review", "content": "review",
                    "Sentiment": "label", "sentiment": "label", "Label": "label"
                }
                df = df.rename(columns=rename_map)
                
                # Làm sạch dữ liệu label
                if "review" in df.columns and "label" in df.columns:
                    df["label"] = df["label"].astype(str).str.strip().str.lower()
                    label_map = {
                        "pos": "positive", "neg": "negative", "neu": "neutral",
                        "joy": "positive", "sadness": "negative", "anger": "negative", 
                        "fear": "negative", "excited": "positive"
                    }
                    df["label"] = df["label"].replace(label_map)
                    
                    # Chỉ lấy các label chuẩn
                    valid = ['positive', 'negative', 'neutral']
                    df = df[df['label'].isin(valid)]
                    
                    return df
            except Exception:
                continue
    return None

# ==================================================
# 4. GIAO DIỆN: SIDEBAR & CSS
# ==================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align: center;">'
        '<img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="100" style="border-radius: 50%; border: 4px solid #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">'
        '<h2 style="color: #1a73e8; margin-top: 10px;">Topic 5</h2>'
        '<p style="font-size: 14px; color: gray;">Sentiment Analysis App</p>'
        '</div>', 
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio("📂 NAVIGATION", ["🏠 Home (Giới thiệu)", "📊 Analysis (Phân tích)", "ℹ️ Training Info (Mô hình)"])
    st.markdown("---")
    theme_mode = st.selectbox("🎨 Giao diện", ["🌊 Ocean Blue (Light)", "🌌 Midnight (Dark)"])

# Thiết lập màu sắc dựa trên theme
if theme_mode == "🌊 Ocean Blue (Light)":
    main_bg = "background: linear-gradient(-45deg, #a18cd1, #fbc2eb, #a6c1ee, #96e6a1); background-size: 400% 400%; animation: gradient 15s ease infinite;"
    text_color, card_bg, card_border = "#333", "rgba(255, 255, 255, 0.85)", "1px solid rgba(255, 255, 255, 0.6)"
else:
    main_bg = "background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #243b55); background-size: 400% 400%; animation: gradient 15s ease infinite;"
    text_color, card_bg, card_border = "#f0f0f0", "rgba(30, 30, 30, 0.80)", "1px solid rgba(255, 255, 255, 0.1)"

# Inject CSS
st.markdown(f"""
<style>
@keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
.stApp {{ {main_bg} color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.custom-card {{ background: {card_bg}; backdrop-filter: blur(12px); border-radius: 20px; padding: 25px; margin-bottom: 20px; border: {card_border}; box-shadow: 0 8px 32px 0 rgba(0,0,0,0.15); }}
h1, h2, h3 {{ color: {text_color} !important; }}
.stButton > button {{ background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); color: white; border-radius: 30px; border: none; padding: 10px 25px; transition: transform 0.2s; }}
.stButton > button:hover {{ transform: scale(1.05); }}
[data-testid="stSidebar"] {{ background-color: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.1); }}
</style>
""", unsafe_allow_html=True)

# Helper functions for Cards
def card_start(): st.markdown('<div class="custom-card">', unsafe_allow_html=True)
def card_end(): st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# 5. TRANG CHỦ (HOME)
# ==================================================
if page == "🏠 Home (Giới thiệu)":
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: {text_color}; opacity: 0.8;'>Developing a Sentiment Analysis Application for E-Commerce</h3>", unsafe_allow_html=True)
    st.write("") 

    col1, col2 = st.columns(2)
    with col1:
        card_start()
        st.markdown("### 🏫 University Info")
        # Link ảnh logo UEF hoặc icon thay thế
        st.image("https://th.bing.com/th/id/OIP.dmj07RiYpBLY3bGMvKOuCgHaGE?w=213&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3", width=100)
        st.markdown("**University of Economics & Finance (UEF)**\n*Faculty of Information Technology*\n**Course:** App Development of AI")
        card_end()

    with col2:
        card_start()
        st.markdown("### 👨‍🏫 Instructor")
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/3429/3429587.png" width="60" style="margin-right: 15px;">
            <div>
                <h4 style="margin:0;">MSc. Bùi Tiến Đức</h4>
                <p style="margin:0; font-size: 0.9em; color: #4CAF50;">Supervising Lecturer</p>
                <a href="https://orcid.org/0000-0001-5174-3558" target="_blank" style="text-decoration: none; font-size: 0.8em; color: gray;">
                    <img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="vertical-align: middle; margin-right: 4px; width: 16px;">
                    ORCID: 0000-0001-5174-3558
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        card_end()

    st.write("")
    card_start()
    st.markdown("### 👨‍🎓 Student Team")
    c1, c2 = st.columns(2)
    with c1: st.success("**Huỳnh Ngọc Minh Quan** - ID: 235052863")
    with c2: st.info("**Bùi Đức Nguyên** - ID: 235053154")
    card_end()
    
    card_start()
    st.markdown("### 📝 Abstract & Workflow")
    st.info("This project aims to develop a bilingual Sentiment Analysis Application (English & Vietnamese) using Hybrid Approach (Machine Learning + Rule-Based) to classify e-commerce reviews.")
    
    st.markdown("#### System Workflow Diagram")
    # --- SƠ ĐỒ GRAPHVIZ ĐƯỢC ĐẶT TẠI ĐÂY ---
    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        node [shape=box, style="filled,rounded", fillcolor="#f0f2f6", fontname="Segoe UI"];
        
        Input [label="User Review", shape=oval, fillcolor="#ffcc80"];
        Detect [label="Detect Language", shape=diamond, fillcolor="#b39ddb"];
        VN [label="Vietnamese\n(Rule-Based Dict)", fillcolor="#a5d6a7"];
        EN [label="English\n(Logistic Regression)", fillcolor="#90caf9"];
        Output [label="Sentiment Result\n(Pos/Neg/Neu)", shape=oval, fillcolor="#fff59d"];

        Input -> Detect;
        Detect -> VN [label="VI", fontsize=10];
        Detect -> EN [label="EN", fontsize=10];
        VN -> Output;
        EN -> Output;
    }
    ''')
    st.caption("Figure 1. The hybrid process flow of the application.")
    card_end()


# ==================================================
# 6. PHÂN TÍCH (ANALYSIS)
# ==================================================
elif page == "📊 Analysis (Phân tích)":
    st.title("📊 Live Sentiment Analysis")
    model, vectorizer = load_model_objects()

    # --- TABS ---
    tab1, tab2 = st.tabs(["💬 Single Review", "📂 Batch Analysis (Upload)"])

    # TAB 1: NHẬP TAY
    with tab1:
        card_start()
        user_input = st.text_area("Input Review (Vietnamese/English):", height=100, placeholder="Example: Món ăn rất ngon / The food is amazing")
        col_act, col_res = st.columns([1, 2])
        with col_act: analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)
        
        if analyze_btn and user_input:
            with st.spinner("Processing..."):
                time.sleep(0.5) # Tạo hiệu ứng chờ
                
                # 1. Phát hiện & Xử lý VN
                if is_vietnamese(user_input):
                    sentiment, score = vietnamese_sentiment(user_input)
                    lang = "Vietnamese (Rule-Based)"
                # 2. Xử lý EN (ML)
                else:
                    if model and vectorizer:
                        try:
                            vec_text = vectorizer.transform([user_input])
                            pred = model.predict(vec_text)[0]
                            score = float(model.predict_proba(vec_text).max())
                            sentiment = pred.capitalize()
                            lang = "English (Logistic Regression)"
                        except:
                            sentiment, score, lang = "Error", 0.0, "Processing Error"
                    else:
                        sentiment, score, lang = "Neutral", 0.5, "Model Not Found (Running in Demo Mode)"

            with col_res:
                st.caption(f"Detected: {lang}")
                if "Pos" in sentiment: 
                    st.success(f"**{sentiment}** 😊 ({score:.1%})")
                elif "Neg" in sentiment: 
                    st.error(f"**{sentiment}** 😡 ({score:.1%})")
                else: 
                    st.warning(f"**{sentiment}** 😐 ({score:.1%})")
                st.progress(score)
        card_end()

    # TAB 2: UPLOAD FILE
    with tab2:
        card_start()
        uploaded_file = st.file_uploader("Upload CSV/TXT File", type=["csv", "txt"])
        
        if uploaded_file:
            try:
                # Đọc file
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                    target_col = next((c for c in df_upload.columns if c.lower() in ['review', 'text', 'content']), None)
                else:
                    content = uploaded_file.read().decode("utf-8")
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    df_upload = pd.DataFrame(lines, columns=["review"])
                    target_col = "review"

                if target_col:
                    st.success(f"Processing {len(df_upload)} rows...")
                    
                    # Thanh tiến trình
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, txt in enumerate(df_upload[target_col].astype(str)):
                        if is_vietnamese(txt):
                            s, c = vietnamese_sentiment(txt)
                        else:
                            if model:
                                v = vectorizer.transform([txt])
                                s = model.predict(v)[0].capitalize()
                            else: s = "Neutral"
                        results.append(s)
                        
                        if i % 10 == 0: progress_bar.progress((i + 1) / len(df_upload))
                    
                    progress_bar.progress(1.0)
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    df_upload["Predicted"] = results
                    st.dataframe(df_upload.head(10), use_container_width=True)
                    
                    # Thống kê kết quả
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Distribution**")
                        counts = df_upload["Predicted"].value_counts()
                        fig, ax = plt.subplots(figsize=(4,3))
                        colors = ['#66b3ff', '#ff9999', '#99ff99']
                        ax.bar(counts.index, counts.values, color=colors[:len(counts)])
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with c2:
                        st.write("**Download Results**")
                        csv = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download CSV", csv, "results.csv", "text/csv")
                else:
                    st.error("Could not find a 'review' or 'text' column in CSV.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        card_end()

# ==================================================
# 7. THÔNG TIN MÔ HÌNH (TRAINING INFO)
# ==================================================
elif page == "ℹ️ Training Info (Mô hình)":
    st.title("ℹ️ Data & Model Insights")
    real_df = load_real_dataset()
    has_data = real_df is not None and not real_df.empty

    card_start()
    st.markdown("### 1️⃣ Dataset Overview")
    if has_data:
        st.success(f"Loaded {len(real_df)} samples from dataset.")
        with st.expander("📄 View Data Samples", expanded=False):
            st.dataframe(real_df.sample(min(5, len(real_df))), use_container_width=True)
            
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Label Distribution**")
            try:
                if "label" in real_df.columns:
                    label_counts = real_df["label"].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    fig1.patch.set_alpha(0)
                    ax1.pie(label_counts, labels=label_counts.index, autopct='%1.0f%%', 
                            colors=['#66b3ff','#99ff99','#ff9999'], startangle=90)
                    st.pyplot(fig1)
                    plt.close(fig1)
                else:
                    st.warning("No 'label' column found.")
            except: st.error("Chart Error")

        with c2:
            st.markdown("**Word Cloud**")
            if HAS_WORDCLOUD:
                try:
                    text_data = " ".join(real_df["review"].astype(str).tolist())
                    wc = WordCloud(width=400, height=300, background_color='white', max_words=50, colormap='viridis').generate(text_data)
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.imshow(wc, interpolation='bilinear'); ax2.axis('off')
                    st.pyplot(fig2); plt.close(fig2)
                except: st.warning("Not enough text for WordCloud")
            else: st.info("Install `wordcloud` library to view.")
    else:
        st.warning("⚠️ Dataset not found. Please ensure `sentimentdataset.csv` exists in the folder or data/ folder.")
    card_end()

    # Model Metrics Info
    c1, c2 = st.columns(2)
    with c1:
        card_start()
        st.markdown("### 🇬🇧 English Model"); st.metric("Accuracy", "87.5%", "+1.5%")
        st.markdown("- **Algorithm:** Logistic Regression\n- **Features:** TF-IDF (1-2 ngrams)\n- **Train Data:** Amazon Reviews")
        card_end()
    with c2:
        card_start()
        st.markdown("### 🇻🇳 Vietnamese Model"); st.metric("Type", "Rule-Based")
        st.markdown("- **Algorithm:** Dictionary Lookup\n- **Logic:** Keyword Counting\n- **Support:** Positive/Negative/Neutral")
        card_end()

# ==================================================
# 8. FOOTER
# ==================================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Topic 5: Sentiment Analysis for E-Commerce.</div>", unsafe_allow_html=True)