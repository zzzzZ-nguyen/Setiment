import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Cấu hình Matplotlib an toàn (Tránh lỗi thread) ---
import matplotlib
matplotlib.use('Agg')

# --- Import an toàn các thư viện phụ ---
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# --- Import PyTorch (Nếu có) ---
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==================================================
# 1. ĐỊNH NGHĨA CLASS PYTORCH (Bắt buộc để load model .pth)
# ==================================================
if HAS_TORCH:
    class SentimentNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SentimentNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

# ==================================================
# 2. HÀM XỬ LÝ NGÔN NGỮ (RULE-BASED VIETNAMESE)
# ==================================================
def is_vietnamese(text: str) -> bool:
    """Phát hiện tiếng Việt qua dấu câu đặc trưng"""
    if not isinstance(text, str): return False
    return bool(re.search(r"[àáạảãâăđêôơưíìịỉĩúùụủũýỳỵỷỹ]", text.lower()))

# Từ điển cảm xúc (Simple Dictionary)
VI_POS = ["tốt", "tuyệt", "hài lòng", "xuất sắc", "ổn", "đẹp", "ngon", "ưng ý", "hoàn hảo", "thích", "ok", "good", "xịn", "phê", "chất lượng", "đáng tiền", "nhanh", "nhiệt tình"]
VI_NEG = ["tệ", "xấu", "kém", "thất vọng", "dở", "lỗi", "tồi", "không tốt", "chán", "tiếc", "phí", "đau", "bực", "chậm", "thái độ", "lừa đảo", "hỏng"]

def vietnamese_sentiment(text: str):
    score = 0
    t = text.lower()
    for w in VI_POS:
        if w in t: score += 1
    for w in VI_NEG:
        if w in t: score -= 1

    # Tính confidence giả lập
    confidence = min(0.6 + abs(score) * 0.1, 0.99)
    
    if score > 0: return "positive", confidence
    elif score < 0: return "negative", confidence
    else: return "neutral", 0.55

# ==================================================
# 3. LOAD MODEL (HỖ TRỢ CẢ SKLEARN VÀ PYTORCH)
# ==================================================
@st.cache_resource
def load_models():
    """
    Load:
    1. Vectorizer
    2. Logistic Regression Model (.pkl)
    3. PyTorch Model (.pth) - Optional
    """
    paths_to_check = ["models", os.path.join("..", "models"), "."]
    model_dir = None
    
    # Tìm thư mục chứa model
    for p in paths_to_check:
        if os.path.exists(os.path.join(p, "model_en.pkl")):
            model_dir = p
            break
            
    sk_model = None
    vectorizer = None
    torch_model = None
    torch_classes = {0: 'negative', 1: 'neutral', 2: 'positive'} # Default

    if model_dir:
        # Load Sklearn
        try:
            sk_model = joblib.load(os.path.join(model_dir, "model_en.pkl"))
            vectorizer = joblib.load(os.path.join(model_dir, "vectorizer_en.pkl"))
        except: pass

        # Load PyTorch
        if HAS_TORCH:
            try:
                pth_path = os.path.join(model_dir, "model_en_torch.pth")
                if os.path.exists(pth_path):
                    checkpoint = torch.load(pth_path)
                    input_dim = checkpoint.get('input_dim', 5000)
                    hidden_dim = checkpoint.get('hidden_dim', 128)
                    output_dim = checkpoint.get('output_dim', 3)
                    
                    t_model = SentimentNN(input_dim, hidden_dim, output_dim)
                    t_model.load_state_dict(checkpoint['model_state_dict'])
                    t_model.eval()
                    torch_model = t_model
                    
                    # Cập nhật class mapping nếu có trong file save
                    if 'classes' in checkpoint:
                        # Đảo ngược mapping: {'neg':0} -> {0:'neg'}
                        cls = checkpoint['classes']
                        if isinstance(list(cls.keys())[0], str):
                             torch_classes = {v: k for k, v in cls.items()}
                        else:
                             torch_classes = cls
            except Exception as e:
                print(f"PyTorch load error: {e}")

    return sk_model, vectorizer, torch_model, torch_classes

# ==================================================
# 4. HÀM DỰ ĐOÁN CHUNG
# ==================================================
def predict_sentiment(text, use_torch=False, sk_model=None, vectorizer=None, torch_model=None, torch_classes=None):
    # 1. Check Tiếng Việt -> Rule Based
    if is_vietnamese(text):
        lbl, conf = vietnamese_sentiment(text)
        return lbl, conf, "Rule-Based (VN)"

    # 2. Check Tiếng Anh/Khác -> Machine Learning
    if not vectorizer:
        return "neutral", 0.0, "No Model"

    try:
        vec = vectorizer.transform([text])
        
        # Dùng PyTorch (Neural Network)
        if use_torch and torch_model and HAS_TORCH:
            tensor_in = torch.tensor(vec.toarray(), dtype=torch.float32)
            with torch.no_grad():
                outputs = torch_model(tensor_in)
                probs = torch.softmax(outputs, dim=1)
                max_prob, idx = torch.max(probs, 1)
                label = torch_classes.get(idx.item(), "unknown")
                return label, max_prob.item(), "Neural Network (PyTorch)"
        
        # Dùng Logistic Regression (Mặc định)
        elif sk_model:
            label = sk_model.predict(vec)[0]
            conf = sk_model.predict_proba(vec).max()
            return label, conf, "Logistic Regression"
            
    except Exception as e:
        return "error", 0.0, str(e)

    return "neutral", 0.5, "Fallback"

# ==================================================
# 5. GIAO DIỆN CHÍNH
# ==================================================
def show():
    st.markdown("<h2 style='color:#2b6f3e;'>📊 Advanced Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.caption("Hybrid System: Rule-Based (VN) + ML/Deep Learning (EN)")

    # Load Resources
    sk_model, vectorizer, torch_model, torch_classes = load_models()
    
    # Sidebar/Option Settings
    with st.expander("⚙️ Analysis Settings", expanded=False):
        model_option = st.radio(
            "Select English Model:",
            ("Logistic Regression (Fast)", "Neural Network (Accurate)"),
            index=0
        )
        use_torch = "Neural Network" in model_option and torch_model is not None
        
        if "Neural Network" in model_option and torch_model is None:
            st.warning("⚠️ PyTorch model not found. Falling back to Logistic Regression.")

    # TABS
    tab1, tab2 = st.tabs(["💬 Single Analysis", "📂 Batch Analysis (File)"])

    # --- TAB 1: SINGLE INPUT ---
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.text_area("Enter Customer Review:", height=150, placeholder="Example: Món ăn rất ngon nhưng phục vụ hơi chậm...")
            analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)
        
        with col2:
            st.info("💡 **Tip:** System automatically detects language.")
            if analyze_btn and user_input:
                with st.spinner("Processing..."):
                    time.sleep(0.3)
                    
                    label, conf, method = predict_sentiment(
                        user_input, 
                        use_torch=use_torch, 
                        sk_model=sk_model, 
                        vectorizer=vectorizer, 
                        torch_model=torch_model,
                        torch_classes=torch_classes
                    )

                    # Hiển thị
                    st.divider()
                    st.markdown(f"**Method:** `{method}`")
                    
                    if label == "positive":
                        st.success(f"### 😊 Positive ({conf:.1%})")
                    elif label == "negative":
                        st.error(f"### 😡 Negative ({conf:.1%})")
                    else:
                        st.warning(f"### 😐 Neutral ({conf:.1%})")

    # --- TAB 2: BATCH ANALYSIS ---
    with tab2:
        st.markdown("### Upload Dataset")
        file = st.file_uploader("Choose file (CSV, TXT, DOCX)", type=["csv", "txt", "docx"])
        
        if file:
            reviews = []
            filename = file.name
            
            # Đọc File
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(file)
                    text_col = next((c for c in df.columns if c.lower() in ['review', 'text', 'content']), None)
                    if text_col: reviews = df[text_col].astype(str).tolist()
                elif filename.endswith(".txt"):
                    reviews = [line.strip() for line in file.read().decode("utf-8").splitlines() if line.strip()]
                elif filename.endswith(".docx") and HAS_DOCX:
                    doc = Document(file)
                    reviews = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            except Exception as e:
                st.error(f"Error reading file: {e}")

            # Xử lý
            if reviews:
                st.info(f"Analyzing {len(reviews)} reviews with **{'Neural Network' if use_torch else 'Logistic Regression'}**...")
                
                results = []
                bar = st.progress(0)
                
                for i, txt in enumerate(reviews):
                    l, c, m = predict_sentiment(txt, use_torch, sk_model, vectorizer, torch_model, torch_classes)
                    results.append({"Review": txt, "Sentiment": l, "Confidence": round(c, 2)})
                    if i % 5 == 0: bar.progress((i + 1) / len(reviews))
                
                bar.progress(1.0); time.sleep(0.2); bar.empty()
                
                # Hiển thị kết quả
                res_df = pd.DataFrame(results)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(res_df))
                c2.metric("Positive", len(res_df[res_df['Sentiment']=='positive']))
                c3.metric("Negative", len(res_df[res_df['Sentiment']=='negative']))
                
                st.divider()
                
                # Charts
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    try:
                        counts = res_df["Sentiment"].value_counts()
                        fig1, ax1 = plt.subplots(figsize=(3, 3))
                        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999','#99ff99'])
                        st.pyplot(fig1); plt.close(fig1)
                    except: pass
                
                with col_viz2:
                    if HAS_WORDCLOUD:
                        try:
                            txt_all = " ".join(res_df["Review"].astype(str))
                            wc = WordCloud(width=300, height=300, background_color='white').generate(txt_all)
                            fig2, ax2 = plt.subplots(figsize=(3, 3))
                            ax2.imshow(wc, interpolation='bilinear'); ax2.axis('off')
                            st.pyplot(fig2); plt.close(fig2)
                        except: pass

                st.download_button("⬇️ Download Result CSV", res_df.to_csv(index=False), "results.csv", "text/csv")

if __name__ == "__main__":
    show()