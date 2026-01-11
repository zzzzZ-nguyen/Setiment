import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import random

# --- 1. CẤU HÌNH MATPLOTLIB (QUAN TRỌNG ĐỂ TRÁNH TREO APP) ---
import matplotlib
matplotlib.use('Agg') # Bắt buộc dùng backend không giao diện
import matplotlib.pyplot as plt

# --- 2. IMPORT AN TOÀN CHO CÁC THƯ VIỆN BỔ SUNG ---
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==================================================
# 🧠 PYTORCH MODEL CLASS (Phải giống hệt file train)
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
# 📦 LOAD MODEL & DATA OBJECTS
# ==================================================
@st.cache_resource
def load_model_objects():
    """Load Scikit-learn Model, Vectorizer và PyTorch Model (nếu có)"""
    paths_to_check = [
        "models",
        os.path.join("..", "models"),
        "."
    ]
    
    sklearn_model = None
    vectorizer = None
    torch_model = None
    torch_info = None

    # 1. Tìm thư mục chứa model
    model_dir = None
    for p in paths_to_check:
        if os.path.exists(os.path.join(p, "model_en.pkl")):
            model_dir = p
            break
    
    if not model_dir:
        return None, None, None, None

    # 2. Load Scikit-Learn components
    try:
        sklearn_path = os.path.join(model_dir, "model_en.pkl")
        vec_path = os.path.join(model_dir, "vectorizer_en.pkl")
        
        if os.path.exists(sklearn_path) and os.path.exists(vec_path):
            sklearn_model = joblib.load(sklearn_path)
            vectorizer = joblib.load(vec_path)
    except Exception as e:
        print(f"Error loading Sklearn: {e}")

    # 3. Load PyTorch Model (Optional)
    if HAS_TORCH:
        try:
            torch_path = os.path.join(model_dir, "model_en_torch.pth")
            if os.path.exists(torch_path):
                checkpoint = torch.load(torch_path)
                
                # Khởi tạo model architecture từ checkpoint info
                input_dim = checkpoint.get('input_dim', 5000) # Default fallback
                hidden_dim = checkpoint.get('hidden_dim', 128)
                output_dim = checkpoint.get('output_dim', 3)
                
                t_model = SentimentNN(input_dim, hidden_dim, output_dim)
                t_model.load_state_dict(checkpoint['model_state_dict'])
                t_model.eval() # Set eval mode
                
                torch_model = t_model
                torch_info = checkpoint.get('classes', {0:'neg', 1:'neu', 2:'pos'})
        except Exception as e:
            print(f"Error loading PyTorch: {e}")

    return sklearn_model, vectorizer, torch_model, torch_info

# ==================================================
# 📥 DATA LOADING UTILS
# ==================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'#\d+', '', text)
    text = text.replace('_', ' ')
    return text.strip().lower()

def normalize_label(val):
    s = str(val).strip().upper()
    if s in ['POSITIVE', 'POS', 'GOOD', 'JOY', 'HAPPY', 'LOVE', 'LIKE']: return 'positive'
    if s in ['NEGATIVE', 'NEG', 'BAD', 'SAD', 'ANGER', 'HATE', 'DISLIKE']: return 'negative'
    if s in ['NEUTRAL', 'NEU', 'OKAY', 'NORMAL', 'AVERAGE']: return 'neutral'
    return None

def parse_sentiwordnet(df):
    data = []
    if 'PosScore' in df.columns and 'NegScore' in df.columns:
        for _, row in df.iterrows():
            try:
                p = float(row['PosScore'])
                n = float(row['NegScore'])
                term = str(row['SynsetTerms'])
                label = 'neutral'
                if p > n and p > 0: label = 'positive'
                elif n > p and n > 0: label = 'negative'
                
                terms = term.split(',')
                for t in terms:
                    cleaned = clean_text(t)
                    if len(cleaned) > 2:
                        data.append({'review': cleaned, 'label': label})
            except: continue
    return pd.DataFrame(data) if data else None

def parse_text_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        data = []
        buffer = []
        for line in lines:
            line = line.strip()
            if not line: continue
            lbl = normalize_label(line)
            if lbl:
                if buffer:
                    data.append({'review': " ".join(buffer), 'label': lbl})
                    buffer = []
            else:
                buffer.append(line)
        return pd.DataFrame(data) if data else None
    except: return None

@st.cache_data
def load_real_dataset():
    all_dfs = []
    files_map = {
        "csv_main": ["data/sentimentdataset.csv", "sentimentdataset.csv"],
        "swn": ["data/VietSentiWordnet", "VietSentiWordnet"],
        "txt": ["data/text_input.txt", "text_input.txt"]
    }

    # 1. Load CSV
    for p in files_map["csv_main"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding='utf-8', on_bad_lines='skip')
                df = df.rename(columns=lambda x: x.strip().lower())
                rename_map = {"text": "review", "content": "review", "sentiment": "label", "rating": "label"}
                df = df.rename(columns=rename_map)
                if "review" in df.columns and "label" in df.columns:
                    df["label"] = df["label"].apply(normalize_label)
                    all_dfs.append(df[["review", "label"]].dropna())
                    break
            except: pass

    # 2. Load SentiWordNet
    for p in files_map["swn"]:
        if os.path.exists(p):
            try:
                try: df = pd.read_csv(p, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
                except: df = pd.read_csv(p, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                if not df.empty and df.columns[0].startswith('#'):
                    cols = list(df.columns); cols[0] = cols[0].replace('#', '').strip(); df.columns = cols
                df_swn = parse_sentiwordnet(df)
                if df_swn is not None: all_dfs.append(df_swn)
            except: pass

    # 3. Load Text
    for p in files_map["txt"]:
        if os.path.exists(p):
            df_txt = parse_text_file(p)
            if df_txt is not None: all_dfs.append(df_txt)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['review'])
        return final_df
    return None

# ==================================================
# 📊 HÀM HIỂN THỊ CHÍNH
# ==================================================
def show():
    st.markdown("<h2 style='color:#1a73e8;'>ℹ️ Model Training Dashboard</h2>", unsafe_allow_html=True)
    st.write("Overview of dataset, metrics, and **Multi-Model Comparison**.")
    
    # Load Data & Models
    real_df = load_real_dataset()
    has_data = real_df is not None and not real_df.empty
    sk_model, vectorizer, torch_model, torch_classes = load_model_objects()

    st.markdown("---")

    # --- SECTION 1: DATA OVERVIEW ---
    st.subheader("1️⃣ Training Data Overview")
    if has_data:
        st.success(f"✅ Total Combined Samples: **{len(real_df)}**")
        
        with st.expander("📄 View Data Sample"):
            st.dataframe(real_df.sample(min(10, len(real_df))).reset_index(drop=True), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📊 Label Distribution")
            try:
                counts = real_df["label"].value_counts()
                fig, ax = plt.subplots(figsize=(4,4))
                fig.patch.set_alpha(0)
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999','#99ff99'])
                st.pyplot(fig)
                plt.close(fig)
            except: st.error("Chart Error")

        with c2:
            st.markdown("##### ☁️ Frequent Words")
            if HAS_WORDCLOUD:
                try:
                    text = " ".join(real_df["review"].astype(str).tolist())
                    wc = WordCloud(width=400, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(4,4))
                    ax.imshow(wc); ax.axis('off')
                    st.pyplot(fig); plt.close(fig)
                except: st.warning("Not enough data for WordCloud")
            else: st.warning("WordCloud lib missing")
    else:
        st.warning("⚠️ No data found.")

    st.markdown("---")

    # --- SECTION 2: MODEL STATUS ---
    st.subheader("2️⃣ Model Status")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        if sk_model:
            st.success("✅ **Logistic Regression (Sklearn)** Loaded")
            st.caption("Optimized for CPU & Speed")
        else:
            st.error("❌ Logistic Regression Missing")

    with col_m2:
        if torch_model:
            st.success("✅ **Neural Network (PyTorch)** Loaded")
            st.caption("Deep Learning (Feed Forward)")
        else:
            st.warning("⚠️ PyTorch Model Not Found (Run train script again to generate)")

    st.markdown("---")

    # --- SECTION 3: LIVE COMPARISON ---
    st.subheader("3️⃣ Live Model Comparison")
    st.caption("Pick random samples to see how different models classify them.")

    if sk_model and vectorizer and has_data:
        if st.button("🎲 Random Test (Compare Models)"):
            samples = real_df.sample(min(3, len(real_df)))
            
            for _, row in samples.iterrows():
                txt = row['review']
                lbl = row['label']
                
                # Preprocessing
                vec = vectorizer.transform([txt])
                
                # 1. Sklearn Predict
                pred_sk = sk_model.predict(vec)[0]
                prob_sk = sk_model.predict_proba(vec).max()
                
                # 2. PyTorch Predict (if avail)
                pred_torch = "N/A"
                prob_torch = 0.0
                if torch_model and HAS_TORCH:
                    try:
                        # Convert sparse matrix to dense tensor
                        tensor_in = torch.tensor(vec.toarray(), dtype=torch.float32)
                        with torch.no_grad():
                            outputs = torch_model(tensor_in)
                            probs = torch.softmax(outputs, dim=1)
                            max_prob, idx = torch.max(probs, 1)
                            
                            # Map index back to label string using saved classes
                            # Reverse map: {v: k for k, v in map.items()} needed if classes is {'neg':0}
                            # Assuming classes is {0: 'negative', ...} from previous script logic
                            idx_val = idx.item()
                            # Logic fallback mapping if needed
                            rev_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                            
                            # Nếu torch_classes lưu dạng {'negative': 0}, cần đảo key/value
                            if torch_classes and isinstance(list(torch_classes.keys())[0], str):
                                rev_map = {v: k for k, v in torch_classes.items()}
                            
                            pred_torch = rev_map.get(idx_val, "unknown")
                            prob_torch = max_prob.item()
                    except Exception as e:
                        pred_torch = f"Err: {str(e)[:10]}"

                # --- DISPLAY ---
                with st.container():
                    st.markdown(f"📝 *\"{str(txt)[:100]}...\"*")
                    st.caption(f"True Label: **{str(lbl).upper()}**")
                    
                    c_sk, c_torch = st.columns(2)
                    
                    # Sklearn Column
                    with c_sk:
                        is_corr = (pred_sk == lbl)
                        color = "green" if is_corr else "red"
                        st.markdown(f"""
                        <div style='background:#f0f2f6; padding:10px; border-radius:8px; border-left:4px solid {color}'>
                            <b>🤖 Logistic Regression</b><br>
                            Pred: <b style='color:{color}'>{pred_sk.upper()}</b><br>
                            Conf: {prob_sk:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # PyTorch Column
                    with c_torch:
                        if torch_model:
                            is_corr = (pred_torch == lbl)
                            color = "green" if is_corr else "red"
                            st.markdown(f"""
                            <div style='background:#e6fffa; padding:10px; border-radius:8px; border-left:4px solid {color}'>
                                <b>🧠 Neural Network</b><br>
                                Pred: <b style='color:{color}'>{pred_torch.upper()}</b><br>
                                Conf: {prob_torch:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("PyTorch model not loaded")
                    
                    st.divider()

if __name__ == "__main__":
    show()