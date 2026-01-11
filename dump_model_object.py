# ==========================================
# 🧠 TRAINING SCRIPT - UNIVERSAL LOADER
# Hỗ trợ: CSV, SentiWordNet, Labeled Text File
# Models: Scikit-Learn (Logistic) + PyTorch (Neural Net)
# ==========================================

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- IMPORT PYTORCH ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ Warning: PyTorch not found. Skipping Neural Network training.")

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_DIR = "models"
POSSIBLE_PATHS = [
    "data/sentimentdataset.csv", 
    "data/VietSentiWordnet", 
    "data/VietSentiWordnet.txt",
    "data/sentiment_results.csv",
    "data/text_input.txt",
    "sentimentdataset.csv",
    "VietSentiWordnet",
    "VietSentiWordnet.txt",
    "sentiment_results.csv",
    "text_input.txt",
]

def clean_text(text):
    """Làm sạch văn bản cơ bản"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'#\d+', '', text) 
    text = text.replace('_', ' ')
    return text.strip().lower()

def normalize_label(val):
    """Chuẩn hóa nhãn về 3 loại chính"""
    s = str(val).strip().upper()
    if s in ['POSITIVE', 'POS', 'GOOD', 'JOY', 'HAPPY', 'LOVE', 'LIKE']: return 'positive'
    if s in ['NEGATIVE', 'NEG', 'BAD', 'SAD', 'ANGER', 'HATE', 'DISLIKE']: return 'negative'
    if s in ['NEUTRAL', 'NEU', 'OKAY', 'NORMAL', 'AVERAGE']: return 'neutral'
    return None

# ==========================================
# 1. DATA PARSERS (GIỮ NGUYÊN)
# ==========================================
def parse_csv(df, source_name=""):
    cols = df.columns.str.lower()
    text_col = next((c for c in df.columns if c.lower() in ['text', 'review', 'content', 'comment']), None)
    label_col = next((c for c in df.columns if c.lower() in ['sentiment', 'label', 'rating']), None)
    
    if text_col and label_col:
        print(f"   [CSV Parser] Found columns: {text_col}, {label_col}")
        df = df[[text_col, label_col]].dropna()
        df.columns = ['review', 'label']
        df['review'] = df['review'].apply(clean_text)
        df['label'] = df['label'].apply(normalize_label)
        return df.dropna()
    return None

def parse_sentiwordnet(df, source_name=""):
    req_cols = ['PosScore', 'NegScore', 'SynsetTerms']
    if not all(c in df.columns for c in req_cols):
        return None

    print(f"   [SentiWordNet Parser] Processing scores from {source_name}...")
    data = []
    for _, row in df.iterrows():
        try:
            term = str(row['SynsetTerms'])
            p_score = float(row['PosScore'])
            n_score = float(row['NegScore'])
            
            label = 'neutral'
            if p_score > n_score and p_score > 0: label = 'positive'
            elif n_score > p_score and n_score > 0: label = 'negative'
            
            terms = term.split(',')
            for t in terms:
                cleaned = clean_text(t)
                if cleaned and len(cleaned) > 2:
                    data.append({'review': cleaned, 'label': label})
        except:
            continue
    return pd.DataFrame(data)

def parse_text_file(path):
    print(f"   [Text Parser] Reading raw text from {path}...")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        data = []
        buffer_text = []
        for line in lines:
            line = line.strip()
            if not line: continue
            label_candidate = normalize_label(line)
            if label_candidate:
                if buffer_text:
                    full_text = " ".join(buffer_text)
                    data.append({'review': clean_text(full_text), 'label': label_candidate})
                    buffer_text = []
            else:
                buffer_text.append(line)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"   Warning: Text parse error - {e}")
        return None

def load_data():
    all_dfs = []
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            print(f"📂 Đang xử lý: {path}")
            if path.endswith('.txt') and "senti" not in path.lower():
                df = parse_text_file(path)
                if df is not None and not df.empty: all_dfs.append(df)
                continue
            try:
                try: df_raw = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
                except: df_raw = pd.read_csv(path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                
                if not df_raw.empty:
                    if df_raw.columns[0].startswith('#'):
                        new_cols = list(df_raw.columns)
                        new_cols[0] = new_cols[0].replace('#', '').strip()
                        df_raw.columns = new_cols
                    
                    df_senti = parse_sentiwordnet(df_raw, path)
                    if df_senti is not None: all_dfs.append(df_senti)
                    else:
                        df_csv = parse_csv(df_raw, path)
                        if df_csv is not None: all_dfs.append(df_csv)
            except Exception as e:
                print(f"   ⚠️ Lỗi đọc file cấu trúc bảng: {e}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['review']).dropna()
        print("="*40)
        print(f"🎉 TỔNG HỢP DỮ LIỆU: {len(final_df)} mẫu")
        print(f"   - Phân bố: {final_df['label'].value_counts().to_dict()}")
        print("="*40)
        return final_df['review'].tolist(), final_df['label'].tolist()
    
    print("\n⚠️ KHÔNG TÌM THẤY DỮ LIỆU. DÙNG MẪU.")
    return (["good", "bad"], ["positive", "negative"])

# ==========================================
# 2. PYTORCH MODEL DEFINITION
# ==========================================
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
            return x # Return logits, CrossEntropyLoss handles softmax

# ==========================================
# 3. TRAINING FLOW
# ==========================================
def train_and_dump():
    print("🚀 BẮT ĐẦU HUẤN LUYỆN...")
    
    # --- STEP 1: LOAD & VECTORIZE ---
    texts, labels = load_data()
    
    # Mapping Label Text -> ID
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    # Lọc chỉ lấy các label hợp lệ
    valid_indices = [i for i, l in enumerate(labels) if l in label_map]
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    y_numeric = [label_map[l] for l in labels]

    print("⚙️ Vectorizing...")
    # Giới hạn max_features để tránh OOM khi train Neural Net
    MAX_FEATURES = 5000 
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(texts)

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Lưu vectorizer (dùng chung cho cả 2 model)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_en.pkl"))

    # --- STEP 2: TRAIN SCIKIT-LEARN (BASELINE) ---
    print("\n--------- [SKLEARN LOGISTIC REGRESSION] ---------")
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"🎯 Accuracy (Logistic): {acc_lr:.2%}")
    
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "model_en.pkl"))
    
    # --- STEP 3: TRAIN PYTORCH (NEURAL NET) ---
    if HAS_TORCH:
        print("\n--------- [PYTORCH NEURAL NETWORK] ---------") 
        
        # Chuyển đổi Sparse Matrix -> Dense Tensor
        # Lưu ý: Convert toarray() có thể tốn RAM nếu dataset quá lớn
        X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Tạo DataLoader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Khởi tạo Model
        input_dim = X_train.shape[1]
        hidden_dim = 128
        output_dim = 3 # Neg, Neu, Pos
        
        model = SentimentNN(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training Loop
        epochs = 15 # Tăng số epoch nếu cần
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        # Evaluation
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            acc_torch = (predicted.numpy() == y_test).mean()
            print(f"🎯 Accuracy (PyTorch NN): {acc_torch:.2%}")

        # Save PyTorch Model
        torch_save_path = os.path.join(MODEL_DIR, "model_en_torch.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'classes': label_map # Lưu mapping để dùng lúc predict
        }, torch_save_path)
        print(f"✅ PyTorch model saved to: {torch_save_path}")

    print("\n✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")

if __name__ == "__main__":
    train_and_dump()