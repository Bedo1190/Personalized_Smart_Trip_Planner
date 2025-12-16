from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
import pandas as pd
from model_utils import load_model

app = FastAPI(title="Kategori Bazlı Gezi Öneri API")

# Global değişkenler
model = None
data = None
places_df = None
metadata = None

# --- Veri Modelleri ---
class CategoryRequest(BaseModel):
    categories: list[str]  # Örn: ["Museums", "Landmarks"]
    top_k: int = 5

class PlaceRecommendation(BaseModel):
    place_name: str
    category: str
    score: float

# --- Başlangıç (Startup) ---
@app.on_event("startup")
async def load_artifacts():
    global model, data, places_df, metadata
    
    print("📂 Veriler yükleniyor...")
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
        
    data = artifacts['data']
    places_df = artifacts['places_df']
    metadata = artifacts['metadata']
    
    # Kategori sütunundaki boşlukları temizleyelim ve küçük harfe çevirelim (eşleşme kolaylığı için)
    # Eğer veri setinde kategori sütununun adı farklıysa burayı düzeltmelisin (örn: 'Category' vs 'category')
    if 'category' in places_df.columns:
        places_df['category_clean'] = places_df['category'].astype(str).str.strip()
    else:
        # Hata önleyici dummy kolon
        places_df['category_clean'] = "General"

    print("🧠 Model yükleniyor...")
    # Model boyutlarına dikkat (32, 16)
    model = load_model('gnn_model_weights.pth', metadata, hidden_channels=32, out_channels=16)
    print("✅ Sistem hazır!")

# --- Yardımcı Fonksiyon: Kategoriden Vektör Çıkarma ---
def get_recommendations_by_category(selected_categories, k=5):
    # 1. Modelden güncel embeddingleri al
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        place_embs = out['place'] # Tüm mekanların vektörleri (Shape: [Num_Places, 16])

    # 2. Seçilen kategorilere ait mekanların indekslerini bul
    # Büyük/küçük harf duyarlılığını kaldırmak için filtreleme yapıyoruz
    selected_indices = []
    
    for cat in selected_categories:
        # Kısmi eşleşme (contains) veya tam eşleşme yapabiliriz. Burada tam eşleşme kullanıyoruz.
        matches = places_df[places_df['category_clean'] == cat].index.tolist()
        selected_indices.extend(matches)
    
    selected_indices = list(set(selected_indices)) # Tekrarları kaldır

    if not selected_indices:
        return None  # Bu kategorilerde hiç mekan bulunamadı

    # 3. "Sanal Kullanıcı" Vektörü Oluştur
    # Seçilen mekanların vektörlerini alıp ortalamasını (mean) alıyoruz.
    target_embs = place_embs[selected_indices]
    interest_vector = torch.mean(target_embs, dim=0) # (Shape: [16])

    # 4. Tüm mekanlarla benzerliği hesapla (Matrix Multiplication / Dot Product)
    # interest_vector'ü [16] boyutundan [1, 16] yapıp çarpıyoruz
    scores = torch.matmul(place_embs, interest_vector.unsqueeze(1)).squeeze()

    # 5. En yüksek skorlu k mekanı bul
    top_k_scores, top_k_indices = torch.topk(scores, k + len(selected_indices)) 
    # Biraz fazla çekiyoruz çünkü input olarak verilenleri sonuçtan çıkarmak isteyebiliriz.

    recommendations = []
    added_count = 0
    
    for score, idx in zip(top_k_scores, top_k_indices):
        place_idx = idx.item()
        
        # İstersen input olarak verilen kategorideki yerleri de önerebilirsin
        # ya da "farklı ama alakalı" yerleri önermek için filtreleyebilirsin.
        # Şimdilik hepsini gösteriyoruz.
        
        place_info = places_df.iloc[place_idx]
        
        recommendations.append({
            "place_name": place_info['name'],
            "category": place_info.get('category', 'Unknown'),
            "score": float(score)
        })
        
        added_count += 1
        if added_count >= k:
            break
            
    return recommendations

# --- Endpoints ---

@app.post("/recommend_by_interest", response_model=list[PlaceRecommendation])
async def recommend(request: CategoryRequest):
    recs = get_recommendations_by_category(request.categories, request.top_k)
    
    if recs is None:
        raise HTTPException(status_code=404, detail="Seçilen kategorilerde mekan bulunamadı. Lütfen /categories endpointinden listeyi kontrol edin.")
        
    return recs

@app.get("/categories")
async def get_all_categories():
    """Sistemde mevcut olan kategorileri listeler."""
    if places_df is not None and 'category_clean' in places_df.columns:
        cats = places_df['category_clean'].unique().tolist()
        return {"available_categories": sorted(cats)}
    return {"error": "Categories not loaded"}