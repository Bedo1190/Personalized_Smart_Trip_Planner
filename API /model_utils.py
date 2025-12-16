import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

# 1. Temel GNN Sınıfı (HeteroGNN bunu kullanıyor)
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

# 2. HeteroGNN Sınıfı (Ana Modelin)
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.gnn = GNN(hidden_channels, out_channels)
        # Notebook'ta aggr='sum' kullanılmış, burası aynı kalmalı
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)

# 3. Modeli Yükleme Fonksiyonu
def load_model(path, metadata, hidden_channels=32, out_channels=16):
    # Notebook'taki eğitim parametrelerine göre default değerleri ayarladım (32, 16)
    
    # Modeli başlat (metadata vererek)
    model = HeteroGNN(metadata, hidden_channels=hidden_channels, out_channels=out_channels)
    
    # Ağırlıkları yükle
    # map_location='cpu' ile GPU'da eğitilmiş olsa bile CPU'da çalışmasını sağlıyoruz
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    # Modeli değerlendirme moduna al (Dropout vb. kapatır)
    model.eval()
    
    return model