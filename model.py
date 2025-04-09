import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# bu sınıf pytorch kullanılarak oluşturulmuş bir yapay sinir ağı modeli
#iki doğrusal katmandan oluşur ve durumlara karşılık gelen Q-değerlerini tahmin eder
#bu tahminler, ajan (agent) tarafından en iyi eylemi seçmek için kullanılır.
class Linear_QNet(nn.Module):
    #modelin katmanlarını tanımlar
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)#girdi gizli katmana bağlayan doğrusal katman
        self.linear2 = nn.Linear(hidden_size, output_size)#gizli katmandan çıkış katmanına bağlayan doğrusal katman

    #modelin ileri yayılımını yani forward propagation'ını tanımlar
    def forward(self, x):
        x = F.relu(self.linear1(x))#relu aktivasyon fonksiyonu uygulandı
        x = self.linear2(x)
        return x#modelin tahmin ettiği q-değerlerini döndürür
    
    #modelin ağırlıklarını bir dosyaya kaydeder
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#linear_qnet sınıfını eğitmek için oluşturukdu
#modelin ağırlıklarını güncellemek için gerekli olan optimizasyon ve kayıp hesaplama işlemlerini gerçekleştirir
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    #parametrelerdeki verileri pytorch tensörlerine dönüştürür ve tek bir örnek varsa boyutlarını genişletir
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, ) #tuple

        #modelin tahmin ettiği Q-değerlerini ve hedef Q-değerlerini hesaplar
        pred = self.model(state)#modelin tahmin ettiği Q-değerleri

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]# oyun bitmişse hedef Q-değerini sadece rewarda eşitler
            #oyun bitmediyse hedef Q-değerini günceller
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        #modelin ağırlıklarını günceller
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()