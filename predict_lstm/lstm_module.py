import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):   
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out

# 모델 로드
lstm_model = LSTMModel(input_size=20, hidden_size=64, num_layers=2, output_size=3)
lstm_model.load_state_dict(torch.load("./predict_lstm/lstm_model.pth"))
lstm_model.eval()

def run_lstm_inference(model, feature_buffer):
    if not feature_buffer:
        return "None"
    input_tensor = torch.tensor(feature_buffer, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label_map = {0: "None", 1: "Putting", 2: "Taking"}
        return label_map.get(pred.item(), "None")
