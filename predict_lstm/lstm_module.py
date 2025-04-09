import torch
import torch.nn as nn

# LSTM 모델 정의
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

# LSTM 추론 함수
def run_lstm_inference(feature_buffer):
    global action_result
    if not feature_buffer:
        return
    lstm_input = feature_buffer[:]
    lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = lstm_model(lstm_input_tensor)
        _, predicted_label = torch.max(output, 1)
        if predicted_label.item() == 0:
            action_result = "None"
        elif predicted_label.item() == 1:
            action_result = "Putting"
        elif predicted_label.item() == 2:
            action_result = "Taking"
    print("예측된 동작:", action_result)
    feature_buffer.clear()
