import os
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from predict import get_data
import torch.nn as nn
import torch


app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"], supports_credentials=True)
UPLOAD_FOLDER = './save_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class Net(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1, layers=2, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.function = torch.sigmoid

    def forward(self, X):
        X, _ = self.lstm(X, None)
        X = X[:, -1, :]
        X = self.dropout(X)
        X = self.linear(X)
        X = self.function(X)
        return X


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'csv.csv'))
    return {'message': '上传成功'}, 200


@app.route('/getdata', methods=['GET'])
def get_stockdata():
    try:
        path = './save_files/csv.csv'
        if not os.path.isfile(path):
            return make_response(jsonify({"error": "No file"}), 404)
        data = get_data(path)
    finally:
        os.remove('./save_files/csv.csv')
    return jsonify(data)


if __name__ == '__main__':
    # 运行app
    app.run(debug=True)
