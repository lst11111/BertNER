import pickle as pkl
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchcrf import CRF
import torch

# class ModelOutput:
#   def __init__(self, logits, loss=None):
#     self.logits = logits
#     self.loss = loss

class BertNer(nn.Module):
    def __init__(self, kernel_size=3, num_filters=256, model_type=None):
        super(BertNer, self).__init__()

        # 加载BERT预训练模型和配置
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_config = BertConfig.from_pretrained('bert-base-chinese')
        self.hidden_size = 768
        self.label_index, self.index_label = pkl.load(open('./data/dataParams.pkl', "rb"))
        self.num_labels = len(self.index_label)#17

        # 模型类型
        self.model_type = model_type

        # 定义不同模型类型的层
        if model_type == 'bert_cnn':
            self.conv = nn.Conv2d(1, num_filters, kernel_size=(kernel_size, self.hidden_size),
                                  padding=1,stride=1)  # 卷积核大小 (kernel_size, hidden_size)
            # 定义Dropout层
            self.dropout = nn.Dropout(0.5)
            # 定义全连接层
            self.fc = nn.Linear(num_filters, self.num_labels)  # 将卷积后的特征映射到num_labels维度

        elif model_type == 'bert_bilstm':
            self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, bidirectional=True, batch_first=True)
            #self.linear = nn.Linear(768 * 2, self.num_labels)  # 2 * hidden_size for bidirectional LSTM
            self.fc = nn.Linear(2 * 768, self.num_labels)  # 2 * hidden_size for bidirectional LSTM
        elif model_type == 'bert_transformer':
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
            self.fc = nn.Linear(768, self.num_labels)

        # 如果使用CRF层
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, model_type=None):
        # BERT输出
        loss = None
        logits = None
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        if model_type == 'bert_cnn':
            # 对BERT输出进行卷积
            sequence_output = sequence_output.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
            cnn_output = torch.relu(self.conv(sequence_output))  # [batch_size, num_filters, seq_len, 1]

            # 池化操作，保留每个token的卷积特征
            # 使用自适应平均池化将最后一维压缩为1

            # 使用池化操作将最后一维从3压缩为1
            pooled_output = nn.functional.adaptive_avg_pool2d(cnn_output, (cnn_output.size(2), 1))  # [batch_size, num_filters, seq_len, 1]
            pooled_output = pooled_output.squeeze(3)  # [batch_size, num_filters, seq_len]
            pooled_output = pooled_output.permute(0, 2, 1)
            # Dropout
            pooled_output = self.dropout(pooled_output)  # [batch_size, seq_len, num_filters]

            # 全连接层映射到num_labels
            logits = self.fc(pooled_output)  # [batch_size, seq_len, num_labels]




        elif model_type == 'bert_bilstm':
            # 使用BERT输出 + BiLSTM
            lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, 2 * lstm_hidden_size]
            #seq_out = self.linear(lstm_output)  # 4,512,5
            logits = self.fc(lstm_output)  # [batch_size, seq_len, num_labels]

        elif model_type == 'bert_transformer':
            # 使用BERT输出 + Transformer Encoder
            sequence_output = sequence_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
            transformer_output = self.transformer_encoder(sequence_output)  # [seq_len, batch_size, hidden_size]
            transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]
            logits = self.fc(transformer_output)  # [batch_size, seq_len, num_labels]

        # CRF层的解码
        if labels is not None:
            crf_loss = self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            loss = -crf_loss
        outputs = {
            'logits': logits ,
            'loss': loss,
        }
        return outputs
