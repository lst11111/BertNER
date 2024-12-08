import torch
import torch.nn as nn
from dataprocess import create_data_loader
import time
import logging
from tqdm import tqdm
from model import BertNer
import numpy as np
import datetime
import os
from sklearn.metrics import f1_score  # 用于计算F1值
from spacy.training import biluo_tags_to_offsets
import spacy
from seqeval.metrics import classification_report
# 配置日志，输出到log.txt文件中
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log/train_log.txt"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 模型类型
model_type = "bert_transformer"  # 如果使用不同模型，请确保 BertNer 类能够处理 model_type
#nlp =spacy.load("D:\sapcy_thing\zh_core_web_sm-3.7.0\zh_core_web_sm\zh_core_web_sm-3.7.0")

from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics.sequence_labeling import get_entities

def compute_entity_f1(true_entities, pred_entities):
    # 初始化计数器
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 遍历真实实体并与预测实体进行比较
    for true_entity in true_entities:
        if true_entity in pred_entities:
            true_positives += 1  # 真正例
        else:
            false_negatives += 1  # 假负例

    # 对于预测的实体，检查是否为假正例
    for pred_entity in pred_entities:
        if pred_entity not in true_entities:
            false_positives += 1  # 假正例

    # 计算 Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1



def compute_f1(batch_masks, batch_labels, batch_prediction, index_tag):
    all_prediction, all_labels = [], []

    # 遍历所有batch中的句子
    for i in range(len(batch_masks)):
        max_length = batch_masks[i].shape[0]  # 获取每个句子的最大长度

        # 计算有效token数（即mask值为1的部分）
        length = (batch_masks[i].cpu().numpy() == 1).sum()  # 计算整个句子的有效长度

        if length > 0:  # 如果句子有有效的token
            _label = batch_labels[i][1:length-1].cpu().numpy().tolist()  # 截取有效标签
            _predict = batch_prediction[i][1:length-1].cpu().numpy().tolist()  # 截取有效预测

            # 如果标签和预测长度一致，才进行计算
            if len(_label) == len(_predict):
                a,b=[],[]
                # label -> tag
                label_tag = [index_tag[label] for label in _label]
                predict_tag = [index_tag[pred] for pred in _predict]

                # 去掉 'PAD' 标签
                label_tag = [tag for tag in label_tag if tag != 'PAD']
                predict_tag = [tag for tag in predict_tag if tag != 'PAD']

                label_entities = get_entities(label_tag)
                predict_entities = get_entities( predict_tag)
                all_labels.append(label_entities)
                all_prediction.append(predict_entities)

    # 计算 F1 值，使用实体级的匹配，使用 weighted 平均
    f1 = compute_entity_f1(all_labels, all_prediction)
    return f1


# 训练函数
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    f1_scores = []  # 用于存储每个 batch 的 F1
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch", ncols=100)

    for batch in pbar:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)  # [batch_size, seq_len]

        optimizer.zero_grad()  # 清除模型参数的梯度

        # 获取模型输出 (logits)
        modelout = model(input_ids, attention_mask,labels,model_type = model_type)  # [batch_size, seq_len, num_classes]
        loss = modelout['loss'] # 假设有 compute_loss 方法来计算损失

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()

        # 获取每个 token 的预测类别
        #predictions = torch.argmax(modelout['logits'], dim=2)  # [batch_size, seq_len]
        #f1 = compute_f1(attention_mask, labels, predictions, label_map,nlp=nlp)
        #f1_scores.append(f1)  # 存储每个 batch 的 F1

        pbar.set_postfix(loss=loss.item())  # 更新进度条

    avg_loss = total_loss / len(train_loader)
    #avg_f1 = np.mean(f1_scores)  # 计算整个 epoch 的平均F1
    logger.info(f"Train Loss (Epoch {epoch + 1}): {avg_loss:.4f}")
    #logger.info(f"Train F1 (Epoch {epoch + 1}):{avg_f1:.4f}")
    return avg_loss


# 验证函数
def dev(model, dev_loader, device, label_map):
    model.eval()  # 切换为评估模式，关闭梯度计算
    total_loss = 0
    f1_scores = []

    pbar = tqdm(dev_loader, desc="Evaluating", unit="batch", ncols=100)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)  # [batch_size, seq_len]

            # 获取模型输出 (logits)
            modelout = model(input_ids, attention_mask,labels,model_type = model_type)
            #loss = modelout['loss']#
            #total_loss += loss.item()
            logits = modelout['logits']
            probabilities = torch.softmax(logits, dim=-1)
            # 获取每个 token 的预测类别
            predictions = torch.argmax(probabilities, dim=2)  # [batch_size, seq_len]
            f1 = compute_f1(attention_mask, labels, predictions, label_map)
            if f1 is not None:
                f1_scores.append(f1)

            pbar.set_postfix( f1=np.mean(f1_scores))  # 更新进度条

    #avg_loss = total_loss / len(dev_loader)
    avg_f1 = np.mean(f1_scores)
    #logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation F1: {avg_f1:.4f}")

    return  avg_f1


# 测试函数
def test(model, test_loader, device, label_map):
    model.eval()  # 切换为评估模式，关闭梯度计算
    total_loss = 0
    f1_scores = []

    pbar = tqdm(test_loader, desc="Testing", unit="batch", ncols=100)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)  # [batch_size, seq_len]

            # 获取模型输出 (logits)
            modelout = model(input_ids, attention_mask,labels,model_type = model_type)  # [batch_size, seq_len, num_classes]
            loss = modelout['loss']#
            total_loss += loss.item()

            # 获取每个 token 的预测类别
            predictions = torch.argmax(modelout['logits'], dim=2)  # [batch_size, seq_len]
            f1 = compute_f1(attention_mask, labels, predictions, label_map)
            f1_scores.append(f1)

            pbar.set_postfix(loss=loss.item(), f1=np.mean(f1_scores))  # 更新进度条

    avg_loss = total_loss / len(test_loader)
    avg_f1 = np.mean(f1_scores)

    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test F1: {avg_f1:.4f}")

    return avg_loss, avg_f1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BertNer(model_type=model_type).to(device)

    train_loader, dev_loader, test_loader, label_map = create_data_loader()  # 加载NER数据并返回label_map

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    epochs = 10
    best_val_f1 = 0  # 用于保存最优验证F1值
    best_model_state_dict = None  # 用于保存最优模型参数

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch, device)
        val_f1 = dev(model, dev_loader, device, label_map)

        # 保存最好的模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state_dict = model.state_dict()

        logger.info(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Validation F1: {val_f1:.4f}")

    # 最优模型保存
    torch.save(best_model_state_dict, "best_model.pth")
    logger.info("Best model saved!")

    # 加载最好的模型进行测试
    model.load_state_dict(best_model_state_dict)
    test_loss, test_f1 = test(model, test_loader, device, label_map)
    logger.info(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
