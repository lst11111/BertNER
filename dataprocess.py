import pickle as pkl
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# class getlabelslen():
#     def __init__(self):
#         self.label_index, self.index_label = pkl.load(open('./data/dataParams.pkl', "rb"))
#         self.num_labels = len(self.label_index)
#     def getlabelslen(self):
#         return self.num_labels
class MyDataset(Dataset):
    def __init__(self, datafile, with_labels=True):
        self.all_text, self.all_label = self.read_data(datafile)
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.label_index, self.index_label = self.build_label_index(self.all_label)
        self.max_len = 80

    def __getitem__(self, index):
        text = self.all_text[index]
        # Tokenize the text with padding and truncation
        encodings = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len+2,  # Considering [CLS] and [SEP] tokens
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Get input_ids and attention_mask
        text_id = encodings['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encodings['attention_mask'].squeeze(0)

        if self.with_labels:  # True if the dataset has labels
            label = self.all_label[index][:self.max_len]
            label_id = np.array([0] + [self.label_index.get(i, 1) for i in label] + [0] +
                                [0] * (self.max_len - len(label)))  # Pad the label
            label_id = torch.tensor(label_id, dtype=torch.int64)#82
            return text_id, attention_mask, label_id
        else:
            return text_id, attention_mask

    def __len__(self):
        return len(self.all_text)

    def read_data(self, file):
        # Read data from file
        all_data = open(file, "r", encoding="utf-8", errors='replace').read().split("\n")
        texts, labels = [], []
        text_one, label_one = [], []
        for data in all_data:
            if data != "":
                text, label = data.split()
                text_one.append(text)
                label_one.append(label)
            else:
                texts.append(text_one)
                labels.append(label_one)
                text_one, label_one = [], []
        return texts, labels

    def build_label_index(self, labels):
        label_to_index = {"PAD": 0, "UNK": 1}
        for label in labels:
            for i in label:
                if i not in label_to_index:
                    label_to_index[i] = len(label_to_index)
        index_to_label = list(label_to_index)
        pkl.dump([label_to_index, index_to_label], open('./data/dataParams.pkl', "wb"))
        return label_to_index, index_to_label


def create_data_loader():
    train_dataset = MyDataset(datafile='data/weiboNER.conll.train', with_labels=True)
    dev_dataset = MyDataset(datafile='data/weiboNER.conll.dev', with_labels=True)
    test_dataset = MyDataset(datafile='data/weiboNER.conll.test', with_labels=True)

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=5, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    index_label = train_dataset.index_label
    return train_dataloader, dev_dataloader, test_dataloader, index_label


if __name__ == "__main__":
    trainLoader, devLoader, testLoader,index_label = create_data_loader()
    for batch_text, batch_attention_mask, batch_label in trainLoader:

        print("Input IDs:\n", batch_text)
        print("Attention Mask:\n", batch_attention_mask)
        print("Label IDs:\n", batch_label)
        print("index_label",index_label)
        break