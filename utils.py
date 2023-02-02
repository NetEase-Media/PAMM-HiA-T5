import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5-base')

desc2index_reduced = {'CORPORATE': 0, 'ECONOMICS': 1, 'GOVERNMENT': 2, 'MARKETS': 3,
                      'STRATEGY': 4, 'LEGAL': 5, 'REGULATION': 6, 'SHARE LISTINGS': 7,
                      'PERFORMANCE': 8, 'INSOLVENCY': 9, 'FUNDING': 10, 'OWNERSHIP CHANGES': 11,
                      'PRODUCTION': 12, 'NEW PRODUCTS': 13, 'RESEARCH': 14, 'CAPACITY': 15,
                      'MARKETING': 16, 'ADVERTISING': 17, 'CONTRACTS': 18, 'MONOPOLIES': 19,
                      'MANAGEMENT': 20, 'LABOUR': 21, 'ECONOMIC PERFORMANCE': 22, 'MONETARY': 23, 'INFLATION': 24,
                      'CONSUMER FINANCE': 25, 'GOVERNMENT FINANCE': 26, 'OUTPUT': 27, 'EMPLOYMENT': 28,
                      'TRADE': 29, 'HOUSING STARTS': 30, 'LEADING INDICATORS': 31, 'EUROPEAN COMMUNITY': 32,
                      'CRIME, LAW ENFORCEMENT': 33, 'DEFENCE': 34, 'INTERNATIONAL RELATIONS': 35, 'DISASTERS AND ACCIDENTS': 36,
                      'ARTS, CULTURE, ENTERTAINMENT': 37, 'ENVIRONMENT AND NATURAL WORLD': 38, 'FASHION': 39, 'HEALTH': 40,
                      'LABOUR ISSUES': 41, 'MILLENNIUM ISSUES': 42, 'OBITUARIES': 43, 'HUMAN INTEREST': 44,
                      'DOMESTIC POLITICS': 45, 'BIOGRAPHIES, PERSONALITIES, PEOPLE': 46, 'RELIGION': 47, 'SCIENCE AND TECHNOLOGY': 48,
                      'SPORTS': 49, 'TRAVEL AND TOURISM': 50, 'WAR, CIVIL WAR': 51, 'ELECTIONS': 52,
                      'WEATHER': 53, 'WELFARE, SOCIAL SERVICES': 54, 'EQUITY MARKETS': 55, 'BOND MARKETS': 56,
                      'MONEY MARKETS': 57, 'COMMODITY MARKETS': 58, 'ACCOUNTS': 59, 'COMMENT': 60,
                      'SHARE CAPITAL': 61, 'BONDS': 62, 'LOANS': 63, 'CREDIT RATINGS': 64,
                      'MERGERS': 65, 'ASSET TRANSFERS': 66, 'PRIVATISATIONS': 67, 'DOMESTIC MARKETS': 68,
                      'EXTERNAL MARKETS': 69, 'MARKET SHARE': 70, 'DEFENCE CONTRACTS': 71, 'MANAGEMENT MOVES': 72,
                      'MONEY SUPPLY': 73, 'CONSUMER PRICES': 74, 'WHOLESALE PRICES': 75, 'PERSONAL INCOME': 76,
                      'CONSUMER CREDIT': 77, 'RETAIL SALES': 78, 'EXPENDITURE': 79, 'GOVERNMENT BORROWING': 80,
                      'INDUSTRIAL PRODUCTION': 81, 'CAPACITY UTILIZATION': 82, 'INVENTORIES': 83, 'UNEMPLOYMENT': 84,
                      'BALANCE OF PAYMENTS': 85, 'MERCHANDISE TRADE': 86, 'RESERVES': 87, 'EC INTERNAL MARKET': 88,
                      'EC CORPORATE POLICY': 89, 'EC AGRICULTURE POLICY': 90, 'EC MONETARY': 91, 'EC INSTITUTIONS': 92,
                      'EC ENVIRONMENT ISSUES': 93, 'EC COMPETITION': 94, 'EC EXTERNAL RELATIONS': 95, 'EC GENERAL': 96,
                      'INTERBANK MARKETS': 97, 'FOREX MARKETS': 98, 'SOFT COMMODITIES': 99, 'METALS TRADING': 100,
                      'ENERGY MARKETS': 101, 'ANNUAL RESULTS': 102}

desc2label_reduced = {'CORPORATE': 'CCAT', 'ECONOMICS': 'ECAT', 'GOVERNMENT': 'GCAT', 'MARKETS': 'MCAT', 'STRATEGY': 'C11', 'LEGAL': 'C12', 'REGULATION': 'C13', 'SHARE LISTINGS': 'C14', 'PERFORMANCE': 'C15', 'INSOLVENCY': 'C16', 'FUNDING': 'C17', 'OWNERSHIP CHANGES': 'C18', 'PRODUCTION': 'C21', 'NEW PRODUCTS': 'C22', 'RESEARCH': 'C23', 'CAPACITY': 'C24', 'MARKETING': 'C31', 'ADVERTISING': 'C32', 'CONTRACTS': 'C33', 'MONOPOLIES': 'C34', 'MANAGEMENT': 'C41', 'LABOUR': 'C42', 'ECONOMIC PERFORMANCE': 'E11', 'MONETARY': 'E12', 'INFLATION': 'E13', 'CONSUMER FINANCE': 'E14', 'GOVERNMENT FINANCE': 'E21', 'OUTPUT': 'E31', 'EMPLOYMENT': 'E41', 'TRADE': 'E51', 'HOUSING STARTS': 'E61', 'LEADING INDICATORS': 'E71', 'EUROPEAN COMMUNITY': 'G15', 'CRIME, LAW ENFORCEMENT': 'GCRIM', 'DEFENCE': 'GDEF', 'INTERNATIONAL RELATIONS': 'GDIP', 'DISASTERS AND ACCIDENTS': 'GDIS', 'ARTS, CULTURE, ENTERTAINMENT': 'GENT', 'ENVIRONMENT AND NATURAL WORLD': 'GENV', 'FASHION': 'GFAS', 'HEALTH': 'GHEA', 'LABOUR ISSUES': 'GJOB', 'MILLENNIUM ISSUES': 'GMIL', 'OBITUARIES': 'GOBIT', 'HUMAN INTEREST': 'GODD', 'DOMESTIC POLITICS': 'GPOL', 'BIOGRAPHIES, PERSONALITIES, PEOPLE': 'GPRO', 'RELIGION': 'GREL', 'SCIENCE AND TECHNOLOGY': 'GSCI', 'SPORTS': 'GSPO', 'TRAVEL AND TOURISM': 'GTOUR', 'WAR, CIVIL WAR': 'GVIO', 'ELECTIONS': 'GVOTE', 'WEATHER': 'GWEA', 'WELFARE, SOCIAL SERVICES': 'GWELF', 'EQUITY MARKETS': 'M11', 'BOND MARKETS': 'M12', 'MONEY MARKETS': 'M13', 'COMMODITY MARKETS': 'M14', 'ACCOUNTS': 'C151', 'COMMENT': 'C152', 'SHARE CAPITAL': 'C171', 'BONDS': 'C172', 'LOANS': 'C173', 'CREDIT RATINGS': 'C174', 'MERGERS': 'C181', 'ASSET TRANSFERS': 'C182', 'PRIVATISATIONS': 'C183', 'DOMESTIC MARKETS': 'C311', 'EXTERNAL MARKETS': 'C312', 'MARKET SHARE': 'C313', 'DEFENCE CONTRACTS': 'C331', 'MANAGEMENT MOVES': 'C411', 'MONEY SUPPLY': 'E121', 'CONSUMER PRICES': 'E131', 'WHOLESALE PRICES': 'E132', 'PERSONAL INCOME': 'E141', 'CONSUMER CREDIT': 'E142', 'RETAIL SALES': 'E143', 'EXPENDITURE': 'E211', 'GOVERNMENT BORROWING': 'E212', 'INDUSTRIAL PRODUCTION': 'E311', 'CAPACITY UTILIZATION': 'E312', 'INVENTORIES': 'E313', 'UNEMPLOYMENT': 'E411', 'BALANCE OF PAYMENTS': 'E511', 'MERCHANDISE TRADE': 'E512', 'RESERVES': 'E513', 'EC INTERNAL MARKET': 'G151', 'EC CORPORATE POLICY': 'G152', 'EC AGRICULTURE POLICY': 'G153', 'EC MONETARY': 'G154', 'EC INSTITUTIONS': 'G155', 'EC ENVIRONMENT ISSUES': 'G156', 'EC COMPETITION': 'G157', 'EC EXTERNAL RELATIONS': 'G158', 'EC GENERAL': 'G159', 'INTERBANK MARKETS': 'M131', 'FOREX MARKETS': 'M132', 'SOFT COMMODITIES': 'M141', 'METALS TRADING': 'M142', 'ENERGY MARKETS': 'M143', 'ANNUAL RESULTS': 'C1511'}
label2desc_reduced = {'CCAT': 'CORPORATE', 'ECAT': 'ECONOMICS', 'GCAT': 'GOVERNMENT', 'MCAT': 'MARKETS', 'C11': 'STRATEGY', 'C12': 'LEGAL', 'C13': 'REGULATION', 'C14': 'SHARE LISTINGS', 'C15': 'PERFORMANCE', 'C16': 'INSOLVENCY', 'C17': 'FUNDING', 'C18': 'OWNERSHIP CHANGES', 'C21': 'PRODUCTION', 'C22': 'NEW PRODUCTS', 'C23': 'RESEARCH', 'C24': 'CAPACITY', 'C31': 'MARKETING', 'C32': 'ADVERTISING', 'C33': 'CONTRACTS', 'C34': 'MONOPOLIES', 'C41': 'MANAGEMENT', 'C42': 'LABOUR', 'E11': 'ECONOMIC PERFORMANCE', 'E12': 'MONETARY', 'E13': 'INFLATION', 'E14': 'CONSUMER FINANCE', 'E21': 'GOVERNMENT FINANCE', 'E31': 'OUTPUT', 'E41': 'EMPLOYMENT', 'E51': 'TRADE', 'E61': 'HOUSING STARTS', 'E71': 'LEADING INDICATORS', 'G15': 'EUROPEAN COMMUNITY', 'GCRIM': 'CRIME, LAW ENFORCEMENT', 'GDEF': 'DEFENCE', 'GDIP': 'INTERNATIONAL RELATIONS', 'GDIS': 'DISASTERS AND ACCIDENTS', 'GENT': 'ARTS, CULTURE, ENTERTAINMENT', 'GENV': 'ENVIRONMENT AND NATURAL WORLD', 'GFAS': 'FASHION', 'GHEA': 'HEALTH', 'GJOB': 'LABOUR ISSUES', 'GMIL': 'MILLENNIUM ISSUES', 'GOBIT': 'OBITUARIES', 'GODD': 'HUMAN INTEREST', 'GPOL': 'DOMESTIC POLITICS', 'GPRO': 'BIOGRAPHIES, PERSONALITIES, PEOPLE', 'GREL': 'RELIGION', 'GSCI': 'SCIENCE AND TECHNOLOGY', 'GSPO': 'SPORTS', 'GTOUR': 'TRAVEL AND TOURISM', 'GVIO': 'WAR, CIVIL WAR', 'GVOTE': 'ELECTIONS', 'GWEA': 'WEATHER', 'GWELF': 'WELFARE, SOCIAL SERVICES', 'M11': 'EQUITY MARKETS', 'M12': 'BOND MARKETS', 'M13': 'MONEY MARKETS', 'M14': 'COMMODITY MARKETS', 'C151': 'ACCOUNTS', 'C152': 'COMMENT', 'C171': 'SHARE CAPITAL', 'C172': 'BONDS', 'C173': 'LOANS', 'C174': 'CREDIT RATINGS', 'C181': 'MERGERS', 'C182': 'ASSET TRANSFERS', 'C183': 'PRIVATISATIONS', 'C311': 'DOMESTIC MARKETS', 'C312': 'EXTERNAL MARKETS', 'C313': 'MARKET SHARE', 'C331': 'DEFENCE CONTRACTS', 'C411': 'MANAGEMENT MOVES', 'E121': 'MONEY SUPPLY', 'E131': 'CONSUMER PRICES', 'E132': 'WHOLESALE PRICES', 'E141': 'PERSONAL INCOME', 'E142': 'CONSUMER CREDIT', 'E143': 'RETAIL SALES', 'E211': 'EXPENDITURE', 'E212': 'GOVERNMENT BORROWING', 'E311': 'INDUSTRIAL PRODUCTION', 'E312': 'CAPACITY UTILIZATION', 'E313': 'INVENTORIES', 'E411': 'UNEMPLOYMENT', 'E511': 'BALANCE OF PAYMENTS', 'E512': 'MERCHANDISE TRADE', 'E513': 'RESERVES', 'G151': 'EC INTERNAL MARKET', 'G152': 'EC CORPORATE POLICY', 'G153': 'EC AGRICULTURE POLICY', 'G154': 'EC MONETARY', 'G155': 'EC INSTITUTIONS', 'G156': 'EC ENVIRONMENT ISSUES', 'G157': 'EC COMPETITION', 'G158': 'EC EXTERNAL RELATIONS', 'G159': 'EC GENERAL', 'M131': 'INTERBANK MARKETS', 'M132': 'FOREX MARKETS', 'M141': 'SOFT COMMODITIES', 'M142': 'METALS TRADING', 'M143': 'ENERGY MARKETS', 'C1511': 'ANNUAL RESULTS'}

class Rcv1Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.path = os.path.join(data_dir, type_path + '.txt')

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.texts_original = []
        self.labels_original = []
        self.ids = []
        self.labels_bfs = []
        self.texts = []
        self.labels = []
        self.all_paths_mask = []

        self._build()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_original = self.texts_original[index]
        label_original = self.labels_original[index]
        doc_id = self.ids[index]
        label_bfs_original = self.labels_bfs[index]

        texts_tokens = self.texts[index]["input_ids"].squeeze()
        labels_tokens = self.labels[index]["input_ids"].squeeze()
        texts_mask = self.texts[index]["attention_mask"].squeeze()
        labels_mask = self.labels[index]["attention_mask"].squeeze()
        paths_mask = self.all_paths_mask[index]


        return {"texts_tokens": texts_tokens, "texts_mask": texts_mask, "labels_tokens": labels_tokens,
                "labels_mask": labels_mask, "text_original": text_original, "label_original": label_original, "paths_mask": paths_mask, "doc_id":doc_id, "label_bfs_original":label_bfs_original}

    def _build(self):
        with open(self.path, encoding='utf-8') as f:
            for index, l in enumerate(f):

                content = json.loads(l)
                id, label, label_bfs, title, headline, text = content.get('id'), content.get('topics_desc_flat_reduced'), content.get('topics_hie'), content.get('title'), content.get('headline'), content.get('plain_text')

                text_ = '{} {}'.format(headline, text)
                text_ = text_ + ' </s>'
                label = label + " </s>"
                tokenized_texts = self.tokenizer.batch_encode_plus(
                    [text_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                )
                tokenized_labels = self.tokenizer.batch_encode_plus(
                    [label], max_length=90, pad_to_max_length=True, return_tensors="pt"
                )
                self.texts_original.append(text)
                self.labels_original.append(label[:-5])
                self.ids.append('{}'.format(id))
                self.labels_bfs.append('{}'.format(label_bfs))
                self.texts.append(tokenized_texts)
                self.labels.append(tokenized_labels)

                label_mask = tokenized_labels["attention_mask"].squeeze()
                label_list, label_tokens, label_range, label2range, ancestor_label2 = self.get_label_information(label)

                d_mask = []
                row = [0] * len(label_mask)
                row[0]=1
                d_mask.append(row)

                for i in range(label_mask.shape[0]-1):
                    row = [0] * (len(label_mask)-1)
                    for k, range_ in enumerate(label_range):
                        if i >= range_[0] and i < range_[1]:
                            position_one = [[range_[0], i + 1]]
                            if len(ancestor_label2[k]) != 0:
                                ancestor_range = [label2range[m] for m in ancestor_label2[k]]
                                position_one.extend(ancestor_range)
                            for position in position_one:
                                for l in range(position[0], position[1]):
                                    row[l] = 1
                            break
                        else:
                            continue
                    if i==len(label_tokens)-2:
                        row=[1]*(len(label_tokens)-1)+[0]*(label_mask.shape[0]-len(label_tokens))
                    if i<len(label_tokens)-1:
                        row.insert(0,1)
                    else:
                        row=[1]*(len(label_tokens))+[0]*(label_mask.shape[0]-len(label_tokens))
                    d_mask.append(row)
                self.all_paths_mask.append(torch.tensor(d_mask))


    def get_ancestor(self, node):
        if node == 'CCAT' or node == 'ECAT' or node == 'GCAT' or node == 'MCAT':
            return []
        if node in ['GCRIM', 'GDEF', 'GDIP', 'GDIS', 'GENT', 'GENV', 'GFAS', 'GHEA', 'GJOB', 'GMIL', 'GOBIT', 'GODD',
                    'GPOL', 'GPRO', 'GREL', 'GSCI', 'GSPO', 'GTOUR', 'GVIO', 'GVOTE', 'GWEA', 'GWELF']:
            return ['GCAT']
        if node == 'C1511':
            return ['CCAT', 'C15', 'C151']

        if len(node) == 3:
            if node[0] == 'C':
                return ['CCAT']
            elif node[0] == 'E':
                return ['ECAT']
            elif node[0] == 'G':
                return ['GCAT']
            elif node[0] == 'M':
                return ['MCAT']
        elif len(node) == 4:
            if node[0] == 'C':
                return ['CCAT', node[:-1]]
            elif node[0] == 'E':
                return ['ECAT', node[:-1]]
            elif node[0] == 'G':
                return ['GCAT', node[:-1]]
            elif node[0] == 'M':
                return ['MCAT', node[:-1]]

    def get_label_information(self, label):
        label_tokens = tokenizer.encode(label)
        label_list = label[:-5].replace('/', '_').split('_')
        if len(label_list) == 1:
            label_range = [[0, len(label_tokens)]]
            label2range = {label[:-5]: [0, len(label_tokens)]}
            ancestor_label = [[]]
            ancestor_label2 = [[]]
            return label_list, label_tokens, label_range, label2range, ancestor_label2

        split_index = [i for i,j in enumerate(label_tokens) if (j == 834 or j == 87)]
        label_range = [[0, split_index[0]]]
        for i in range(1, len(split_index)):
            label_range.append([split_index[i-1], split_index[i]])
        label_range.append([split_index[-1], len(label_tokens)-1])
        label2range = {}
        for i, j in enumerate(label_list):
            label2range[j] = label_range[i]

        ancestor_label = [self.get_ancestor(desc2label_reduced[i]) for i in label_list]
        ancestor_label2 = [[label2desc_reduced[j] for j in i] for i in ancestor_label]
        return label_list, label_tokens, label_range, label2range, ancestor_label2

def get_label_onehot(outputs):
    outputs_onehot = []
    truncated = {}
    for output in outputs:
        output_onehot = [0] * 103
        desc_split = output.split('_')
        for desc in desc_split:
            if desc in desc2index_reduced:
                output_onehot[desc2index_reduced[desc]] = 1
            else:
                if desc not in truncated:
                    truncated[desc] = 1
                else:
                    truncated[desc] += 1
        outputs_onehot.append(output_onehot)
    outputs_onehot = np.array(outputs_onehot)
    if truncated:
        print('Abnormal truncation:')
        print(truncated)
    return outputs_onehot

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from transformers import T5Tokenizer
    data_dir = "./data/rcv1_data"
    max_seq_length = 300
    tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5-base')
    train_dataset = Rcv1Dataset(tokenizer=tokenizer, data_dir=data_dir, type_path="rcv1_train",
                                max_len=max_seq_length)
    print(train_dataset[0])
    print(train_dataset[1])
