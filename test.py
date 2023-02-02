from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from transformers import T5Tokenizer

from utils import Rcv1Dataset
from utils import get_label_onehot
from utils import desc2index_reduced
from train_dmask import T5FineTuner

tokenizer = T5Tokenizer.from_pretrained('./pretrain_model/t5-base')

print('Model loading*****************************************************************************')
ckpt_path = './ckpt_rcv1/...'
print('Loaded ckpt_path:\t{}'.format(ckpt_path))
model = T5FineTuner.load_from_checkpoint(ckpt_path)
model = model.cuda()

print('Read test data to memory***********************************************************************')
dataset = Rcv1Dataset(tokenizer, 'rcv1_data', 'rcv1_test', 300)
loader = DataLoader(dataset, batch_size=200, num_workers=30)
model.model.eval()

texts = []
outputs = []
targets = []
doc_ids = []
labels_bfs = []

for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['texts_tokens'].cuda(),
                                attention_mask=batch['texts_mask'].cuda(),
                                max_length=90)
    dec = [tokenizer.decode(ids) for ids in outs]

    texts.extend(batch['text_original'])
    outputs.extend(dec)
    targets.extend(batch['label_original'])
    doc_ids.extend(batch['doc_id'])
    labels_bfs.extend(batch['label_bfs_original'])

badcase_path = 'badcase_rcv1_dmask.txt'
print('Write badcase to file:{}'.format(badcase_path))
with open(badcase_path, 'w') as file_out:
    for i in range(len(outputs)):
        content = {}
        truncated_label = outputs[i].split('_')[-1]
        string = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(doc_ids[i], targets[i], outputs[i], labels_bfs[i], targets[i]==outputs[i], truncated_label in desc2index_reduced, truncated_label, texts[i].replace('\n', ' '))
        file_out.write(string)
print('True label:')
labels_true = get_label_onehot(targets)
print('Predicted label:')
labels_pred = get_label_onehot(outputs)


print('Calculate related metrics********************************************************************')
micro_f1 = f1_score(labels_true, labels_pred, average='micro')
macro_f1 = f1_score(labels_true, labels_pred, average='macro')
weighted_f1 = f1_score(labels_true, labels_pred, average = 'weighted')
print('micro_f1:{}\tmacro_f1{}\tweighted_f1{}'.format(micro_f1, macro_f1, weighted_f1))