import os
import json
import torch.utils.data as data
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


class PubMedQATaskDataset(data.Dataset):

    def __init__(self, name, all_folds=False, split='test'):
        self.name = name
        subset_str = 'pubmed_qa_labeled_fold{fold_id}'
        folds = [0] if not all_folds else list(range(10))

        bigbio_data = []
        source_data = []
        for fold_id in folds:
            bb_data = load_dataset(self.name, f'{subset_str.format(fold_id=fold_id)}_bigbio_qa', split=split)
            s_data = load_dataset(self.name, f'{subset_str.format(fold_id=fold_id)}_source', split=split)
            bigbio_data.append(bb_data)
            source_data.append(s_data)
        bigbio_data = concatenate_datasets(bigbio_data)
        source_data = concatenate_datasets(source_data)

        keys_to_keep = ['id', 'question', 'context', 'answer', 'LONG_ANSWER']
        data_elms = []
        for elm_idx in tqdm(range(len(bigbio_data)), desc='Preparing data'):
            data_elms.append({k: bigbio_data[elm_idx][k] for k in keys_to_keep[:4]})
            data_elms[-1].update({keys_to_keep[-1].lower(): source_data[elm_idx][keys_to_keep[-1]]})
        
        self.data = data_elms

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def mock_knowledge_base(self, output_dir, one_file_per_sample=False, samples_per_file=500, sep='\n', jsonl=False):
        """
        Write PubMed contexts to a text file, newline seperated
        """
        pubmed_kb_dir = os.path.join(output_dir, 'pubmed_doc')
        os.makedirs(pubmed_kb_dir, exist_ok=True)

        file_ext = 'jsonl' if jsonl else 'txt'

        if not one_file_per_sample:
            context_str = ''
            context_files = []
            for idx in range(len(self.data)):
                if (idx + 1) % samples_per_file == 0:
                    context_files.append(context_str.rstrip(sep))
                else:
                    if jsonl:
                        context_elm_str = json.dumps({'id': self.data[idx]["id"],'context': self.data[idx]["context"]})
                    else:
                        context_elm_str = self.data[idx]["context"]
                    context_str += f'{context_elm_str}{sep}'

            for file_idx in range(len(context_files)):
                filepath = os.path.join(pubmed_kb_dir, f'context{file_idx}.{file_ext}')
                with open(filepath, 'w') as f:
                    f.write(context_files[file_idx])
        
        else:
            assert not jsonl, "Does not support jsonl if one_file_per_sample is True"
            for idx in range(len(self.data)):
                filepath = os.path.join(pubmed_kb_dir, f'{self.data[idx]["id"]}.{file_ext}')
                with open(filepath, 'w') as f:
                    f.write(self.data[idx]["context"])