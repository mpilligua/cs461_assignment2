from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch 
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataset_path, split='train'):
        super().__init__()
        self.dataset_path = dataset_path
        self.data = torch.load(f'{self.dataset_path}/roi_{split}.pt', weights_only=False)
        self.embeddings, self.labels, self.patient_ids = self.data 
        self.unique_patient_ids = list(set(self.patient_ids.tolist()))
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(f'{self.dataset_path}/roi_classes.npy', allow_pickle=True)
        self.labels = self.label_encoder.transform(self.labels)
        self._map_patient_crops()

    def _map_patient_crops(self):
        patient_to_indices = defaultdict(list)
        for idx, patient_id in enumerate(self.patient_ids):
            patient_to_indices[patient_id].append(idx)
        
        self.patient_to_indices = dict(patient_to_indices)

    def __len__(self):
        return len(self.unique_patient_ids)

    def __getitem__(self, idx):
        patient_id =self.unique_patient_ids[idx]
        idxs = self.patient_to_indices[patient_id]
        embeddings = self.embeddings[idxs]
        label = self.labels[idxs[0]]
        return embeddings, label 
