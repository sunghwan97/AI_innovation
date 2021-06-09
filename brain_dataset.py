import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

class BrainDataset(Dataset):
	_label_col = 'Group'
	_level_0 = ['lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'Left-Lateral-Ventricle', 'Left-Cerebellum-White-Matter',

         'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',

         '3rd-Ventricle', '4th-Ventricle', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area',

         'Left-VentralDC', 'Right-Lateral-Ventricle', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',

         'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',

         'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC']

	def __init__(self, file_path, expand_dim=False, level=0):
		super(BrainDataset, self).__init__()
		raw_data = pd.read_csv(file_path)

		# Label Categorization
		label = raw_data['Group']
		self.label = label.replace(['CN', 'MCI', 'AD'], [0, 1, 2]).astype(int).to_numpy()

		_level = self._level_0 if level == 0 else \
				 self._level_1 if level == 1 else \
				 self._level_2

		# Data Normalization (Z score)
		data = raw_data.loc[:, _level].to_numpy()
		scaler = StandardScaler().fit(data)
		self.data = scaler.transform(data)
		assert len(label) == len(data)

		if expand_dim:
			self.data = np.expand_dims(self.data, axis=1)

	def __getitem__(self, index):
		return self.data[index], self.label[index]
	
	def __len__(self):
		return len(self.data)

