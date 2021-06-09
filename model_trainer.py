import os
import copy
import torch
import numpy as np
from utils.earlystopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score
import statistics as stat
import tensorflow as tf
import pandas as pd
import math

'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = BrainDataset('../output.csv')
model = MyModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
batch_size = 16
trainer = ModelTrainer(model, dataset, DEVICE)
result = trainer.train(optimizer, 
					   criterion, 
					   batch_size=batch_size, 
					   epochs=epochs,
					   kfold=10,
					   iteration=5)
'''

def findTpTnFpFn(df, label):
	tp= df.loc[label,label]
	fp = df[label].sum() - tp
	fn = df.loc[label].sum() - tp

	df = df.drop(label, axis=0)
	df = df.drop(label, axis=1)

	tn = df.sum().sum()

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	specificity = tn / (tn + fp)

	return precision, recall, specificity

def delNan(arr):
	temp = []

	for i in range(len(arr)):
		if(math.isnan(arr[i])):
			continue
		else:
			temp.append(arr[i])
	
	return temp

class ModelTrainer:
	def __init__(self, model, dataset, DEVICE=None):
		if (dataset.data is None) or (dataset.label is None):
			raise ValueError("Dataset should have 'data' and 'label' variable with numpy.ndarray type")

		self.model = model
		self.reset_state = copy.deepcopy(model.state_dict())
		self.dataset = dataset
		if DEVICE is None:
			DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.DEVICE = DEVICE

	def train(self, optimizer, criterion, batch_size=1, epochs=1,
			  kfold=2, iteration=1, shuffle=True, random_state=None,
			  filepath=None, patience=7):
		best_state = []
		best_accuracy = 0.
		
		_kfold = KFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
		_data = self.dataset.data.numpy() if isinstance(self.dataset.data, torch.Tensor) else self.dataset.data
		_label = self.dataset.label

		minimum_early_stopping_epochs = 10		
		result = np.zeros((iteration, kfold), dtype=np.float)

		labels = [0, 1, 2]
		df_sum = pd.DataFrame(data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], columns = labels, index = labels)
		
		prec0_result = []
		rec0_result = []
		spec0_result = []

		prec1_result = []
		rec1_result = []
		spec1_result = []

		prec2_result = []
		rec2_result = []
		spec2_result = []

		for iter_index in range(iteration):
			for fold_index, (train_idx, test_idx) in enumerate(_kfold.split(_data)):
				print("=" * 12)
				print("Iter {} Fold {}".format(iter_index, fold_index))
				print("=" * 12)
				_model = self.model
				_model.load_state_dict(self.reset_state)
				x_train_fold = torch.from_numpy(_data[train_idx]).float()
				x_test_fold = torch.from_numpy(_data[test_idx]).float()
				y_train_fold = torch.from_numpy(_label[train_idx])
				y_test_fold = torch.from_numpy(_label[test_idx])

				train_data = TensorDataset(x_train_fold, y_train_fold)
				test_data = TensorDataset(x_test_fold, y_test_fold)

				train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
				test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
				
				early_stopping = EarlyStopping(patience=patience)
				
				prec0_arr = []
				rec0_arr = []
				spec0_arr = []

				prec1_arr = []
				rec1_arr = []
				spec1_arr = []

				prec2_arr = []
				rec2_arr = []
				spec2_arr = []
				
				for epoch in range(epochs):
					_model.train()
					for index, (data, label) in enumerate(train_loader):
						data, label = data.to(self.DEVICE), label.to(self.DEVICE)

						optimizer.zero_grad()
						output = _model(data)
						loss = criterion(output, label)
						loss.backward()
						optimizer.step()
						
						print("Epoch{} Training {:5.2f}% | Loss: {:.4f}".format(
							   epoch,
							   (index + 1) * batch_size / len(train_loader.dataset) * 100.,
							   loss.item()), end='\r')
					
					#print(_model.output_layer.weight.grad)
					_model.eval()
					test_loss = 0.
					correct = 0
					with torch.no_grad():
						for index, (data, label) in enumerate(test_loader):
							data, label = data.to(self.DEVICE), label.to(self.DEVICE)
							output = _model(data)
							loss = criterion(output, label)
						
							test_loss += loss.item()
							# Loss history?
							pred = output.data.max(1, keepdim=True)[1]
						
							pred_copy = pred.data.view_as(label)
							test_y = pred_copy.tolist()
							result_rf = label.tolist()
							cf_matrix = confusion_matrix(test_y, result_rf)

							df = pd.DataFrame(
									data=confusion_matrix(test_y, result_rf, labels=labels),
									columns=labels,
									index=labels
									)
							
							df_sum = df_sum.add(df)


							correct += pred.eq(label.data.view_as(pred)).cpu().sum()
							print("Testing... {:5.2f}%".format(
								   (index + 1) * batch_size / len(test_loader.dataset)), end='\r')
						
					
					prec_0, rec_0, spec_0 = findTpTnFpFn(df_sum, 0)
					prec_1, rec_1, spec_1 = findTpTnFpFn(df_sum, 1)
					prec_2, rec_2, spec_2 = findTpTnFpFn(df_sum, 2)				

					prec0_arr.append(prec_0)
					rec0_arr.append(rec_0)
					spec0_arr.append(spec_0)

					prec1_arr.append(prec_1)
					rec1_arr.append(rec_1)
					spec1_arr.append(spec_1)
					
					prec2_arr.append(prec_2)
					rec2_arr.append(rec_2)
					spec2_arr.append(spec_2)

					# 초기화)
					df_sum = pd.DataFrame(data = [[0,0,0],[0,0,0],[0,0,0]], columns = labels, index = labels)
					
					test_loss /= len(test_loader.dataset)
					accuracy = correct / float(len(test_loader.dataset))
					result[iter_index, fold_index] = accuracy
					print("Epoch{} Test Result: loss {:.4f} | accuracy {:.5f}({}/{})".format(
						   epoch, test_loss, accuracy, correct, len(test_loader.dataset)))
				
					if filepath is not None:
						if not os.path.isdir(filepath):
							os.mkdir(filepath)
						torch.save(_model.state_dict(), os.path.join(filepath, f"model{iter_index}_{fold_index}_" + datetime.datetime.now().strftime("%m%d_%H:%M:%S")))
					
					if epoch >= minimum_early_stopping_epochs:
						early_stopping(test_loss)
					if early_stopping.early_stop:
						print("Early stopping")
						break		
				
				prec0_arr = delNan(prec0_arr)
				rec0_arr = delNan(rec0_arr)
				spec0_arr = delNan(spec0_arr)

				prec1_arr = delNan(prec1_arr)
				rec1_arr = delNan(rec1_arr)
				spec1_arr = delNan(spec1_arr)

				prec2_arr = delNan(prec2_arr)
				rec2_arr = delNan(rec2_arr)
				spec2_arr = delNan(spec2_arr)

				prec0_result.append(stat.mean(prec0_arr))
				rec0_result.append(stat.mean(rec0_arr))
				spec0_result.append(stat.mean(spec0_arr))

				prec1_result.append(stat.mean(prec1_arr))
				rec1_result.append(stat.mean(rec1_arr))
				spec1_result.append(stat.mean(spec1_arr))

				prec2_result.append(stat.mean(prec2_arr))
				rec2_result.append(stat.mean(rec2_arr))
				spec2_result.append(stat.mean(spec2_arr))
				"""
				print('prec0_result = ', prec0_result)
				print('rec0_result = ', rec0_result)
				print('spec0_result = ', spec0_result)
				print()
				print('prec1_result = ', prec1_result)
				print('rec1_result = ', rec1_result)
				print('spec1_result = ', spec1_result)
				print()
				print('prec2_result = ', prec2_result)
				print('rec2_result = ', rec2_result)
				print('spec2_result = ', spec2_result)
				"""
			iter_accuracy = result[iter_index].mean()
			if (iter_accuracy > best_accuracy):
				best_state = _model.state_dict()
				best_accuracy = iter_accuracy
			print('=' * 12)
			print("Iteration {} complete with {:5.2f}% average accuracy".format(
				   iter_index, iter_accuracy * 100.))
			print('=' * 12)
		
		prec0_final = stat.mean(prec0_result)
		rec0_final = stat.mean(rec0_result)
		spec0_final = stat.mean(spec0_result)

		prec1_final = stat.mean(prec1_result)
		rec1_final = stat.mean(rec1_result)
		spec1_final = stat.mean(spec1_result)

		prec2_final = stat.mean(prec2_result)
		rec2_final = stat.mean(rec2_result)
		spec2_final = stat.mean(spec2_result)
		
		print('prec0_arr = ', prec0_result)
		print('rec0_arr = ', rec0_result)
		print('spec0_arr = ', spec0_result)
		print()
		print('prec1_arr = ', prec1_result)
		print('rec1_arr = ', rec1_result)
		print('spec1_arr = ', spec1_result)
		print()
		print('prec2_arr = ', prec2_result)
		print('rec2_arr = ', rec2_result)
		print('spec2_arr = ', spec2_result)
		print('================================================================================')
		print('prec0 = ', prec0_final)
		print('rec0 = ', rec0_final)
		print('spec0 = ', spec0_final)
		print()
		print('prec1 = ', prec1_final)
		print('rec1 = ', rec1_final)
		print('spec1 = ', spec1_final)
		print()
		print('prec2 = ', prec2_final)
		print('rec2 = ', rec2_final)
		print('spec2 = ', spec2_final)
		print()
		print("Training complete with {:5.2f}%".format(result.mean()))
		self.model.load_state_dict(best_state)
		return result
