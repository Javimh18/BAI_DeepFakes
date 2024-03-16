import os
import torch

import numpy as np 
from tqdm import tqdm
from config import VAE_ALPHA
from sklearn.metrics import roc_curve, auc

import torch.nn as nn
import torch.nn.init as init 
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, device, path):

    list_loss_train = []
    list_loss_val = []

    list_acc_train = []
    list_acc_val = []    

    best_val_accuracy = 0

    model.to(device)
    print("Training on:", device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        val_accuracy = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())

                val_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs)
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / len(val_loader.dataset)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(path, 'best_val.pth'))

        list_loss_train.append(train_loss)
        list_loss_val.append(val_loss)

        list_acc_train.append(train_accuracy)
        list_acc_val.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
       # print(f"{correct_val, len(val_loader.dataset)}")
       # _, _, _ , _ = test_model(model, test_loader, criterion, device)

    return list_loss_train, list_acc_train, list_loss_val, list_acc_val

def validate_model(best_model, val_loader, criterion, device):

    list_outputs = []
    list_labels = []  
    
    best_model.to(device)

    best_model.eval()
    val_loss = 0
    correct_val = 0

    with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())

                val_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs)
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()

                for output, label in zip(outputs, labels):
                    list_outputs.append(output.cpu().squeeze().numpy())
                    list_labels.append(label.cpu().squeeze().numpy())

    accuracy_validation = correct_val/len(val_loader.dataset)
    average_loss = val_loss/len(val_loader.dataset)

    auc_best_val, _, _ = computeROC(list_labels, list_outputs, path=None)

    print(f'Best Validation => Loss: {average_loss:.4f}, Accuracy:{accuracy_validation:.4f}%, AUC:{auc_best_val:.4f}')
   # print(f"{correct_val, len(val_loader.dataset)}")
    return accuracy_validation, average_loss, auc_best_val


def test_model(best_model, test_loader, criterion, device):

    list_loss_test = []
    list_acc_test = [] 

    list_outputs = []
    list_labels = []  
    
    best_model.to(device)

    best_model.eval()
    test_loss = 0
    hits = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            test_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            hits += (predicted == labels.unsqueeze(1)).sum().item()
                        
            labels_numpy = labels.cpu().squeeze().numpy()

            for output, label in zip(outputs, labels):
                list_outputs.append(output.cpu().squeeze().numpy())
                list_labels.append(label.cpu().squeeze().numpy())

    accuracy = hits/len(test_loader.dataset)
    average_loss = test_loss/len(test_loader.dataset)

   # print(accuracy, average_loss)

    return average_loss, accuracy, list_outputs, list_labels

def initialize_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def initialize_weights_xavier_normal(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def computeROC(list_labels, list_outputs, path):
    fpr, tpr, thresholds = roc_curve(list_labels, list_outputs)
    roc_auc = auc(fpr, tpr)

    if path:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(path, 'ROC.png'))
        plt.close()    

    return roc_auc, fpr, tpr

































class VAETrainer:
    """
    This class is in charge of train the Variational Autoencoder
    """
    def __init__(self, 
                 model, 
                 optimizer,  
                 train_data_loader,
                 valid_data_loader,
                 epochs,
                 recon_loss_weight=VAE_ALPHA,
                 save_dir='./models/vae',
                 save_if_improves=True,
                 save_every = None,
                 patience=10,
                 type='vae'):
        
        # device initialization
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # model initialization
        self.model = model.to(self.device)
        self.type = type
        
        # other params initialization
        self.optimizer = optimizer
        self.recon_loss_weight = recon_loss_weight
        
        # epochs related information
        self.epochs = epochs
        self.start_epoch = 1
        self.checkpoint_dir = save_dir
        
        # check for existence of the path where the results and the model are going to be saved
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if save_every is not None and save_if_improves is True:
            print("save_every and save_if_improves are mutually exclusive parameters.\n\
                Check your configuration in the VAETrainer object.")
            exit()
        elif save_every is not None:
            self.save_every = save_every
            self.save_if_improves = False
        else: 
            self.save_every = None
            self.save_if_improves = save_if_improves
        self.patience = patience
        
        # data loader info
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        
        
    def train(self):
        """
        Logic of the whole training
        """
        epochs_w_o_improve = 0
        current_best_loss = np.inf
        epoch_best = self.start_epoch
        train_data = []
        val_data = []
        for epoch in range(self.start_epoch, self.epochs+1):
            print(f">>>>>>>>>>>>> EPOCH: {epoch}")
            results_for_epoch = self._train_epoch(epoch)
            train_epoch_data, val_epoch_data = results_for_epoch
            
            train_data.append(train_epoch_data)
            val_data.append(val_epoch_data)
            
            print(f"INFO: Train INFO for epoch {epoch}: ")
            for loss in train_epoch_data.keys():
                print(f"\t{loss}: {train_epoch_data[loss]}")
                
            print(f"\nINFO: validation INFO for epoch {epoch}: ")
            for loss in val_epoch_data.keys():
                print(f"\t{loss}: {val_epoch_data[loss]}")
            
            # update the loss in case it improved
            epoch_loss = val_epoch_data['val_loss']
            if epoch_loss < current_best_loss and self.save_if_improves:
                
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                
                print(f"INFO: New best val_loss reached: {epoch_loss} in epoch {epoch}. Saving model...")
                # saving the model
                path_to_save_best_model = os.path.join(self.checkpoint_dir, f"{self.type}_best_{epoch}.pth")
                torch.save(self.model.state_dict(), path_to_save_best_model)
                if epoch > self.start_epoch: # only remove after the model has already a checkpoint
                    os.remove(os.path.join(self.checkpoint_dir, f"{self.type}_best_{epoch_best}.pth"))
                
                # updating values of control
                epoch_best = epoch
                current_best_loss = epoch_loss
                epochs_w_o_improve = 0
            else: 
                epochs_w_o_improve +=1
                
            # if the save_every option is given
            if (self.save_every is not None) and (epoch % self.save_every == 0):
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                # saving the model
                torch.save(self.model.state_dict(), path_to_save_best_model)
                
            # if patience limit reached. stop training
            if epochs_w_o_improve == self.patience:
                print(f"WARNING: Patience limit reached. Exiting...\n\
                    Best model saved under {path_to_save_best_model} path.")
                return train_data, val_data
                
        return train_data, val_data
   
    def _train_epoch(self, epoch):
        """
        Logic behind just one epoch of training.
        
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        
        # put model into training mode
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        for _, _, (data, _) in tqdm(self.train_data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            
            self.optimizer.zero_grad()
            x_recons, mu, logvar, _ = self.model(x)
            loss, loss_recon, loss_kl = self._compute_mse_kl_loss(x, x_recons, mu, logvar)
            # compute gradients and backprop
            loss.backward()
            self.optimizer.step()
            
            # update loss values (reconstructed loss + KL loss)
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item() 
            
        # saving it into a dict for INFO loging
        train_log = {
            'loss': total_loss / len(self.train_data_loader),
            'loss_recon': total_recon / len(self.train_data_loader),
            'loss_kl': total_kl / len(self.train_data_loader)
        }
        
        # validation after epoch
        self.model.eval()

        # reset the losses
        total_val_loss = 0.0
        total_val_recon = 0.0
        total_val_kl = 0.0    
        
        with torch.no_grad():
            for _, _, (data, _) in self.valid_data_loader:
                x = data.type('torch.FloatTensor').to(self.device)

                x_recons, mu, logvar, _ = self.model(x)
                
                loss, loss_recon, loss_kl = self._compute_mse_kl_loss(x, x_recons, mu, logvar)
                
                # update loss values (reconstructed loss + KL loss)
                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl += loss_kl.item() 

        # saving it into a dict for INFO loging
        val_log = {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_loss_recon': total_recon / len(self.valid_data_loader),
            'val_loss_kl': total_kl / len(self.valid_data_loader)
        }
        
        return train_log, val_log
     
    
    def _compute_mse_kl_loss(self, x, x_recon, mu, logvar):
        
        # MSE loss
        error = x-x_recon
        recon_loss = torch.mean(torch.square(error), axis=[1,2,3])
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mu) -
                                   torch.exp(logvar), axis=1) 
        
        return torch.mean(recon_loss + self.recon_loss_weight*kl_loss, dim=0) , torch.mean(recon_loss), torch.mean(kl_loss)