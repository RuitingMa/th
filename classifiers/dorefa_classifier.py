import os
import torch
import numpy as np
import torch.nn.functional as nnf
from torch import save, no_grad
from tqdm import tqdm
import shutil

class DorefaClassifier():
    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def save_checkpoint(state, checkpoint):
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, '{0}_checkpoint.pth.tar'.format(tail))
        save(state, filename)

        return

    def test(self, criterion):
        self.model.eval()
        top1 = 0
        test_loss = 0.
        # preds=[]
        preds = torch.Tensor([])
        confs = torch.Tensor([])
        test_result = [0,0,0,0,0,0,0,0,0,0]
        classes = [0,0,0,0,0,0,0,0,0,0]

        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                for m in range(100):
                    classes[target[m]] += 1
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  
                preds = torch.cat ((preds,pred))
                re = pred.eq(target.view_as(pred))
                top1 += re.sum().item()    
                # top1 += pred.eq(target.view_as(pred)).sum().item()
                probs = nnf.softmax(output, dim=1)
                confs = torch.cat ((confs,probs))
                conf = torch.max(probs, 1)
                for m in range(0,100):
                    if (re[m]==False):
                        test_result[target[m]] += 1

        print ("len")
        print (len(self.test_loader.sampler))
        print (top1)
        top1_acc = 100. * top1 / len(self.test_loader.sampler)
        print ("Test Result:")
        print (test_result)
        print ("Classes")
        print (classes)
        # print (preds.size())
        self.confidences = confs
        return top1_acc,preds,confs



    def train_step(self, criterion, optimizer):
        losses = []
        self.model.train()

        for data, target in tqdm(self.train_loader,
                total=len(self.train_loader)):
            

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            
            output = self.model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            

        return losses

    def train(self, criterion, optimizer, epochs, scheduler,
            checkpoint=None):

        if checkpoint is None:
            raise ValueError('Specify a valid checkpoint')

        
        best_accuracy = 0.

        losses = []
        accuracies = []



        for epoch in range(1, epochs+1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']  
            # test_accuracy = self.test(criterion)
            # accuracies.append(test_accuracy)
            if scheduler:     
                scheduler.step()
            # is_best = test_accuracy > best_accuracy
            # if is_best:
            #     best_accuracy = test_accuracy
            
            print('Train Epoch {0}\t Loss: {1:.6f} \t lr: {2:.4f}'
                    .format(epoch, epoch_losses.mean(), lr))
            # print('Best accuracy: {:.3f} '.format(best_accuracy))

            self.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': self.model.state_dict(),
                # 'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
                'criterion': criterion,
                }, checkpoint)

        test_accuracy,preds,conf = self.test(criterion)
        print ("Test Accuracy: {}.".format(test_accuracy))
        print ("==================")

        return test_accuracy,preds,conf
    
    def get_test_loader (self):
        return self.test_loader

    def get_model (self):
        return self.model

    def set_label (self,label):
        self.label = label
    
    def get_label (self):
        return self.label

    def get_confidences (self):
        return self.confidences

    def set_confidences (self, confidences):
        self.confidences = confidences
