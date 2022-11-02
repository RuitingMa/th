import os
import torch
import numpy as np
import torch.nn.functional as nnf
from torch import save, no_grad
from tqdm import tqdm
from models.xnor_layers import XNORConv2d
import shutil


class BnnClassifier():
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
        # if is_best:
        #     shutil.copyfile(filename, os.path.join(head,
        #         '{0}_best.pth.tar'.format(tail)))

        return

    def test(self, criterion):
        self.model.eval()
        top1 = 0
        test_loss = 0.
        # preds=[]
        preds = torch.Tensor([])
        confs = torch.Tensor([])

        with no_grad():
            for data, target in tqdm(self.test_loader):
                # print (target)
                data, target = data.to(self.device), target.to(self.device)   
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True) 
                # preds.append(pred)
                # print ("pred: " + str(len(pred)) )  
                preds = torch.cat ((preds,pred))
                top1 += pred.eq(target.view_as(pred)).sum().item()
                probs = nnf.softmax(output, dim=1)
                confs = torch.cat ((confs,probs))
                conf = torch.max(probs, 1)
                

        top1_acc = 100. * top1 / len(self.test_loader.sampler)

        # print (confs)
        return top1_acc,preds,confs


    def top1_accuracy(self):
        return top1_accuracy(self.model, self.test_loader, self.device)


    def train_step(self, criterion, optimizer):
        losses = []
        for data, target in tqdm(self.train_loader,
                total=len(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1,1))
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