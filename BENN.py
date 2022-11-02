import torch
import numpy as np
from tqdm import tqdm
from torch import no_grad

"""Includes the different ensemble voting algorithms discussed in the thesis"""
class BENN():
    def __init__(self,device,test_loader):
        self.device = device
        self.test_loader = test_loader
        torch.manual_seed(0)
        if self.device == 'cuda':
            torch.backends.cudnn.deterministic=True
            torch.cuda.manual_seed(0)
    
    # majority voting
    def calculate_majority_voting(self,ensemble):
        T = torch.mode(ensemble,1)[0]
        i = 0
        top1 = 0
        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)  
                pred = T[(i*100):((i+1)*100)]
                top1 += pred.eq(target.view_as(pred)).sum().item()
                i+=1 

        top1_acc = 100. * top1 / len(self.test_loader.sampler)
        print ("Ensemble Accuracy Hard Voting: {}.".format(top1_acc))
    # soft voting, average confidence score
    def calculate_average_confidence (self,confidences):
        top1 = 0
        c = 0
        T_1 = np.zeros(100)
        confidence_subsets = []
        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                for i in range(0,100):
                    l = []
                    for j in range(0,len(confidences)):
                        c_1 = confidences[j][i+c*100]
                        l.append(c_1)
                    s = torch.stack(l)
                    means = torch.mean(s,dim=0)
                    pred_1 = torch.argmax(means)
                    T_1[i] = pred_1.item()
                T_t = torch.from_numpy(T_1)
                top1 += T_t.eq(target.view_as(T_t)).sum().item()
                c+=1 
        top1_acc = 100. * top1 / len(self.test_loader.sampler)
        print ("Ensemble Accuracy Average Confidence: {}.".format(top1_acc))

    # soft voting, sum confidence score
    def calculate_sum_confidence (self,confidences):
        top1 = 0
        c = 0
        T_1 = np.zeros(100)
        confidence_subsets = []
        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                for i in range(0,100):
                    l = []
                    for j in range(0,len(confidences)):
                        c_1 = confidences[j][i+c*100]
                        l.append(c_1)
                    s = torch.stack(l)
                    sums = torch.sum(s,dim=0)
                    pred_1 = torch.argmax(sums)
                    T_1[i] = pred_1.item()
                T_t = torch.from_numpy(T_1)
                top1 += T_t.eq(target.view_as(T_t)).sum().item()
                c+=1 
        top1_acc = 100. * top1 / len(self.test_loader.sampler)
        print ("Ensemble Accuracy Sum Confidence: {}.".format(top1_acc))

    # soft voting, square confidence score 
    def calculate_square_confidence (self,confidences):
        top1_mean = 0
        c = 0
        T_mean = np.zeros(100)
        means_final = torch.Tensor([])
        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                for i in range(0,100):
                    l = []
                    for j in range(0,len(confidences)):
                        c_1 = torch.square(confidences[j][i+c*100])
                        l.append(c_1)
                    s = torch.stack(l)
                    means = torch.mean(s,dim=0)
                    means_final = torch.cat ((means_final,torch.unsqueeze(means,0)))
                    pred_mean = torch.argmax(means)
                    T_mean[i] = pred_mean.item()
                T_t_mean = torch.from_numpy(T_mean)
                re = T_t_mean.eq(target.view_as(T_t_mean))
                top1_mean += re.sum().item()
                c+=1 
        top1_acc_mean = 100. * top1_mean / len(self.test_loader.sampler)
        print ("Ensemble Accuracy Mean Square Confidence: {}.".format(top1_acc_mean))
        return means_final

    # Goes through the main classifier's confidence score
    # And chooses which of the expert classifiers to use in the voting process
    # Gives the highest accuracy
    def square_confidence_modified (self,classifications,main_model_confs,main_model_preds):
        # go through each single confidence score (ie each photo)
        counter = 0
        top1_mean = 0
        T_mean = np.zeros(100)
        test_result = [0,0,0,0,0,0,0,0,0,0]
        results_diff_1 = [0,0,0,0,0,0,0,0,0,0]
        results_diff_2 = [0,0,0,0,0,0,0,0,0,0]

        for data, target in tqdm(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            for m in range(0,100):
                l = []
                possible_digits = []
                #======================
                # if (torch.argmax(main_model_confs[m+counter*100])==9):
                #     c_1 = torch.square(main_model_confs[m+counter*100])
                #     l.append(c_1)
                #======================
                # Find digits classified with higher confidence
                for i in range (0,10):
                    if (main_model_confs[m+counter*100][i]>=0.15):
                        possible_digits.append(i)
                # Find models trained on these possible digits
                for p in possible_digits:
                    for j in range(len(classifications)):
                        if (p in classifications[j].get_label()):
                                c_1 = torch.square(classifications[j].get_confidences()[m+counter*100])
                                l.append(c_1)

                # If no possible digits have been found              
                if (len(l)==0):
                    max_conf_digit = torch.argmax(main_model_confs[m+counter*100])
                    for j in range(len(classifications)):
                        if (max_conf_digit in classifications[j].get_label()):
                            c_1 = torch.square(classifications[j].get_confidences()[m+counter*100])
                            l.append(c_1)
                s = torch.stack(l)
                means = torch.mean(s,dim=0)
                pred_mean = torch.argmax(means)
                T_mean[m] = pred_mean.item()
            T_t_mean = torch.from_numpy(T_mean)
            re = T_t_mean.eq(target.view_as(T_t_mean))
            top1_mean += re.sum().item()
            main_preds = main_model_preds[(counter*100):((counter+1)*100)]
            result_new = main_preds.eq(target.view_as(main_preds))
            counter += 1
            for k in range(0,100):
                    if (re[k]==False):
                        test_result[target[k]] += 1
                    # images that are incorrectly classified by the ensemble but correctly by the initial model
                    if (re[k]==False and result_new[k]==True):
                        results_diff_1[target[k]] += 1
                    # images that are correctly classified by the ensemble but not by the initial model
                    if (re[k]==False and result_new[k]==False):
                        results_diff_2[target[k]] += 1    
        top1_acc_mean = 100. * top1_mean / len(self.test_loader.sampler)
        print ("Ensemble Accuracy Weighted Mean Square Confidence: {}.".format(top1_acc_mean))
        print (test_result)
        print (results_diff_1)
        print (results_diff_2)
        return T_t_mean

# helper function
def replace_tensor_elements (t):
    res = t.clone()
    res[t<2.0] = 0
    return res