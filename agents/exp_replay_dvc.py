import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
import torch.nn as nn
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from loss import agmax_loss, cross_entropy_loss

import torch.nn as nn
from models.resnet1 import ResNet18
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle


class ExperienceReplay_DVC(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_DVC, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.agent = params.agent
        self.dl_weight = params.dl_weight

        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
           ColorJitter(0.4, 0.4, 0.4, 0.1),
           RandomGrayscale(p=0.2)

        )
        self.L2loss = torch.nn.MSELoss()
        self.soft_ = nn.Softmax(dim=1)

    
    def distribute_samples(self, probabilities, M):
        # Normalize the probabilities
        total_probability = sum(probabilities.values())
        normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}
    
        # Calculate the number of samples for each class
        samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
        
        # Check if there's any discrepancy due to rounding and correct it
        discrepancy = M - sum(samples.values())
        
        for key in samples:
            if discrepancy == 0:
                break
            if discrepancy > 0:
                samples[key] += 1
                discrepancy -= 1
            elif discrepancy < 0 and samples[key] > 0:
                samples[key] -= 1
                discrepancy += 1

        return samples


    def distribute_excess(self, lst):
        # Calculate the total excess value
        total_excess = sum(val - 500 for val in lst if val > 500)
    
        # Number of elements that are not greater than 500
        recipients = [i for i, val in enumerate(lst) if val < 500]
    
        num_recipients = len(recipients)
    
        # Calculate the average share and remainder
        avg_share, remainder = divmod(total_excess, num_recipients)
    
        lst = [val if val <= 500 else 500 for val in lst]
        
        # Distribute the average share
        for idx in recipients:
            lst[idx] += avg_share
        
        # Distribute the remainder
        for idx in recipients[:remainder]:
            lst[idx] += 1
        
        # Cap values greater than 500
        for i, val in enumerate(lst):
            if val > 500:
                return self.distribute_excess(lst)
                break
    
        return lst
    
    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        
        
        unique_classes = set()
        for _, labels, indices_1 in train_loader:
            unique_classes.update(labels.numpy())
        

        device = "cuda"
        Model_Carto = ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        

        mapping = {value: index for index, value in enumerate(unique_classes)}
        reverse_mapping = {index: value for value, index in mapping.items()}


        # Initializing the dictionaries        
        confidence_by_class = {class_id: {epoch: [] for epoch in range(8)} for class_id, __ in enumerate(unique_classes)}

        
        # Training
        Carto = torch.zeros((8, len(y_train)))
        for epoch_ in range(8):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            for batch_idx, (inputs, targets, indices_1) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)                
                targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
                
                optimizer_.zero_grad()
                outputs = Model_Carto(inputs)
                soft_ = self.soft_(outputs)
                confidence_batch = []
        
                # Accumulate confidences and counts
                for i in range(targets.shape[0]):
                    confidence_batch.append(soft_[i,targets[i]].item())
                    
                    # Update the dictionary with the confidence score for the current class for the current epoch
                    confidence_by_class[targets[i].item()][epoch_].append(soft_[i, targets[i]].item())

                loss = criterion_(outputs, targets)
                loss.backward()
                optimizer_.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
                conf_tensor = torch.tensor(confidence_batch)
                Carto[epoch_, indices_1] = conf_tensor
                
            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))

            scheduler_.step()

        mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}
        std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(8)])) for class_id, __ in enumerate(unique_classes)}
        
        ##print("std_of_means_by_class", std_of_means_by_class)

        Confidence_mean = Carto.mean(dim=0)
        Variability = Carto.std(dim=0)
        
        ##plt.scatter(Variability, Confidence_mean, s = 2)
        
        ##plt.xlabel("Variability") 
        ##plt.ylabel("Confidence") 
        
        ##plt.savefig('scatter_plot.png')
        
        # set up model
        self.model = self.model.train()
        self.transform = self.transform.cuda()
        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y, indices_1 = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = self.transform(batch_x)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    y = self.model(batch_x, batch_x_aug)
                    z, zt, _,_ = y
                    ce = cross_entropy_loss(z, zt, batch_y, label_smoothing=0)


                    agreement_loss, dl = agmax_loss(y, batch_y, dl_weight=self.dl_weight)
                    loss  = ce + agreement_loss + dl

                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(z, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z, batch_x)
                    _, pred_label = torch.max(z, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    if  self.params.retrieve == 'MGI':
                        mem_x, mem_x_aug, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    else:
                        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                        if mem_x.size(0) > 0:
                            mem_x_aug = self.transform(mem_x)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        y = self.model(mem_x, mem_x_aug)
                        z, zt, _,_ = y
                        ce = cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                        agreement_loss, dl = agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                        loss_mem = ce  + agreement_loss + dl

                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(z, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(z,
                                                                                                         mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(z, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.backward()
                    self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
        
        
        list_of_indices = []
        counter__ = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                counter__ +=1
                list_of_indices.append(i)

        top_n = counter__

        # Find the indices that would sort the array
        sorted_indices_1 = np.argsort(Confidence_mean.numpy())
        sorted_indices_2 = np.argsort(Variability.numpy())
        
        #top_indices_1 = sorted_indices_1[:top_n] #hard to learn
        #top_indices_sorted = top_indices_1 #hard to learn
        
        #top_indices_1 = sorted_indices_1[-top_n:] #easy to learn
        #top_indices_sorted = top_indices_1[::-1] #easy to learn
        
        #top_indices_1 = sorted_indices_2[-top_n:] #ambigiuous
        #top_indices_sorted = top_indices_1[::-1] #ambiguous


        ##top_indices_sorted = sorted_indices_1 #hard to learn
        
        ##top_indices_sorted = sorted_indices_1[::-1] #easy to learn

        top_indices_sorted = sorted_indices_2[::-1] #ambiguous

        
        subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
        trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=self.batch, shuffle=False, num_workers=0)

        images_list = []
        labels_list = []
        
        for images, labels, indices_1 in trainloader_C:  # Assuming train_loader is your DataLoader
            images_list.append(images)
            labels_list.append(labels)
        
        all_images = torch.cat(images_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)


        updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}
        
        ##print("updated_std_of_means_by_class", updated_std_of_means_by_class)

        dist = self.distribute_samples(updated_std_of_means_by_class, top_n)

        
        num_per_class = top_n//len(unique_classes)
        counter_class = [0 for _ in range(len(unique_classes))]

        if len(y_train) == top_n:
            condition = [num_per_class for _ in range(len(unique_classes))]
            diff = top_n - num_per_class*len(unique_classes)
            for o in range(diff):
                condition[o] += 1
        else:
            condition = [value for k, value in dist.items()]


        check_bound = len(y_train)/len(unique_classes)
        ##print("check_bound", check_bound)
        ##print("condition", condition, sum(condition))
        for i in range(len(condition)):
            if condition[i] > check_bound:
                ##print("iiiiiiiii", i)
                condition = self.distribute_excess(condition)
                break

        
        ##print("condition", condition, sum(condition), "top_n", top_n)
        images_list_ = []
        labels_list_ = []
        
        for i in range(all_labels.shape[0]):
            if counter_class[mapping[all_labels[i].item()]] < condition[mapping[all_labels[i].item()]]:
                counter_class[mapping[all_labels[i].item()]] += 1
                labels_list_.append(all_labels[i])
                images_list_.append(all_images[i])
            if counter_class == condition:
                ##print("yesssss")
                break

        
        all_images_ = torch.stack(images_list_)
        all_labels_ = torch.stack(labels_list_)

        indices = torch.randperm(all_images_.size(0))
        shuffled_images = all_images_[indices]
        shuffled_labels = all_labels_[indices]
        ##print("shuffled_labels.shape", shuffled_labels.shape)
        
        self.buffer.buffer_label[list_of_indices] = shuffled_labels.to(device)
        self.buffer.buffer_img[list_of_indices] = shuffled_images.to(device)
        
        self.after_train()

