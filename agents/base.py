from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle

from corruptions import *
from torchvision.transforms import ToPILImage, PILToTensor
import torchvision

class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)

    
    def evaluate(self, test_loaders, task_num):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        acc_array_augmented = np.zeros(len(test_loaders))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                acc_augmented = AverageMeter()
                for i, (batch_x, batch_y, indices_1) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    
                    
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL',  'SCP']:
                        feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                        #old ncm
                        means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        means = means.transpose(1, 2)
                        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        _, pred_label = dists.min(1)
                        # may be faster
                        # feature = feature.squeeze(2).T
                        # _, preds = torch.matmul(means, feature).max(0)
                        correct_cnt = (np.array(self.old_labels)[
                                           pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    
                    
                    else:
                    
                        if self.agent == 'ER_DVC':
                            y = self.model(batch_x,batch_x)
                            logits,_,_,_ = y
                            _, pred_label = torch.max(logits, 1)
                            correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)
                        
                            if task_num == 9:
                            
                                np_seed_state = np.random.get_state()
                                torch_seed_state = torch.get_rng_state()
                                
            
                                # List to hold all the batches with distortions applied
                                all_batches = []
                                
                                # Convert the batch of images to a list of PIL images
                                to_pil = ToPILImage()
                                batch_x_pil_list = [to_pil(img.cpu()) for img in batch_x]  
                                
                                distortions = [
                                    gaussian_noise, shot_noise, impulse_noise, defocus_blur, motion_blur,
                                    zoom_blur, fog, snow, elastic_transform, pixelate, jpeg_compression
                                ]
                        
                                # Process each image in the batch
                                for batch_idx, batch_x_pil in enumerate(batch_x_pil_list):
                                    # List to hold the original and distorted images for the current batch image
                                    augmented_images = []
                                    
                                    # Add the original image to the list
                                    augmented_images.append(batch_x[batch_idx])
                                    
                                    # Loop through the distortions and apply them to the current image
                                    for function in distortions:
                                        if function in [pixelate, jpeg_compression]:
                                            # For functions returning tensors
                                            img_processed = PILToTensor()(function(batch_x_pil)).to(dtype=batch_x.dtype).to("cuda") / 255.0
                                        else:
                                            # For functions returning images
                                            img_processed = torch.tensor(function(batch_x_pil).astype(float) / 255.0, dtype=batch_x.dtype).to("cuda").permute(2, 0, 1)
                        
                                        # Append the distorted image
                                        augmented_images.append(img_processed)
                        
                                    # Concatenate the original and distorted images
                                    augmented_images_concatenated = torch.stack(augmented_images, dim=0)
                                    all_batches.append(augmented_images_concatenated)
                        
                                # Concatenate all the augmented batches along the batch dimension
                                batch_x_augmented = torch.cat(all_batches, dim=0)
                                
                                # Repeat each label for the number of augmentations plus the original image
                                batch_y_augmented = batch_y.repeat_interleave(len(distortions) + 1)
                                
       ##                 
       ##                         # Extract the first 12 images to display (or fewer if there are less than 12 images)
       ##                         images_display = [batch_x_augmented[j] for j in range(min(12, batch_x_augmented.size(0)))]
       ##                 
       ##                         # Make a grid from these images
       ##                         grid = torchvision.utils.make_grid(images_display, nrow=len(images_display))  # Adjust nrow based on actual images
       ##                         
       ##                         # Save grid image with unique name for each batch
       ##                         torchvision.utils.save_image(grid, 'grid_image.png')
       ##                         
                                
                                np.random.set_state(np_seed_state)
                                torch.set_rng_state(torch_seed_state)


                                y_augmented = self.model(batch_x_augmented,batch_x_augmented)
                                logits_augmented,__augmented,__augmented,__augmented = y_augmented
                                __augmented, pred_label_augmented = torch.max(logits_augmented, 1)
                                correct_cnt_augmented = (pred_label_augmented == batch_y_augmented).sum().item()/batch_y_augmented.size(0)
                        
                        
                        else:
                            y = self.model(batch_x)
                            logits,_ = y
                            _, pred_label = torch.max(logits, 1)
                            correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                    

                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen-1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                        elif task == self.task_seen -1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                    if task_num == 9:
                        acc_augmented.update(correct_cnt_augmented, batch_y_augmented.size(0))
                acc_array[task] = acc.avg()
                if task_num == 9:
                    acc_array_augmented[task] = acc_augmented.avg()
        print(acc_array)
        if task_num == 9:
            print(acc_array_augmented)
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
            with open('confusion', 'wb') as fp:
                pickle.dump([correct_lb, predict_lb], fp)
        
        if task_num == 9:
            return acc_array, acc_array_augmented

        else:
            return acc_array
