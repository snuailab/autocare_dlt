import math
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def seg_evaluation(output, target, classes, loss_manager):
    conf_sum = np.zeros((len(classes), len(classes)))
    cor, tot = 0, 0
    loss_sum = []
    for batch_idx, batched_data in enumerate(output):
        batched_data = [x.squeeze() for x in batched_data]
        batched_label = target[batch_idx]
        labels_for_loss = []
        for i, data in enumerate(batched_data):
            label = batched_label[i]
            label = label["labels"]
            # for calculate validation loss on CUDA
            labels_for_loss.append({"labels": label.cuda()})

            pred = torch.argmax(data, dim=0)
            pred = pred.reshape(-1)
            label = label.reshape(-1)
            cor += torch.sum(pred==label).item()
            tot += len(label)
            conf = confusion_matrix(label, pred)

            conf_indices = torch.unique(torch.cat([pred, label]))
            for i in range(conf.shape[0]):
                for j in range(conf.shape[1]):
                    conf_sum[int(conf_indices[i].item()), int(conf_indices[j].item())] += conf[i][j]
        
        # for calculate validation loss on CUDA
        batched_data = [x.cuda() for x in batched_data]
        loss, _ = loss_manager(batched_data, labels_for_loss)
        loss_sum.append(loss.item())

    accuracy = cor/tot
    recall_dict = {}
    precision_dict = {}
    rev_conf = conf_sum.transpose(1, 0)
    for i in range(len(classes)):
        recall = conf_sum[i][i]/sum(conf_sum[i])
        precision = rev_conf[i][i]/sum(rev_conf[i])
        if recall == recall and recall > 0:
            recall_dict[classes[i]] = recall
            precision_dict[classes[i]] = precision
        if precision == precision and precision > 0:
            recall_dict[classes[i]] = recall
            precision_dict[classes[i]] = precision
    avg_loss = sum(loss_sum)/len(loss_sum)

    return avg_loss, accuracy, recall_dict, precision_dict