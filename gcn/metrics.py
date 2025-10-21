import torch
import torch.nn.functional as F

def masked_softmax_cross_entropy(preds, labels, mask):
    
    labels = labels.argmax(dim=1)
    loss = F.cross_entropy(preds, labels, reduction='none')

    mask = mask.float()
    mask /= mask.mean()
    loss *= mask

    return loss.mean()


def masked_accuracy(preds, labels, mask):
 

    pred_classes = torch.argmax(preds, dim=1)
    true_classes = torch.argmax(labels, dim=1)

    correct_prediction = (pred_classes == true_classes).float()

    # Convert mask to float and normalize it
    mask = mask.float()
    mask = mask / mask.mean()
    correct_prediction *= mask

    return correct_prediction.mean()
