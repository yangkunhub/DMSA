import torch


def get_aff_loss(inputs, targets):


    inputs=inputs.max(1)[1]
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)
#    print('inputs:', inputs.shape)
#    print('targets',targets.shape)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count
    
    if pos_loss<0:
        pos_loss = -pos_loss
    if neg_loss<0:
        neg_loss = -neg_loss
            
    return 0.5 * pos_loss + 0.5 * neg_loss