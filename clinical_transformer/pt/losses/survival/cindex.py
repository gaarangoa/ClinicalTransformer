import torch

def cindex_loss(Target, y_pred, sigma=1.0):
    """
    :param Target: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X (final layer output)
    :return: c-index loss:1.0-sum_{j,k} { w_{jk} * 1/(1+exp( (eta_k-eta_j)/sigma ))}
  
                w_{jk}=delta_j*I(T_j<T_k) / sum_{j,k} {delta_j*I(T_j<T_k)}
                delta_j = 0,1 [censored, deceased]
                sigma =1.0, a smoothing parameter for the exponential functions
    """
    y_true = Target[:, 0].float()
    event = Target[:, 1].float()

    n = y_pred.shape[0]

    etak = y_pred.repeat([1, n], n)
    etaj = etak.t()

    etaMat = etak - etaj
    sigmoid_eta = 1.0 / (1.0 + torch.exp(etaMat / sigma))  

    eventI = event.repeat([n, 1], n).t()
    weightsj = y_true.repeat([n, 1], n).t()
    weightsk = y_true.repeat([n, 1], n)

    rank_mat = torch.where(weightsj == weightsk, torch.tensor(1e-8), (weightsj < weightsk).float())
    rank_mat = rank_mat - torch.diag(torch.zeros(n, device=y_pred.device) + 1e-8) # set diagonals to zero

    # # # matrix of comparable pairs
    rank_mat = rank_mat * eventI

    rank_mat = rank_mat / rank_mat.sum()

    loss = (sigmoid_eta * rank_mat).sum() 
    
    return 1 - loss


def cindex_loss_squared(Target, y_pred, sigma=1.0):
    """
    :param Target: n x 2 tensor: n, (time, event)
    :param y_pred: n x 1 tensor: n, beta*X (final layer output)
    :return: c-index loss:1.0-sum_{j,k} { w_{jk} * 1/(1+exp( (eta_k-eta_j)/sigma ))}
  
                w_{jk}=delta_j*I(T_j<T_k) / sum_{j,k} {delta_j*I(T_j<T_k)}
                delta_j = 0,1 [censored, deceased]
                sigma =1.0, a smoothing parameter for the exponential functions

    # x = (np.array(list(range(0, 11, 1)))+0.01) / 10
    # sns.scatterplot(x=x, y=-x**2 + x)
    
    """
    y_true = Target[:, 0].float()
    event = Target[:, 1].float()

    n = y_pred.shape[0]

    etak = y_pred.repeat([1, n], n)
    etaj = etak.t()

    etaMat = etak - etaj
    sigmoid_eta = 1.0 / (1.0 + torch.exp(etaMat / sigma))  

    eventI = event.repeat([n, 1], n).t()
    weightsj = y_true.repeat([n, 1], n).t()
    weightsk = y_true.repeat([n, 1], n)

    rank_mat = torch.where(weightsj == weightsk, torch.tensor(1e-8), (weightsj < weightsk).float())
    rank_mat = rank_mat - torch.diag(torch.zeros(n, device=y_pred.device) + 1e-8) # set diagonals to zero

    # # # matrix of comparable pairs
    rank_mat = rank_mat * eventI

    rank_mat = rank_mat / rank_mat.sum()

    loss = (sigmoid_eta * rank_mat).sum() 
    
    return loss * (1 - loss)