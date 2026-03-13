import torch

class MaskPredictionLoss(torch.nn.Module):
    """
    Custom loss function for masked prediction tasks, combining token and value losses.

    Args:
        token_weight (float): Weight for the token loss component. Default is 1.0.
        value_weight (float): Weight for the value loss component. Default is 1.0.
    """
    def __init__(self, token_weight=1.0, value_weight=1.0):
        super(MaskPredictionLoss, self).__init__()
        self.token_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.value_loss = torch.nn.MSELoss(reduction='none')
        self.token_w = token_weight
        self.value_w = value_weight

    def forward(self, tokens, token_out, value_out, token_labels, value_labels):
        """
        Forward pass for the loss calculation.

        Args:
            tokens (torch.Tensor): Input tokens with mask indicators.
            token_out (torch.Tensor): Predicted token outputs.
            value_out (torch.Tensor): Predicted value outputs.
            token_labels (torch.Tensor): True token labels.
            value_labels (torch.Tensor): True value labels.

        Returns:
            torch.Tensor: Combined loss value.
        """
        

        # first we get the masked tokens out of the input tokens (<mask> == 1)
        masked_tokens = (tokens.view(-1) == 1).float()

        # Reshape token outputs and true labels for loss calculation
        tpred = token_out.view(-1, token_out.size(-1))
        ttrue = token_labels.view(-1)

        # Calculate token loss and apply mask
        tloss = self.token_loss(tpred, ttrue)
        tloss = (tloss * masked_tokens).sum() / masked_tokens.sum()

        # Reshape value outputs and true labels for loss calculation
        vtrue = value_labels.view(-1)
        vpred = value_out.view(-1)

        # Calculate value loss and apply mask
        vloss = self.value_loss(vpred, vtrue)
        vloss = (vloss * masked_tokens).sum() / masked_tokens.sum()

        # Combine token and value losses with respective weights
        loss = self.token_w * tloss + self.value_w * vloss

        return loss