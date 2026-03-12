import torch
import torch.nn.functional as F

def ntxent(P1, P2, temperature=0.5):
    """
    Contrastive (NT-Xent) loss for embeddings.
    P1: (N, D) tensor, embeddings for original samples
    P2: (N, D) tensor, embeddings for augmented samples
    """
    N = P1.size(0)

    # Normalize embeddings (important for cosine similarity)
    Z1 = F.normalize(P1, dim=1)
    Z2 = F.normalize(P2, dim=1)

    # Concatenate both views: (2N, D)
    Z = torch.cat([Z1, Z2], dim=0)

    # Similarity matrix (cosine similarity)
    sim = torch.matmul(Z, Z.T) / temperature  # (2N, 2N)

    # Mask to remove self-comparisons
    mask = torch.eye(2 * N, dtype=torch.bool, device=Z.device)
    sim.masked_fill_(mask, -1e10)

    # Labels: for i in [0..N-1], positive pair is i <-> i+N
    labels = torch.arange(N, device=Z.device)
    labels = torch.cat([labels + N, labels])  # (2N,)

    # Cross-entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss