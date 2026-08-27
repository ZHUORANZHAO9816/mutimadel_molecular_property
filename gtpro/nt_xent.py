"""
NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss) for contrastive learning.

Used by both pretraining and finetuning pipelines.
"""
import torch
import torch.nn as nn
import numpy as np


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss for Siamese networks."""

    def __init__(self, device: torch.device, batch_size: int, temperature: float,
                 use_cosine_similarity: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity: bool):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self) -> torch.Tensor:
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dot-product similarity matrix (N x 2N)."""
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            zis: Projection from first view, shape (batch_size, dim)
            zjs: Projection from second view, shape (batch_size, dim)

        Returns:
            Scalar loss.
        """
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # Filter positive scores (diagonals of the off-diagonal blocks)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size, device=self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)