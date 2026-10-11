"""
EXCERPT - Cellular Handover Prediction (UCC Final Year Project, Darragh O'Keeffe)

Mixture-of-Experts handover classifier (PyTorch) and the event-level metric used
to evaluate it. Adapted and lightly cleaned up from the private modelling notebook;
the training loop, cross-validation and plotting code are omitted. Full source on request.

Why a Mixture of Experts: the route mixes dense city-centre cells with sparse coastal
coverage. A gating network can route each sample to the experts suited to that regime
instead of forcing one set of weights to average across all of them.
"""
import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, num_numeric, num_cells, num_modems, embed_dim=32, num_experts=5):
        super().__init__()
        # Learned embeddings (rather than one-hot) let similar cells/modems share structure.
        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.modem_embed = nn.Embedding(num_modems, embed_dim)
        input_dim = num_numeric + 2 * embed_dim

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                          nn.Linear(256, 128), nn.ReLU(),
                          nn.Linear(128, 1))
            for _ in range(num_experts)
        ])
        # Gate sees the same input and outputs a soft weighting over the experts.
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                  nn.Linear(64, num_experts), nn.Softmax(dim=1))

    def forward(self, x_num, x_cell, x_modem):
        x = torch.cat([x_num, self.cell_embed(x_cell), self.modem_embed(x_modem)], dim=1)
        weights = self.gate(x)                                         # (batch, E)
        outputs = torch.stack([e(x) for e in self.experts], dim=2)     # (batch, 1, E)
        logit = torch.bmm(outputs, weights.unsqueeze(2)).squeeze(2)    # weighted sum
        return logit, weights            # weights are kept to analyse expert usage


# Training setup per fold (leave-one-trip-instance-out, scaler fit on train only):
#   loss      = nn.BCEWithLogitsLoss(pos_weight=min(n_neg / n_pos, 10))  # class imbalance
#   optimiser = Adam(lr=1e-3) + ReduceLROnPlateau(factor=0.5, patience=3)
#   50 epochs max, batch size 256, early stopping


def event_detection_rate(test_df, probs, threshold=0.01, k=5):
    """Share of real handovers that got at least one alarm in the k samples
    up to and including the event. Alarms *after* the handover are not credited.
    A low threshold is deliberate: a missed handover costs far more than a false alarm.
    """
    test_df = test_df.assign(pred=(probs.flatten() > threshold).astype(int))
    detected = total = 0
    for _, trip in test_df.groupby("trip_instance"):
        for i in trip.index[trip["handover_next_sample"] == 1]:
            total += 1
            window = [j for j in range(i - k, i + 1) if j in trip.index]
            detected += int(trip.loc[window, "pred"].any())
    return detected / total if total else 0.0
