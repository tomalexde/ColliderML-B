from common_imports import *
from Transformer.TrackTransformer import TrackT
from torchmetrics.classification import MulticlassConfusionMatrix


class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8,
                 num_encoder_layers=4, output_size=4, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.all_possible_labels = list(range(output_size))

        self.conf_matrix = MulticlassConfusionMatrix(num_classes=output_size)
        self.class_names = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']

        self.model = TrackT(
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            output_size=output_size,
        )
        self.learning_rate  = learning_rate
        self.loss_function  = nn.CrossEntropyLoss()
        self.final_cm       = None

    def forward(self, x, mask):
        # Both x (B, max_hits, 3) and mask (B, max_hits) are required
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch   # DataModule now returns (x, mask, y)

        if batch_idx == 0 and self.current_epoch == 0:
            print("\n" + "=" * 40)
            print("PADDED TENSOR SANITY CHECK")
            print(f"x shape:    {x.shape}   (B, max_hits_padded, 3)")
            print(f"mask shape: {mask.shape}  (B, max_hits_padded) — True=pad")
            print(f"real hits per event (first 5): {(~mask).sum(dim=1)[:5].tolist()}")
            print("=" * 40 + "\n")

        y_hat = self(x, mask)
        loss  = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        # Buffers to collect predictions and labels across all batches
        # They are lists of tensors — gathered across GPUs in epoch_end
        self._val_preds  = []
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss  = self.loss_function(y_hat, y)

        # Accumulate softmax probabilities and labels (keep on GPU as tensors)
        self._val_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._val_labels.append(y.detach())

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate this GPU's batches into one tensor each
        preds  = torch.cat(self._val_preds,  dim=0)   # (N_local, 4)
        labels = torch.cat(self._val_labels, dim=0)   # (N_local,)

        # self.all_gather() collects tensors from all GPUs onto every rank,
        # returning shape (num_gpus, N_local, 4) / (num_gpus, N_local).
        # We then flatten to get the full dataset on every rank.
        all_preds  = self.all_gather(preds).view(-1, preds.shape[-1])   # (N_total, 4)
        all_labels = self.all_gather(labels).view(-1)                    # (N_total,)

        # Compute exact AUC on rank 0 only — all ranks have identical data
        # but we only need to log once to avoid duplicate W&B entries
        if self.trainer.is_global_zero:
            y_cpu       = all_labels.cpu().numpy()
            y_hat_cpu   = all_preds.cpu().numpy()
            auc = roc_auc_score(y_cpu, y_hat_cpu, multi_class='ovr',
                                labels=self.all_possible_labels)
            # rank_zero_only=True: only rank 0 logs, preventing duplicate entries
            self.log('val_auc', auc, rank_zero_only=True)

        # Free buffers
        self._val_preds  = []
        self._val_labels = []

    def on_test_epoch_start(self):
        self.conf_matrix.reset()
        self._test_preds  = []
        self._test_labels = []

    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss  = self.loss_function(y_hat, y)

        self._test_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._test_labels.append(y.detach())
        self.log('test_loss', loss, sync_dist=True)
        self.conf_matrix.update(y_hat, y)
        return loss

    def on_test_epoch_end(self):
        # Gather predictions from all GPUs — same approach as validation
        preds  = torch.cat(self._test_preds,  dim=0)
        labels = torch.cat(self._test_labels, dim=0)

        all_preds  = self.all_gather(preds).view(-1, preds.shape[-1])
        all_labels = self.all_gather(labels).view(-1)

        if self.trainer.is_global_zero:
            y_cpu     = all_labels.cpu().numpy()
            y_hat_cpu = all_preds.cpu().numpy()
            auc = roc_auc_score(y_cpu, y_hat_cpu, multi_class='ovr',
                                labels=self.all_possible_labels)
            self.log('test_auc', auc, rank_zero_only=True)

        self._test_preds  = []
        self._test_labels = []

        # Confusion matrix — torchmetrics syncs across GPUs here automatically
        self.final_cm = self.conf_matrix.compute().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        max_epochs      = self.trainer.max_epochs
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps     = max_epochs * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }