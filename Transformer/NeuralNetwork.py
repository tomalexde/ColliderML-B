from common_imports import *
from Transformer.TrackTransformer import TrackT
from torchmetrics.classification import MulticlassConfusionMatrix


class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8,
                 num_encoder_layers=6, output_size=4, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.all_possible_labels = list(range(output_size))
        self.conf_matrix  = MulticlassConfusionMatrix(num_classes=output_size)
        self.class_names  = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']
        self.model        = TrackT(
            feature_dim=feature_dim, hidden_size=hidden_size,
            num_heads=num_heads, num_encoder_layers=num_encoder_layers,
            output_size=output_size,
        )
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.final_cm      = None

    def forward(self, x_packed, cu_seqlens, max_seqlen):
        return self.model(x_packed, cu_seqlens, max_seqlen)

    def _unpack_batch(self, batch):
        """DataModule returns (x_packed, cu_seqlens, max_seqlen, y)."""
        x_packed, cu_seqlens, max_seqlen, y = batch
        # Move cu_seqlens to GPU — flash-attn requires it on the same device
        cu_seqlens = cu_seqlens.to(x_packed.device)
        return x_packed, cu_seqlens, max_seqlen, y

    def training_step(self, batch, batch_idx):
        x, cu, max_s, y = self._unpack_batch(batch)
        y_hat = self(x, cu, max_s)
        loss  = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_preds  = []
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        x, cu, max_s, y = self._unpack_batch(batch)
        y_hat = self(x, cu, max_s)
        loss  = self.loss_function(y_hat, y)
        self._val_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._val_labels.append(y.detach())
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        preds  = torch.cat(self._val_preds,  dim=0)
        labels = torch.cat(self._val_labels, dim=0)
        all_preds  = self.all_gather(preds).view(-1, preds.shape[-1])
        all_labels = self.all_gather(labels).view(-1)
        if self.trainer.is_global_zero:
            auc = roc_auc_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
                                multi_class='ovr', labels=self.all_possible_labels)
            self.log('val_auc', auc, rank_zero_only=True)
        self._val_preds  = []
        self._val_labels = []

    def on_test_epoch_start(self):
        self.conf_matrix.reset()
        self._test_preds  = []
        self._test_labels = []

    def test_step(self, batch, batch_idx):
        x, cu, max_s, y = self._unpack_batch(batch)
        y_hat = self(x, cu, max_s)
        loss  = self.loss_function(y_hat, y)
        self._test_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._test_labels.append(y.detach())
        self.log('test_loss', loss, sync_dist=True)
        self.conf_matrix.update(y_hat, y)
        return loss

    def on_test_epoch_end(self):
        preds  = torch.cat(self._test_preds,  dim=0)
        labels = torch.cat(self._test_labels, dim=0)
        all_preds  = self.all_gather(preds).view(-1, preds.shape[-1])
        all_labels = self.all_gather(labels).view(-1)
        if self.trainer.is_global_zero:
            auc = roc_auc_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
                                multi_class='ovr', labels=self.all_possible_labels)
            self.log('test_auc', auc, rank_zero_only=True)
        self._test_preds  = []
        self._test_labels = []
        self.final_cm = self.conf_matrix.compute().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps     = self.trainer.max_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1e4,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}