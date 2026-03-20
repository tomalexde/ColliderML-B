from common_imports import *
from Transformer.TrackTransformer_Flash import TrackT_Flash
from torchmetrics.classification import MulticlassConfusionMatrix


class LightningNeuralNetwork(pl.LightningModule):
    """NeuralNetwork for FlashAttention version. Batch = (x_packed, cu_seqlens, max_seqlen, y)."""

    def __init__(self, feature_dim=3, hidden_size=256, num_heads=8,
                 num_encoder_layers=6, output_size=4, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.all_possible_labels = list(range(output_size))
        self.conf_matrix   = MulticlassConfusionMatrix(num_classes=output_size)
        self.model         = TrackT_Flash(feature_dim, hidden_size, num_heads,
                                          num_encoder_layers, output_size)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.final_cm      = None

    def forward(self, x_packed, cu_seqlens, max_seqlen):
        return self.model(x_packed, cu_seqlens, max_seqlen)

    def _unpack(self, batch):
        x_packed, cu_seqlens, max_seqlen, y = batch
        cu_seqlens = cu_seqlens.to(x_packed.device)  # FA requires cu_seqlens on GPU
        return x_packed, cu_seqlens, max_seqlen, y

    def training_step(self, batch, batch_idx):
        x, cu, ms, y = self._unpack(batch)
        loss = self.loss_function(self(x, cu, ms), y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_preds, self._val_labels = [], []

    def validation_step(self, batch, batch_idx):
        x, cu, ms, y = self._unpack(batch)
        y_hat = self(x, cu, ms)
        loss  = self.loss_function(y_hat, y)
        self._val_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._val_labels.append(y.detach())
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        preds  = torch.cat(self._val_preds,  dim=0)
        labels = torch.cat(self._val_labels, dim=0)
        all_p  = self.all_gather(preds).view(-1, preds.shape[-1])
        all_l  = self.all_gather(labels).view(-1)
        if self.trainer.is_global_zero:
            auc = roc_auc_score(all_l.cpu().numpy(), all_p.cpu().numpy(),
                                multi_class='ovr', labels=self.all_possible_labels)
            self.log('val_auc', auc, rank_zero_only=True)
        self._val_preds, self._val_labels = [], []

    def on_test_epoch_start(self):
        self.conf_matrix.reset()
        self._test_preds, self._test_labels = [], []

    def test_step(self, batch, batch_idx):
        x, cu, ms, y = self._unpack(batch)
        y_hat = self(x, cu, ms)
        loss  = self.loss_function(y_hat, y)
        self._test_preds.append(torch.softmax(y_hat, dim=1).detach())
        self._test_labels.append(y.detach())
        self.log('test_loss', loss, sync_dist=True)
        self.conf_matrix.update(y_hat, y)
        return loss

    def on_test_epoch_end(self):
        preds  = torch.cat(self._test_preds,  dim=0)
        labels = torch.cat(self._test_labels, dim=0)
        all_p  = self.all_gather(preds).view(-1, preds.shape[-1])
        all_l  = self.all_gather(labels).view(-1)
        if self.trainer.is_global_zero:
            auc = roc_auc_score(all_l.cpu().numpy(), all_p.cpu().numpy(),
                                multi_class='ovr', labels=self.all_possible_labels)
            self.log('test_auc', auc, rank_zero_only=True)
        self._test_preds, self._test_labels = [], []
        self.final_cm = self.conf_matrix.compute().cpu().numpy()

    def configure_optimizers(self):
        optimizer   = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return {"optimizer": optimizer}