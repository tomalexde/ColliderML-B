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
    
    def on_test_epoch_start(self): #Clears confusion matrix between runs
        self.conf_matrix.reset()

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

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss  = self.loss_function(y_hat, y)

        y_cpu        = y.cpu().detach().numpy()
        y_hat_proba  = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_proba, multi_class='ovr',
                            labels=self.all_possible_labels)
        self.log('val_auc',  auc)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss  = self.loss_function(y_hat, y)

        y_cpu       = y.cpu().detach().numpy()
        y_hat_proba = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_proba, multi_class='ovr',
                            labels=self.all_possible_labels)
        self.log('test_auc',  auc)
        self.log('test_loss', loss)
        self.conf_matrix.update(y_hat, y)
        self.final_cm = self.conf_matrix.compute().cpu().numpy()
        return loss

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