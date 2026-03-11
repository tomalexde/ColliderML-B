from common_imports import *
from Transformer.TrackTransformer import TrackT
from torchmetrics.classification import MulticlassConfusionMatrix


class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self, feature_dim=3, hidden_size=256, num_heads=6, 
                 num_encoder_layers=4, output_size=1, learning_rate=0.0001):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters()
        
        # Confusion Matrix metric
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=output_size)
        self.class_names = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']

        self.model = TrackT(feature_dim=feature_dim, 
                                     hidden_size=hidden_size, 
                                     num_heads=num_heads, 
                                     num_encoder_layers=num_encoder_layers, 
                                     output_size=output_size)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.final_cm = None # Placeholder for the export logic
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Only print on the very first batch of the first epoch
        if batch_idx == 0 and self.current_epoch == 0:
            print("\n" + "="*40)
            print("NESTED TENSOR SANITY CHECK")
            print(f"Batch Size: {len(x.nested_size())}")
            # Show first 3 event hit counts to prove they are different (jagged)
            print(f"First 3 event sizes: {x.nested_size()[:3].tolist()}")
            print("="*40 + "\n")

        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # Log AUC (multiclass: probabilities and multi_class='ovr')
        y_cpu = y.cpu().detach().numpy()
        y_hat_proba = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_proba, multi_class='ovr')
        self.log('val_auc', auc)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # Log AUC (multiclass: probabilities and multi_class='ovr')
        y_cpu = y.cpu().detach().numpy()
        y_hat_proba = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_proba, multi_class='ovr')
        self.log('test_auc', auc)
        self.log('test_loss', loss)
        self.conf_matrix.update(y_hat, y)
        self.final_cm = self.conf_matrix.compute().cpu().numpy()
        return loss
    
    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        max_epochs = self.trainer.max_epochs
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = max_epochs * steps_per_epoch

        #Linear Warm-up
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,         # 10% of the training is the 'Warm-up' phase
            anneal_strategy='cos', # Smoothly drops the LR after the peak
            div_factor=25.0,       # Start LR is max_lr / 25
            final_div_factor=1e4   # End LR is max_lr / 10,000
        )

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step", # This ensures the LR updates every batch
        },
    }
