from common_imports import *
from TrackTransformer import TrackT
from torchmetrics.classification import MulticlassConfusionMatrix


class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self, feature_dim=4, hidden_size=256, num_heads=6, 
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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
