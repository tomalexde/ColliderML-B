from common_imports import *
from argparse import ArgumentParser
from Transformer.NeuralNetwork import LightningNeuralNetwork
from Data.DataPrepare import prepare_it_all
import io
import wandb
# Lighting and WANDB
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

#Debugging
torch.set_float32_matmul_precision('high')

def main(hparams):
    #WANDB
    print("Login to WANDB")
    wandb.login()
    # Data Preparation
    # We prepare the nested data lists and wrap them in the DataModule
    print("Preparing nested physics data...")
    # Adjust events/purity/maxhits based on your GPU capacity
    if hparams.num_events == 0:
        hparams.num_events = hparams.num_events_list
    data_module = prepare_it_all(
        events=hparams.num_events, 
        purity_scale=hparams.purity, 
        maxhits=hparams.max_hits,
        batch_size = hparams.batch_size
    )
    # Create a logger
    wandb_logger = WandbLogger(
        project=hparams.wandb_project,
        name=hparams.run_name,    # label shown in the W&B run list
        config=vars(hparams),     # logs all CLI args as hyperparameters
    )

    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=hparams.patience,
        min_delta=0.00,
        verbose=True,
        mode='max'
    )

    # Create checkpoint callback to save on minimum validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        dirpath='checkpoints/',
        filename='TrackT-Baseline-{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}',
        save_top_k=1,
        mode='max'
    )

    # Create a trainer with tensorboard logging, early stopping, and checkpoint saving
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator=hparams.accelerator, 
        devices=hparams.devices,
        precision="32-true",
        gradient_clip_val=0.5 #Debugging
    )

    # Train the model
    model = LightningNeuralNetwork(
        feature_dim=3, 
        hidden_size=hparams.hidden_size,
        num_heads=hparams.nhead,
        num_encoder_layers=hparams.layers,
        output_size=4,
        learning_rate=hparams.lr
    )
    print("Starting training on TrackT...")
    trainer.fit(model, data_module)

    #testing
    best_model = LightningNeuralNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # This runs the test_step we just added
    test_results = trainer.test(best_model, data_module)
    print(test_results)

    # Push test metrics to W&B (test_step metrics aren't logged automatically)
    wandb_logger.experiment.log({
        'test_auc':  test_results[0].get('test_auc'),
        'test_loss': test_results[0].get('test_loss'),
    })

    # --- Plot the Confusion Matrix ---
    cm = best_model.final_cm
    if cm is None:
        print("WARNING: final_cm is None — confusion matrix was not populated during test.")
    else:
        class_names = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='viridis',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.title('Final Physics Classification Accuracy (%)')
        plt.tight_layout()
        save_path = 'test_confusion_matrix.png'
        plt.savefig(save_path, dpi=150)
        plt.close()  # Free memory; plt.show() does nothing on SLURM (no display)
        print(f"Confusion matrix saved to: {os.path.abspath(save_path)}")

        wandb_logger.experiment.log({
            'confusion_matrix': wandb.Image(save_path)
        })
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Execution Args
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Physics/Data Args
    parser.add_argument("--num_events", type=int, default=0)
    parser.add_argument("--num_events_list", type=int, default=[0])
    parser.add_argument("--purity", type=float, default=0)
    parser.add_argument("--max_hits", type=int, default=17000)
    
    
    # Model Architecture Args
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)

    # W&B
    parser.add_argument("--wandb_project",  default="ColliderML-GroupB")
    parser.add_argument("--run_name",       default=None,
                        help="Name for this W&B run (e.g. 'baseline_purity2'). "
                             "Leave blank for W&B to auto-generate one.")
 

    args = parser.parse_args()
    main(args)