from common_imports import *
from argparse import ArgumentParser
from NeuralNetwork import LightningNeuralNetwork
from Data.DataPrepare import prepare_it_all
import io
# Tensorboard logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def main(hparams):
    # Data Preparation
    # We prepare the jagged data lists and wrap them in the DataModule
    print("Preparing jagged physics data...")
    # Adjust events/purity/maxhits based on your GPU capacity
    
    data_module = prepare_it_all(
        events=range(hparams.num_events), 
        purity_scale=hparams.purity, 
        maxhits=hparams.max_hits,
        batch_size = hparams.batch_size
    )
    # Create a logger
    logger = TensorBoardLogger("lightning_logs", name="TrackT")

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
        filename='TrackT-{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}',
        save_top_k=1,
        mode='max'
    )

    # Create a trainer with tensorboard logging, early stopping, and checkpoint saving
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator=hparams.accelerator, 
        devices=hparams.devices,
        precision="16-mixed"
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

    # --- New: Plot the Results ---
    cm = best_model.final_cm
    class_names = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']
    
    # Normalize for the plot
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    #df_perc = pd.DataFrame(cm_perc, index=class_names, columns=class_names)
    #df_perc.to_csv('results_efficiencies.csv')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='viridis', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Final Physics Classification Accuracy (%)')
    plt.savefig('test_confusion_matrix.png')
    plt.show()

    #print("\n--- Results Exported ---")
    #print("Efficiencies saved to: results_efficiencies.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Execution Args
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Physics/Data Args
    parser.add_argument("--num_events", type=int, default=5000)
    parser.add_argument("--purity", type=float, default=10.0)
    parser.add_argument("--max_hits", type=int, default=17000)
    
    
    # Model Architecture Args
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args()
    main(args)