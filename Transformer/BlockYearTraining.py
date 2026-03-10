from common_imports import *
from argparse import ArgumentParser
from NeuralNetwork import LightningNeuralNetwork
from Data.DataPrepare import prepare_it_all
import io
import gc
# Tensorboard logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

#Importing of models
import glob
import os

def plot_multi_year_progress(csv_path):
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot AUC (Higher is better)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Test AUC', color='tab:blue')
    ax1.plot(df['year'], df['val_auc'], marker='o', color='tab:blue', linewidth=2, label='AUC')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second axis for Pileup
    ax2 = ax1.twinx()
    ax2.set_ylabel('Pileup Scale (Noise)', color='tab:red')
    ax2.bar(df['year'], df['purity'], alpha=0.2, color='tab:red', label='Pileup')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Model Robustness vs. Increasing Pileup')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('learning_progress_summary.png')
    plt.show()

def main(hparams):
    # This list will hold results to be saved to CSV
    all_year_metrics = []
    csv_path = "multi_year_results.csv"

    # Convert years to a list if it's an int (e.g., 4 -> [1, 2, 3, 4])
    years = range(1, hparams.years + 1)

    for year in years:
        # --- Data Preparation ---
        # We prepare the nested data lists and wrap them in the DataModule
        print(f"Preparing nested physics data for year {year}...")
        # Adjust events/purity/maxhits based on your GPU capacity
        
        purity = year
        data_module = prepare_it_all(
            events=range(hparams.num_events), 
            purity_scale=purity, 
            maxhits=hparams.max_hits,
            batch_size = hparams.batch_size
        )
        steps_per_epoch = len(data_module.train_dataloader())

        # Create checkpoint callback to save on minimum validation loss
        checkpoint_callback = ModelCheckpoint(
            monitor='val_auc',
            # Put each year in its own sub-folder
            dirpath=f'checkpoints/blockyear/year_{year}/', 
            filename='TrackT-{epoch:02d}-{val_auc:.4f}',
            save_top_k=1,
            mode='max',
            save_last=True  # This creates: checkpoints/blockyear/year_1/last.ckpt
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

        # Create a trainer with tensorboard logging, early stopping, and checkpoint saving
        trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            logger=logger,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator=hparams.accelerator, 
            devices=hparams.devices,
            precision="16-mixed"
        )

        # --- Import Model ---
        if year == 1:
            print("Starting Year 1: Initializing new model.")
            model = LightningNeuralNetwork(
                feature_dim=3, 
                hidden_size=hparams.hidden_size,
                num_heads=hparams.nhead,
                num_encoder_layers=hparams.layers,
                output_size=4,
                learning_rate=hparams.lr
            )
        else:
            # Get the 'last' state of the previous year
            prev_year_path = f'checkpoints/blockyear/year_{year-1}/last.ckpt'
            
            if os.path.exists(prev_year_path):
                print(f"Resuming from Year {year-1} final state: {prev_year_path}")
                model = LightningNeuralNetwork.load_from_checkpoint(prev_year_path, steps_per_epoch=steps_per_epoch)
            else:
                # Fallback: In case Year 1 crashed before finishing
                raise FileNotFoundError(f"Missing last.ckpt for Year {year-11}")
        # --- Train the model ---
        print(f"Starting year {year} training on TrackT...")
        trainer.fit(model, data_module)

        #testing
        best_model = LightningNeuralNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
        
        # This runs the test_step we just added
        test_results = trainer.test(best_model, data_module)
        print(test_results)

        # -- Collect Metrics ---
        metrics = test_results[0]
        metrics['year'] = year
        metrics['purity'] = year # Tracking the noise level
        all_year_metrics.append(metrics)
        
        # Save CSV incrementally in case of crash
        pd.DataFrame(all_year_metrics).to_csv(csv_path, index=False)

        # --- Plot the Results ---
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
        plt.savefig(f'{year}-test_confusion_matrix.png')
        plt.close()

        # --- Clearing Memory ---
        print(f"Clearing VRAM after Year {year}...")
        del trainer
        del data_module
        del best_model
        del model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    #Print Progress
    plot_multi_year_progress(csv)

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
    parser.add_argument("--max_hits", type=int, default=17000)
    
    
    # Model Architecture Args
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)

    #Block Year
    parser.add_argument("--years", type=int, default=4)

    args = parser.parse_args()
    main(args)