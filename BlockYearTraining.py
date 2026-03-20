import matplotlib
matplotlib.use('Agg')  # Headless — must come before pyplot on SLURM

from common_imports import *
from argparse import ArgumentParser
from Transformer.NeuralNetwork_Flash import LightningNeuralNetwork
from Data.DataPrepare import prepare_it_all
import gc
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from Data.DataModule import DataLoad

CURRICULUM = {
    'block_purity': [0, 25, 50, 75, 100],
    'batch_size': [128, 64, 64, 32, 16],
    'epochs': [50, 20, 15, 10 , 7],
    'year_pileup':  [0, 0, 0, 0],   # placeholder — all zeros until pileup is implemented
}


def checkpoint_dir(year: int, block: int) -> str:
    return f'checkpoints/blockyear/block_{block}'


def last_ckpt_path(year: int, block: int) -> str:
    return os.path.join(checkpoint_dir(year, block), 'last.ckpt')


def plot_confusion_matrix(cm, year: int, block: int):
    if cm is None:
        print(f"  WARNING: final_cm is None for Y{year}B{block} — skipping plot.")
        return
    class_names = ['TTBar', 'GGF', 'Dihiggs', 'H-Portal']
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title(f'Year {year} / Block {block}  —  Classification Accuracy (%)')
    plt.tight_layout()
    save_path = f'confusion_matrix_Y{year}B{block}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {os.path.abspath(save_path)}")


def plot_curriculum_summary(csv_path: str):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, year in enumerate(sorted(df['year'].unique())):
        sub = df[df['year'] == year].sort_values('block')
        label = f"Year {year} (pileup={sub['pileup'].iloc[0]})"
        ax.plot(sub['block_label'], sub['test_auc'],
                marker='o', linewidth=2, color=colors[i], label=label)

    ax.set_xlabel('Block')
    ax.set_ylabel('Test AUC')
    ax.set_title('BYF Curriculum: AUC across Years and Blocks')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('curriculum_summary.png', dpi=150)
    plt.close()
    print(f"Summary plot saved → {os.path.abspath('curriculum_summary.png')}")


def main(hparams):
    # Strip SLURM vars so Lightning uses multiprocessing.spawn (needed for JupyterHub)
    import socket
    os.environ.pop("SLURM_NTASKS",    None)
    os.environ.pop("SLURM_JOB_NAME", None)
    def _free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    os.environ["MASTER_PORT"] = str(_free_port())

    all_metrics = []
    csv_path    = 'blockyear_results.csv'

    block_purity = CURRICULUM['block_purity'][:hparams.blocks]
    pileup       = CURRICULUM['year_pileup'][0]
    batch_size   = CURRICULUM['batch_size'][:hparams.blocks]
    epochs       = CURRICULUM['epochs'][:hparams.blocks]
    year         = 0

    for block in range(1, hparams.blocks + 1):
        purity   = block_purity[block - 1]
        batchsize = batch_size[block - 1]
        maxepochs = epochs[block - 1]
        run_name = f"Y{year}_B{block}_purity{purity}_pileup{pileup}"
        print(f"\n  -- Block {block}: purity_scale={purity} --")

        # Data
        if hparams.data_dir is not None:
            data_module = DataLoad(f"{hparams.data_dir}/P{purity}",batchsize, mode = "flash")
        else:
            from Data.DataModule import PackedDataModule
            _dm = prepare_it_all(
                events       = hparams.num_events,
                purity_scale = purity,
                maxhits      = hparams.max_hits,
                batch_size   = batchsize,
            )
            # prepare_it_all returns PaddedDataModule — swap to PackedDataModule for flash
            data_module = PackedDataModule(
                _dm.X_train, _dm.y_train,
                _dm.X_val,   _dm.y_val,
                _dm.X_test,  _dm.y_test,
                batch_size   = batchsize,
            )
        data_module.setup()

        # Model
        if block == 1:
            print("  Initialising fresh model.")
            model = LightningNeuralNetwork(
                feature_dim        = 3,
                hidden_size        = hparams.hidden_size,
                num_heads          = hparams.nhead,
                num_encoder_layers = hparams.layers,
                output_size        = 4,
                learning_rate      = hparams.lr,
            )
        else:
            prev_path  = last_ckpt_path(year, block - 1)
            prev_label = f"Year {year} / Block {block-1}"
            if not os.path.exists(prev_path):
                raise FileNotFoundError(
                    f"Cannot resume — checkpoint missing: {prev_path}\n"
                    f"Expected last.ckpt from {prev_label}"
                )
            print(f"  Resuming from {prev_label}: {prev_path}")
            model = LightningNeuralNetwork.load_from_checkpoint(prev_path)
            model.learning_rate = hparams.lr

        # W&B
        wandb_logger = WandbLogger(
            project = hparams.wandb_project,
            name    = run_name,
            group   = "BYF",
            tags    = [f"year_{year}", f"block_{block}", f"purity_{purity}"],
            config  = {
                'year': year, 'block': block,
                'purity_scale': purity,
                'hidden_size': hparams.hidden_size,
                'num_layers': hparams.layers,
                'lr': hparams.lr,
            },
            #reinit  = True,
            reinit='finish_previous',
        )

        # Callbacks
        ckpt_callback = ModelCheckpoint(
            monitor    = 'val_auc',
            dirpath    = checkpoint_dir(year, block),
            filename   = f'TrackT-Baseline-Y{year}B{block}' + '-{epoch:02d}-{val_auc:.4f}',
            save_top_k = 1,
            mode       = 'max',
            save_last  = True,
        )
        early_stop = EarlyStopping(
            monitor   = 'val_auc',
            patience  = hparams.patience,
            min_delta = 0.0,
            verbose   = True,
            mode      = 'max',
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs        = maxepochs,
            logger            = wandb_logger,
            callbacks         = [early_stop, ckpt_callback],
            accelerator       = hparams.accelerator,
            devices           = hparams.devices,
            strategy          = "ddp_notebook" if hparams.devices > 1 else "auto",
            precision         = '32-true',
            gradient_clip_val = 0.5,
        )

        trainer.fit(model, data_module)

        # Test
        best_model   = LightningNeuralNetwork.load_from_checkpoint(
            ckpt_callback.best_model_path
        )
        test_results = trainer.test(best_model, data_module)

        # Post-test: rank 0 only — avoids duplicate W&B logs and file races
        if trainer.is_global_zero:
            print(f"  Test results: {test_results[0]}")

            wandb_logger.experiment.log({
                'test_auc':  test_results[0].get('test_auc'),
                'test_loss': test_results[0].get('test_loss'),
            })

            plot_confusion_matrix(best_model.final_cm, year, block)
            if best_model.final_cm is not None:
                wandb_logger.experiment.log({
                    'confusion_matrix': wandb.Image(f'confusion_matrix_Y{year}B{block}.png')
                })

            row = {
                'year':         year,
                'block':        block,
                'block_label':  f'Y{year}B{block}',
                'purity_scale': purity,
                'pileup':       pileup,
                **test_results[0],
            }
            all_metrics.append(row)
            pd.DataFrame(all_metrics).to_csv(csv_path, index=False)

            wandb.finish()

        # Free VRAM — all ranks
        del trainer, data_module, best_model, model
        gc.collect()
        torch.cuda.empty_cache()

    if os.environ.get('LOCAL_RANK', '0') == '0':
        plot_curriculum_summary(csv_path)
        print(f"\nAll done. Results CSV → {os.path.abspath(csv_path)}")


if __name__ == '__main__':
    parser = ArgumentParser()

    # Execution
    parser.add_argument('--accelerator',    default='gpu')
    parser.add_argument('--devices',        type=int,   default=4)
    parser.add_argument('--batch_size',     type=int,   default=128)

    # Physics / Data
    parser.add_argument('--data_dir',       type=str,   default=None)
    parser.add_argument('--num_events',     type=int,   default=5000)
    parser.add_argument('--max_hits',       type=int,   default=17000)

    # Model
    parser.add_argument('--hidden_size',    type=int,   default=256)
    parser.add_argument('--nhead',          type=int,   default=8)
    parser.add_argument('--layers',         type=int,   default=6)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--patience',       type=int,   default=25)

    # Curriculum
    parser.add_argument('--years',          type=int,   default=0)
    parser.add_argument('--blocks',         type=int,   default=5)

    # W&B
    parser.add_argument('--wandb_project',  default='ColliderML-GroupB')
    parser.add_argument('--run_tag',        default='BYF_test')

    args = parser.parse_args()
    main(args)