import matplotlib
matplotlib.use('Agg')  # Headless — must come before pyplot on SLURM

from common_imports import *
from argparse import ArgumentParser
from Transformer.NeuralNetwork import LightningNeuralNetwork
from Data.DataPrepare import prepare_it_all
import gc
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------
# Block-Year Format (BYF) Curriculum
#
# Each year has 4 blocks of increasing complexity (purity_scale = raw/track hits).
# Each year also adds more pileup — but resets back to the easiest block (B1)
# before ramping up complexity again.
#
#         Y1     Y2     Y3     Y4
# pileup:  0      1      5     20
# B1:  purity=0   (only hits*)
# B2:  purity=1
# B3:  purity=5
# B4:  purity=10  (most complex)
#
# Checkpoint forwarding:
#   B1 of Y1  → fresh model
#   B2 of Y1  → resumes from last.ckpt of B1/Y1
#   ...
#   B1 of Y2  → resumes from last.ckpt of B4/Y1  ← year boundary
#   B2 of Y2  → resumes from last.ckpt of B1/Y2
#   ...etc
# ---------------------------------------------------------------------------

CURRICULUM = {
    # purity_scale for each block (raw hits added per track-hit)
    'block_purity': [0, 1, 5, 10],
    # pileup level for each year (index 0 = Year 1)
    'year_pileup':  [0, 1, 5, 20],
}


def checkpoint_dir(year: int, block: int) -> str:
    return f'checkpoints/blockyear/year_{year}/block_{block}'


def last_ckpt_path(year: int, block: int) -> str:
    return os.path.join(checkpoint_dir(year, block), 'last.ckpt')


def plot_confusion_matrix(cm, year: int, block: int):
    """Save a normalised confusion matrix for one year/block."""
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
    """AUC-vs-block progress chart across all years."""
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
    # -------------------------------------------------------------------------
    # W&B API key setup — choose ONE of these approaches:
    #
    # (A) RECOMMENDED for SLURM: set in your job script or ~/.bashrc:
    #       export WANDB_API_KEY=your_key_here
    #
    # (B) Programmatic login — uncomment if you can't set env variables:
    #       wandb.login(key="your_key_here")
    # -------------------------------------------------------------------------

    all_metrics = []
    csv_path    = 'blockyear_results.csv'

    block_purity = CURRICULUM['block_purity'][:hparams.blocks]
    year_pileup  = CURRICULUM['year_pileup'][:hparams.years]

    for year in range(1, hparams.years + 1):
        pileup = year_pileup[year - 1]
        print(f"\n{'='*60}")
        print(f"  YEAR {year}  |  pileup={pileup}")
        print(f"{'='*60}")

        for block in range(1, hparams.blocks + 1):
            purity   = block_purity[block - 1]
            run_name = f"Y{year}_B{block}_purity{purity}_pileup{pileup}"
            print(f"\n  -- Block {block}: purity_scale={purity}  pileup={pileup} --")

            # -----------------------------------------------------------------
            # Data
            # NOTE: pileup is a separate concept not yet in DataPrepare.
            # Once you add a `pileup` parameter to prepare_it_all / prepare_data
            # to control how many pileup hits are added per event, uncomment it.
            # -----------------------------------------------------------------
            data_module = prepare_it_all(
                events       = hparams.num_events,
                purity_scale = purity,
                maxhits      = hparams.max_hits,
                batch_size   = hparams.batch_size,
                # pileup     = pileup,   # ← uncomment once DataPrepare supports it
            )
            data_module.setup()

            # -----------------------------------------------------------------
            # Model: fresh on Y1B1, otherwise resume from previous block/year
            # -----------------------------------------------------------------
            if year == 1 and block == 1:
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
                # Carry weights forward from the previous block (or previous year's last block)
                if block == 1:
                    prev_path  = last_ckpt_path(year - 1, hparams.blocks)
                    prev_label = f"Year {year-1} / Block {hparams.blocks}"
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
                model.learning_rate = hparams.lr   # reset LR for each new block

            # -----------------------------------------------------------------
            # W&B Logger
            # Each block = one W&B run. group= ties them together in the UI
            # so you can compare blocks/years in a single experiment view.
            # -----------------------------------------------------------------
            wandb_logger = WandbLogger(
                project = hparams.wandb_project,
                name    = run_name,
                group   = f"BYF_{hparams.run_tag}",   # all blocks/years grouped
                tags    = [
                    f"year_{year}", f"block_{block}",
                    f"purity_{purity}", f"pileup_{pileup}",
                ],
                config  = {
                    'year': year, 'block': block,
                    'purity_scale': purity, 'pileup': pileup,
                    'hidden_size': hparams.hidden_size,
                    'num_layers': hparams.layers,
                    'lr': hparams.lr,
                },
                reinit  = True,   # required when creating multiple loggers in one process
            )

            # -----------------------------------------------------------------
            # Callbacks
            # -----------------------------------------------------------------
            ckpt_callback = ModelCheckpoint(
                monitor    = 'val_auc',
                dirpath    = checkpoint_dir(year, block),
                filename   = f'TrackT-Y{year}B{block}' + '-{epoch:02d}-{val_auc:.4f}',
                save_top_k = 1,
                mode       = 'max',
                save_last  = True,   # last.ckpt is what the next block resumes from
            )
            early_stop = EarlyStopping(
                monitor   = 'val_auc',
                patience  = hparams.patience,
                min_delta = 0.0,
                verbose   = True,
                mode      = 'max',
            )

            # -----------------------------------------------------------------
            # Trainer
            # -----------------------------------------------------------------
            trainer = pl.Trainer(
                max_epochs        = hparams.max_epochs,
                logger            = wandb_logger,
                callbacks         = [early_stop, ckpt_callback],
                accelerator       = hparams.accelerator,
                devices           = hparams.devices,
                precision         = '32-true',
                gradient_clip_val = 0.5,
            )

            trainer.fit(model, data_module)

            # -----------------------------------------------------------------
            # Test on best checkpoint
            # -----------------------------------------------------------------
            best_model   = LightningNeuralNetwork.load_from_checkpoint(
                ckpt_callback.best_model_path
            )
            test_results = trainer.test(best_model, data_module)
            print(f"  Test results: {test_results[0]}")

            # Push test metrics to this block's W&B run
            wandb_logger.experiment.log({
                'test_auc':  test_results[0].get('test_auc'),
                'test_loss': test_results[0].get('test_loss'),
            })

            # Confusion matrix
            plot_confusion_matrix(best_model.final_cm, year, block)

            # -----------------------------------------------------------------
            # Save progress (incremental — safe against SLURM job kills)
            # -----------------------------------------------------------------
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

            # Close this block's W&B run before opening the next one
            wandb.finish()

            # Free VRAM
            del trainer, data_module, best_model, model
            gc.collect()
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    plot_curriculum_summary(csv_path)
    print(f"\nAll done. Results CSV → {os.path.abspath(csv_path)}")


if __name__ == '__main__':
    parser = ArgumentParser()

    # Execution
    parser.add_argument('--accelerator',   default='gpu')
    parser.add_argument('--devices',       type=int,   default=1)
    parser.add_argument('--max_epochs',    type=int,   default=500)
    parser.add_argument('--batch_size',    type=int,   default=32)

    # Physics / Data
    parser.add_argument('--num_events',    type=int,   default=5000)
    parser.add_argument('--max_hits',      type=int,   default=17000)

    # Model
    parser.add_argument('--hidden_size',   type=int,   default=256)
    parser.add_argument('--nhead',         type=int,   default=8)
    parser.add_argument('--layers',        type=int,   default=6)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--patience',      type=int,   default=50)

    # Curriculum
    parser.add_argument('--years',         type=int,   default=4)
    parser.add_argument('--blocks',        type=int,   default=4)

    # W&B
    parser.add_argument('--wandb_project', default='ColliderML-GroupB')
    parser.add_argument('--run_tag',       default='BYF',
                        help='Groups all blocks/years under one experiment in W&B')

    args = parser.parse_args()
    main(args)