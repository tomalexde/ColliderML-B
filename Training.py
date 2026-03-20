import matplotlib
matplotlib.use('Agg')  # Headless — must come before pyplot on SLURM
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)

from common_imports import *
from argparse import ArgumentParser
from Transformer.NeuralNetwork_Flash import LightningNeuralNetwork as LightningNeuralNetwork_Flash
from Transformer.NeuralNetwork_SDPA import LightningNeuralNetwork as LightningNeuralNetwork_SDPA
from Data.DataPrepare import prepare_it_all, create_complex_dataset
from Data.DataModule import DataLoad
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

torch.set_float32_matmul_precision('high')
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)



def get_strategy(devices: int) -> str:
    """Use ddp_notebook in JupyterHub, ddp in sbatch — detected automatically."""
    if devices <= 1:
        return "auto"
    if os.environ.get("JUPYTERHUB_USER"):
        return "ddp_notebook"
    return "ddp"


def _setup_distributed_env():
    """Strip SLURM vars and find a free port — only needed in JupyterHub."""
    if not os.environ.get("JUPYTERHUB_USER"):
        return   # sbatch handles this itself
    import socket
    os.environ.pop("SLURM_NTASKS",    None)
    os.environ.pop("SLURM_JOB_NAME", None)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])

def main(hparams):
    _setup_distributed_env()
    
    #import socket
    #if not hparams.slurm:
    #    os.environ.pop("SLURM_NTASKS", None)
    #    os.environ.pop("SLURM_JOB_NAME", None)
    #    os.environ["MASTER_ADDR"] = "localhost"
    #    def _free_port():
    #        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #            s.bind(('', 0))
    #            return s.getsockname()[1]
    #    os.environ["MASTER_PORT"] = str(_free_port())
    # -------------------------------------------------------------------------
    # Data — mutually exclusive modes, checked in priority order:
    #   1. --data_file: load pre-saved DataModule from disk
    #   2. --isComplex: multi-process complex dataset
    #   3. --isTracks:  track-only dataset
    #   4. default:     prepare_it_all with purity_scale
    #
    # NOTE: type=bool in argparse does NOT work as expected.
    # `--isComplex True` would still be True because any non-empty string is
    # truthy when cast with bool(). Use --isComplex (store_true) instead.
    # -------------------------------------------------------------------------
    print("Preparing physics data...")
    if hparams.data_file is not None:
        data_module = DataLoad(hparams.data_file, hparams.batch_size, hparams.mode)
    elif hparams.isComplex:
        data_module = create_complex_dataset(
            hparams.purity_c, hparams.events_list_c,
            hparams.id_c, hparams.max_hits, hparams.batch_size
        )
    else:
        if hparams.num_events == 0:
            hparams.num_events = hparams.num_events_list
        from Data.DataModule import PackedDataModule
        _dm = prepare_it_all(
            events=hparams.num_events,
            purity_scale=hparams.purity,
            maxhits=hparams.max_hits,
            batch_size=hparams.batch_size
        )
        if hparams.mode == 'flash':
            data_module = PackedDataModule(
                _dm.X_train, _dm.y_train,
                _dm.X_val,   _dm.y_val,
                _dm.X_test,  _dm.y_test,
                batch_size=hparams.batch_size,
            )
        else:
            data_module = _dm

    # -------------------------------------------------------------------------
    # W&B — API key via env var (export WANDB_API_KEY=...) or `wandb login`
    # Do NOT call wandb.login() in a script — it blocks waiting for input on
    # SLURM nodes and will hang the job.
    # -------------------------------------------------------------------------
    wandb_logger = WandbLogger(
        project=hparams.wandb_project,
        name=hparams.run_name,
        config=vars(hparams),
    )

    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=hparams.patience,
        min_delta=0.00,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        dirpath='checkpoints/p0/',
        filename='TrackT-Baseline-{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}',
        save_top_k=1,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        strategy=get_strategy(hparams.devices),
        log_every_n_steps=1,
        precision="32-true",
        gradient_clip_val=0.5,
    )

    if hparams.mode == "sdpa":
        model = LightningNeuralNetwork_SDPA(
            feature_dim=3,
            hidden_size=hparams.hidden_size,
            num_heads=hparams.nhead,
            num_encoder_layers=hparams.layers,
            output_size=4,
            learning_rate=hparams.lr
        )
    else:
        model = LightningNeuralNetwork_Flash(
            feature_dim=3,
            hidden_size=hparams.hidden_size,
            num_heads=hparams.nhead,
            num_encoder_layers=hparams.layers,
            output_size=4,
            learning_rate=hparams.lr
        )
    print("Starting training on TrackT...")
    trainer.fit(model, data_module)

    # Test on best checkpoint
    NN = LightningNeuralNetwork_Flash if hparams.mode == 'flash' else LightningNeuralNetwork_SDPA
    best_model = NN.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_results = trainer.test(best_model, data_module)

    # Post-test: W&B logging, confusion matrix — rank 0 only to avoid
    # duplicate entries and file write races with DDP
    if trainer.is_global_zero:
        print(test_results)

        wandb_logger.experiment.log({
            'test_auc':  test_results[0].get('test_auc'),
            'test_loss': test_results[0].get('test_loss'),
        })

        cm = best_model.final_cm
        if cm is None:
            print("WARNING: final_cm is None — confusion matrix not populated.")
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
            plt.close()
            print(f"Confusion matrix saved to: {os.path.abspath(save_path)}")

            wandb_logger.experiment.log({
                'confusion_matrix': wandb.Image(save_path)
            })

        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Execution
    parser.add_argument("--accelerator",  default="gpu")
    parser.add_argument("--devices",      type=int,  default=4)
    parser.add_argument("--max_epochs",   type=int,  default=500)
    parser.add_argument("--batch_size",   type=int,  default=128)
    parser.add_argument("--mode",   type=str,  default="flash")
    
    # Data mode flags — use store_true, NOT type=bool (argparse bool bug)
    parser.add_argument("--isComplex",    action="store_true",
                        help="Use create_complex_dataset")
    parser.add_argument("--isTracks",     action="store_true",
                        help="Use prepare_tracks_only")

    # Physics / Data
    parser.add_argument("--data_file",          type=str,   default=None)
    parser.add_argument("--num_events",          type=int,   default=0)
    parser.add_argument("--num_events_list",     type=int,   default=0)
    parser.add_argument("--purity",              type=float, default=0)
    parser.add_argument("--max_hits",            type=int,   default=17000)

    # Complex dataset args
    parser.add_argument("--events_list_c", type=int,  nargs='+', default=[100, 100, 100, 100])
    parser.add_argument("--purity_c",      type=float,nargs='+', default=[0, 0, 0, 0])
    parser.add_argument("--id_c",          type=int,  nargs='+', default=[0, 1, 2, 3])

    # Tracks-only args
    parser.add_argument("--num_events_t",       type=int, default=0)
    parser.add_argument("--num_events_list_t",  type=int, default=0)

    # Model
    parser.add_argument("--hidden_size",  type=int,   default=256)
    parser.add_argument("--nhead",        type=int,   default=8)
    parser.add_argument("--layers",       type=int,   default=6)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=25)

    # W&B
    parser.add_argument("--wandb_project", default="ColliderML-GroupB")
    parser.add_argument("--run_name",      default=None)

    #parser.add_argument('--slurm', action='store_true', default=False)

    args = parser.parse_args()
    main(args)