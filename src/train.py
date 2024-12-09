import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import create_data_loaders
from model import LargeScaleModel
import torch
import argparse

def train(args):
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.train_path,
        args.val_path,
        args.batch_size,
        args.num_workers
    )
    
    # Initialize model
    model = LargeScaleModel(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accumulate_grad_batches=args.accumulate_grad_batches
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--input_dim', type=int, required=True)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    args = parser.parse_args()
    train(args) 