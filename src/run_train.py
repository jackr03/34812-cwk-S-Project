from transformers import AutoTokenizer
import torch
from src.config import CONFIG
from src.train_config import TRAIN_CONFIG
from src.data.dataset import create_dataloader
from src.models.TransformerClassifier import TransformerClassifier
from src.train.trainer import Trainer
from src.train.utils import get_device, set_seed


def main():
    #TODO: What is this config seed
    set_seed(CONFIG.seed)

    device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
    )
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.transformer_model)

    print("Loading and tokenizing training data...")
    train_loader = create_dataloader(
        csv_path='data/train.csv',
        tokenizer=tokenizer,
        max_length=TRAIN_CONFIG.max_length,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=True,
    )

    print("Loading and tokenizing dev data...")
    dev_loader = create_dataloader(
        csv_path='data/dev.csv',
        tokenizer=tokenizer,
        max_length=TRAIN_CONFIG.max_length,
        batch_size=TRAIN_CONFIG.batch_size,
        shuffle=False,
    )

    model = TransformerClassifier(
        num_labels=TRAIN_CONFIG.num_labels,
        dropout=TRAIN_CONFIG.dropout,
    ).float().to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=TRAIN_CONFIG,
        device=device,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")


if __name__ == '__main__':
    main()
