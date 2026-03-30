from torchtext.vocab import GloVe

from src.config import CONFIG


class LSTMTokeniser:
    def __init__(self):
        self.glove = GloVe(name='6B', dim=CONFIG.lstm.embedding_dim)

    def encode(self, text: str, max_length: int) -> tuple[list[str], list[int]]:
        """Split, tokenise and then performing padding/truncation before returning (ids, mask)."""
        tokens = text.lower().split()
        ids = [self.glove.stoi.get(token, 0) for token in tokens]
        ids = ids[:max_length]
        padding_required = max_length - len(ids)
        mask = [1] * len(ids) + [0] * padding_required
        ids = ids + [0] * padding_required
        return ids, mask
