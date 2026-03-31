from torchtext.vocab import GloVe

from src.config import CONFIG


class LSTMTokeniser:
    def __init__(self):
        self.glove = GloVe(name='840B', dim=CONFIG.lstm.embedding_dim)

    def encode(self, text: str, max_length: int) -> tuple[list[str], list[int]]:
        """Split, tokenise and then performing padding/truncation before returning (ids, mask)."""
        # Manually offset, reserve 0 for <pad> and 1 for <unk>
        tokens = text.lower().split()
        ids = [self.glove.stoi[token] + 2 if token in self.glove.stoi else 1 for token in tokens]
        ids = ids[:max_length]
        padding_required = max_length - len(ids)
        mask = [1] * len(ids) + [0] * padding_required
        ids = ids + [0] * padding_required
        return ids, mask
