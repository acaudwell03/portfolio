# Built-in libraries
import re
from collections import Counter

# Data manipulation & visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# NLTK
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Scikit-learn
from sklearn.metrics import f1_score


# PyTorch
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Docstrings modified from OpenAI. (2025). ChatGPT [Large language model]. https://chat.openai.com/chat


class TextPreprocessor:
    def __init__(self, min_freq=2, max_len=50, min_kernel_size=3, extra_stopwords=None):
        """Initialize text preprocessing utilities and vocabulary containers.

        Args:
            min_freq (int): Minimum token frequency to include in the vocabulary.
            max_len (int): Maximum sequence length when encoding texts.
            min_kernel_size (int): Minimum sequence length to satisfy CNN kernel size; sequences
                shorter than this are left-padded with `<pad>` ids.
            extra_stopwords (Iterable[str] | None): Optional additional stopwords to extend NLTK's list.
        """
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)
        self.min_kernel_size = min_kernel_size
        self.min_freq = min_freq
        self.max_len = max_len
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.word2idx = {"<pad>": 0, "<unk>": 1}

    def preprocess(self, text):
        """Normalize and tokenize a raw string into a space-joined, stemmed token sequence.

        Processing steps:
        - Remove non-letter characters
        - Lowercase
        - Remove emojis and pictographs
        - Tokenize by whitespace
        - Remove stopwords
        - Porter-stem remaining tokens

        Args:
            text (str): Input raw text.

        Returns:
            str: Space-separated string of processed, stemmed tokens.
        """
        # Clean text: keep only letters, lowercase
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()

        # Emoji pattern
        emoji_pattern = re.compile(
            r'['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags
            ']+', flags=re.UNICODE
        )

        text = emoji_pattern.sub(r'', text)

        # Tokenize
        tokens = text.split()

        # Stem and remove stopwords
        tokens = [self.stemmer.stem(w)
                  for w in tokens if w not in self.stopwords]

        return ' '.join(tokens)

    def build_vocab(self, texts):
        """Build the integer token vocabulary from an iterable of raw texts.

        Tokens whose frequency is at least `min_freq` are added to the vocabulary.

        Args:
            texts (Iterable[str]): Collection of raw text strings.

        Side Effects:
            Populates/extends `word2idx` and `idx2word` mappings.
        """
        freq = Counter()
        for text in texts:
            tokens = self.preprocess(text).split()
            freq.update(tokens)

        idx = 2  # 0 = pad, 1 = unk
        for word, count in freq.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode_batch(self, texts):
        """Encode a batch of raw texts into a left-padded tensor of token ids.

        Each text is preprocessed, mapped to ids with `<unk>` fallback, truncated to `max_len`,
        and then left-padded to at least `min_kernel_size`. Finally, sequences are padded to the
        maximum length in the batch with the `<pad>` id.

        Args:
            texts (Iterable[str]): Batch of raw text strings.

        Returns:
            torch.Tensor: 2D tensor of shape `(batch_size, max_seq_len_in_batch)` containing token ids.
        """
        sequences = []
        for text in texts:
            tokens = self.preprocess(text).split()
            ids = [self.word2idx.get(token, self.word2idx["<unk>"])
                   for token in tokens]
            ids = ids[:self.max_len]

            if len(ids) < self.min_kernel_size:
                ids += [self.word2idx["<pad>"]] * \
                    (self.min_kernel_size - len(ids))
            sequences.append(torch.tensor(ids, dtype=torch.long))

        padded = pad_sequence(
            sequences, batch_first=True, padding_value=self.word2idx["<pad>"]
        )
        return padded


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        """Torch dataset wrapping encoded texts and labels.

        Args:
            texts (torch.Tensor): Long tensor of shape `(num_examples, seq_len)` with token ids.
            labels (Sequence | np.ndarray): Labels corresponding to each example. For multi-label
                classification, this should be a binary matrix-like structure.
        """
        self.texts = texts.clone().detach().long()
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get the `(text, label)` pair at the provided index.

        Args:
            idx (int): Dataset index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded text tensor and its label tensor.
        """
        return self.texts[idx], self.labels[idx]


class MyCnnFunctions():
    def __init__(self, model, device, multi_label=True):
        """Utility class for training, evaluating, and visualizing CNN models.

        Args:
            model (torch.nn.Module): The neural network to train/evaluate.
            device (torch.device | str): Device on which to run computations.
            multi_label (bool): If True, treats the task as multi-label with sigmoid outputs;
                otherwise multi-class with softmax/argmax metrics.
        """
        self.model = model
        self.device = device
        self.multi_label = multi_label

    def _tune_thresholds(self, probs, targets, low=0.1, high=0.9, steps=33):
        """Grid-search per-class probability thresholds maximizing F1.

        Args:
            probs (np.ndarray): Array of shape `(num_examples, num_classes)` with predicted
                probabilities (post-sigmoid) for each class.
            targets (np.ndarray): Binary ground-truth array with the same shape as `probs`.
            low (float): Lower bound of threshold search interval (inclusive).
            high (float): Upper bound of threshold search interval (inclusive).
            steps (int): Number of evenly spaced thresholds between `low` and `high`.

        Returns:
            np.ndarray: Vector of shape `(num_classes,)` with the best threshold per class.
        """
        best_thresholds = []
        for i in range(probs.shape[1]):
            best_f1 = 0.0
            best_t = 0.5
            for t in np.linspace(low, high, steps):
                pred_i = (probs[:, i] >= t).astype(int)
                f1_i = f1_score(targets[:, i], pred_i, zero_division=0)
                if f1_i > best_f1:
                    best_f1 = f1_i
                    best_t = t
            best_thresholds.append(best_t)
        return np.array(best_thresholds)

    def model_trainer(self, epochs, train_loader, val_loader, loss_func, optimizer,
                      patience=5, scheduler=None, min_epochs_before_stop=2):
        """Train the model and track metrics with optional early stopping and scheduling.

        In multi-label mode, sigmoid probabilities are used and a micro-averaged F1 is computed.
        In multi-class mode, accuracy is used as the validation metric.

        Args:
            epochs (int): Maximum number of epochs to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            loss_func (Callable): Loss function taking `(preds, targets)`.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            patience (int): Early stopping patience measured in epochs without improvement.
            scheduler (torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau | None):
                Optional learning rate scheduler. If `ReduceLROnPlateau`, stepped with val loss.
            min_epochs_before_stop (int): Minimum number of epochs to complete before early stopping.

        Returns:
            dict: Training summary containing keys:
                - `train_losses` (List[float])
                - `train_accuracies` (List[float])
                - `val_losses` (List[float])
                - `val_f1s` (List[float])  (or val accuracy if `multi_label` is False)
                - `best_thresholds` (np.ndarray | None)
                - `best_val_f1` (float)
        """
        train_losses, train_accuracies = [], []
        val_losses, val_f1s = [], []

        best_val_f1 = 0.0
        best_thresholds = None
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.model.train()
            correct, total, total_loss = 0.0, 0.0, 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).float()
                optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = loss_func(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)

                if self.multi_label:
                    probs = torch.sigmoid(preds)
                    predicted = (probs > 0.5).float()
                    correct += (predicted == y_batch).float().sum().item()
                    total += y_batch.numel()
                else:
                    _, predicted = torch.max(preds, 1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            train_loss = total_loss / len(train_loader.dataset)
            train_acc = correct / total if total > 0 else 0.0
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Validation
            self.model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device).float()
                    outputs = self.model(x_val)
                    loss = loss_func(outputs, y_val)
                    val_loss += loss.item() * x_val.size(0)

                    if self.multi_label:
                        probs = torch.sigmoid(outputs)
                        all_probs.append(probs.cpu())
                        all_labels.append(y_val.cpu())
                    else:
                        all_probs.append(outputs.cpu())
                        all_labels.append(y_val.cpu())

            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            if self.multi_label:
                all_probs = torch.cat(all_probs).numpy()
                all_labels = torch.cat(all_labels).numpy()
                thresholds = self._tune_thresholds(all_probs, all_labels)
                preds_tuned = (all_probs >= thresholds[None, :]).astype(int)
                val_f1 = f1_score(all_labels, preds_tuned,
                                  average='micro', zero_division=0)
                val_f1s.append(val_f1)
                metric_str = f"Val Micro F1: {val_f1:.4f}"
            else:
                all_preds = torch.cat(all_probs).numpy()
                all_labels = torch.cat(all_labels).numpy()
                val_acc = (np.argmax(all_preds, axis=1) == all_labels).mean()
                val_f1s.append(val_acc)
                val_f1 = val_acc
                metric_str = f"Val Acc: {val_acc:.4f}"
                thresholds = None

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            improved = val_f1 > best_val_f1 + 1e-5
            if improved:
                best_val_f1 = val_f1
                best_thresholds = thresholds
                best_state = self.model.state_dict()
                wait = 0
            else:
                wait += 1

            print(f"Epoch {epoch:<3} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | {metric_str} | Wait: {wait}")

            if epoch >= min_epochs_before_stop and wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_f1s": val_f1s,
            "best_thresholds": best_thresholds,
            "best_val_f1": best_val_f1
        }

    def evaluate(self, dataloader):
        """Compute probabilities and collect true labels over a dataset.

        Note: Uses sigmoid on logits; intended for multi-label evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader): Batches of `(inputs, labels)`.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Predicted probabilities of shape `(num_examples, num_classes)`.
                - Ground-truth labels of the same shape.
        """
        self.model.eval()
        y_probas_all = []
        y_true_all = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                probs = torch.sigmoid(logits)
                y_probas_all.append(probs.cpu().numpy())
                y_true_all.append(y_batch.cpu().numpy())

        y_probas_all = np.vstack(y_probas_all)
        y_true_all = np.vstack(y_true_all)
        return y_probas_all, y_true_all

    def plot_training(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training/validation loss and metric curves over epochs.

        Args:
            train_losses (Sequence[float]): Training loss per epoch.
            val_losses (Sequence[float]): Validation loss per epoch.
            train_accs (Sequence[float]): Training metric (accuracy/F1) per epoch.
            val_accs (Sequence[float]): Validation metric (accuracy/F1) per epoch.
        """
        epochs = range(len(train_losses))

        # Create figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Common formatting for both axes
        for ax in axes:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Loss plot
        axes[0].plot(epochs, train_losses, label='Train Loss',
                     color='#1f497d', marker='o', linewidth=2, markersize=5)
        axes[0].plot(epochs, val_losses, label='Validation Loss',
                     color='#d95f02', marker='s', linewidth=2, markersize=5)
        axes[0].set_xlabel('Epochs', fontsize=14, fontname='serif')
        axes[0].set_ylabel('Loss', fontsize=14, fontname='serif')
        axes[0].set_title('Loss over Epochs', fontsize=16,
                          fontweight='bold', fontname='serif')
        axes[0].legend(fontsize=12)

        # Accuracy / F1 plot
        axes[1].plot(epochs, train_accs, label='Train Metric',
                     color='#1f497d', marker='o', linewidth=2, markersize=5)
        axes[1].plot(epochs, val_accs, label='Validation Metric',
                     color='#d95f02', marker='s', linewidth=2, markersize=5)
        axes[1].set_xlabel('Epochs', fontsize=14, fontname='serif')
        axes[1].set_ylabel('Accuracy / F1', fontsize=14, fontname='serif')
        axes[1].set_title('Metric over Epochs', fontsize=16,
                          fontweight='bold', fontname='serif')
        axes[1].legend(fontsize=12)

        plt.tight_layout()
        plt.show()
