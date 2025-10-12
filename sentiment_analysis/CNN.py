import torch
import torch.nn as nn
import torch.nn.functional as F

# Docstring modified from OpenAI. (2025). ChatGPT [Large language model]. https://chat.openai.com/chat


class CNN(nn.Module):
    """
    Convolutional Neural Network for text classification.

    This model uses an embedding layer followed by multiple parallel 1D convolutional 
    layers with varying kernel sizes. Each convolution is followed by global max pooling 
    to capture the most salient features. The pooled features are concatenated, passed 
    through dropout and fully connected layers, and finally mapped to the output classes.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the word embeddings.
        num_classes (int): Number of output classes.
        num_filters (int): Number of filters (feature maps) in each convolutional layer.
        dropout (float): Dropout probability applied after concatenation.
        kernel_sizes (list of int, optional): List of kernel sizes for the convolutional 
            layers. Default is [3, 4, 5].
        fc1_size (int, optional): Number of units in the first fully connected layer.
            Default is 128.

    Attributes:
        embedding (nn.Embedding): Embedding layer mapping tokens to vectors.
        convs (nn.ModuleList): List of 1D convolutional layers with different kernel sizes.
        dropout (nn.Dropout): Dropout layer applied after concatenation.
        fc1 (nn.Linear): Fully connected layer projecting pooled features.
        fc (nn.Linear): Output fully connected layer producing class logits.

    Forward Input:
        x (Tensor): Input tensor containing token IDs.

    Forward Output:
        logits (Tensor): Output tensor of shape (batch_size, num_classes), containing 
        raw class scores (logits).
    """

    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, dropout, kernel_sizes=[3, 4, 5], fc1_size=128):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Create conv layers for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), fc1_size)
        self.fc = nn.Linear(fc1_size, num_classes)

        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.adaptive_max_pool1d(c, 1).squeeze(
                2)
            conv_outputs.append(p)

        cat = torch.cat(conv_outputs, dim=1)

        x = self.dropout(cat)
        x = F.relu(self.fc1(x))
        logits = self.fc(x)

        return logits
