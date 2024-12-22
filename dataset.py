import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        #dong: ds is the datsaet opus_books downloaded from hugging face
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        #dong: it doesn't matter to use tokenizer_tgt or tokenizer_src cuz both contains SOS, use long int
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        #dong: including split a sentence into words, and then map each word into a token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        #dong: how many padding tokens are needed to reach seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        # Purpose: The <s> token is added to the beginning of the decoder input to signal the start of the sequence generation process. It acts as a prompt for the model to begin generating the target sequence.
        # Why Not </s>: The </s> token is not added to the decoder input because the model is supposed to generate this token itself when it predicts the end of the sequence.
        #Including it in the input would defeat the purpose of training the model to learn when to stop generating.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        #dong: or called `target` (what we expect as output from the decoder)
        #Purpose: The </s> token is added to the label to indicate the end of the sequence. During training, the model learns to predict this token as the final output, signaling that the sequence is complete.
        #Why Not <s>: The <s> token is not needed in the label because the model does not need to predict the start of the sequence;
        # it only needs to predict the continuation and termination of the sequence.
        # example: say </s> is 2, seq_len = 7, pad = 0, then label = tensor([301, 302, 303,   2,   0,   0,   0], dtype=torch.int64)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # dong: `encode_mask` is to mask out the padding tokens from participating self-attention
            # (1, 1, seq_len) it can be broadcasted correctly across the batch and attention heads during
            # the computation of self-attention
            # Example encoder input with padding (assuming pad_token ID is 0)
            # encoder_input = torch.tensor([101, 102, 103, 0, 0, 0])  # Example token IDs
            # Assume pad_token is 0
            # pad_token = 0
            # Create the encoder mask
            # encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
            # tensor([[[1, 1, 1, 0, 0, 0]]], dtype=torch.int32)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) #dong: first 1 broadcast to h, second 1 to seq_len, will be expanded to (batch, h, seq_len, seq_len)
            # dong: `decoder_mask` is a causal mask, each word can only look at the words (non-padding) before it
            # decoder mask size: (1, 1, seq_len, seq_len): here we use two `seq_len` to create a causal mask
            # padding mask: tensor([[1, 1, 1, 0, 0]], dtype=torch.int32)
            # causal mask:
            # tensor([[[1, 0, 0, 0, 0],
            #         [1, 1, 0, 0, 0],
            #         [1, 1, 1, 0, 0],
            #         [1, 1, 1, 1, 0],
            #         [1, 1, 1, 1, 1]]], dtype=torch.int32)
            # decoder mask: The decoder_mask ensures that each position can only attend to itself and previous positions, and it ignores padding tokens.
            # tensor([[[1, 0, 0, 0, 0],
            #         [1, 1, 0, 0, 0],
            #         [1, 1, 1, 0, 0],
            #         [0, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0]]], dtype=torch.int32)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            # dong: below for visualization
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    #dong: make all values above diagonal to be 0
    # suppose size = 5, mask would look like this:
    # tensor([[[0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1],
    #         [0, 0, 0, 1, 1],
    #         [0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0]]], dtype=torch.int32)
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # below line of code will make all values above diag 0s
    return mask == 0