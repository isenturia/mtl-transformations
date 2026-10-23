# Extracts the encoder's final hidden state for every sentence in the .test set
# and saves a CSV in the same format as kennlw's pass_quest_H.csv
# (input, target, pred, context). Also prints accuracy on the .test and .gen sets.
#
# Usage: python extract.py <model folder> <dataset>
# e.g.:  python extract.py models/passive_GRU_0_0.001_256_0 passive

import sys

import torch
import pandas as pd

from models import EncoderRNN, DecoderRNN
from evaluation import evaluate, logits_to_sentence

model_dir = sys.argv[1]
prefix = sys.argv[2]
hidden_size = 256
MAX_LENGTH = 30

# Vocabulary: word <-> index
word2index = {}
index2word = {}
for line in open("index.txt", "r"):
    parts = line.strip().split("\t")
    word2index[parts[0]] = int(parts[1])
    index2word[int(parts[1])] = parts[0]

# Rebuild the model the same way seq2seq.py does, then load the trained weights.
# Run this on a CPU node: models.py moves tensors to the GPU whenever one exists.
encoder = EncoderRNN(len(word2index), hidden_size, "GRU", max_length=MAX_LENGTH)
decoder = DecoderRNN(hidden_size, len(word2index), "GRU", attn=0, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)
encoder.load_state_dict(torch.load(model_dir + "/" + prefix + ".encoder.0.0.0", map_location="cpu"))
decoder.load_state_dict(torch.load(model_dir + "/" + prefix + ".decoder.0.0.0", map_location="cpu"))

# Turns off dropout, so the same sentence always gives the same prediction
encoder.eval()
decoder.eval()

def run_file(filename):
    rows = []
    for line in open(filename, "r"):
        parts = line.strip().split("\t")
        s1 = parts[0].strip().lower()
        s2 = parts[1].strip().lower()

        # One sentence per batch, shaped seq_length x 1 as in seq2seq.py
        words1 = torch.LongTensor([word2index[word] for word in s1.split()]).view(-1, 1)
        words2 = torch.LongTensor([word2index[word] for word in s2.split()]).view(-1, 1)
        batch = [words1, words2]

        encoder_output, encoder_hidden, encoder_outputs = encoder(batch)
        pred = logits_to_sentence(evaluate(encoder, decoder, batch), index2word)[0]

        # encoder_hidden is 1 x 1 x 256; unbind gives the 1 x 256 vector for this
        # sentence, printed the same way as in kennlw's CSV
        context = torch.unbind(encoder_hidden, dim=1)[0]

        rows.append({"input": s1, "target": s2, "pred": pred, "context": str(context)})
    return pd.DataFrame(rows)

test_df = run_file("data/" + prefix + ".test")
gen_df = run_file("data/" + prefix + ".gen")

print("Test accuracy:", (test_df["pred"] == test_df["target"]).mean())
print("Gen accuracy:", (gen_df["pred"] == gen_df["target"]).mean())

out_file = model_dir + "/" + prefix + "_test_context.csv"
test_df.to_csv(out_file, index=False)
print("Saved", len(test_df), "rows to", out_file)
