import numpy as np
import pandas as pd
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = "./ESPF/protein_codes_uniprot.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator="")
sub_csv = pd.read_csv("./ESPF/subword_units_map_uniprot.csv")

idx2word_p = sub_csv["index"].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = "./ESPF/drug_codes_chembl.txt"
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator="")
sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl.csv")

idx2word_d = sub_csv["index"].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545


def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), "constant", constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask), " ".join(t1)


def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), "constant", constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask), " ".join(t1)


class BinDataset(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        d = self.df.iloc[index]["SMILES"]
        p = self.df.iloc[index]["Target Sequence"]

        d_v, input_mask_d, d_t = drug2emb_encoder(d)
        p_v, input_mask_p, p_t = protein2emb_encoder(p)

        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, d_t, p_t, y


class BinDrugDataset(data.Dataset):
    """Get drug dataset to get protein interaction score with given smiles"""
    def __init__(self, drug_fname, protein):
        "Initialization"

        with open(drug_fname, "r") as f:
            self.smiles = f.read().split("\n")

        self.p_v, self.input_mask_p, self.p_t = protein2emb_encoder(protein)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.smiles)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        d = self.smiles[index]

        d_v, input_mask_d, d_t = drug2emb_encoder(d)
        p_v, input_mask_p, p_t = self.p_v, self.input_mask_p, self.p_t

        return d_v, p_v, input_mask_d, input_mask_p, d_t, p_t, d


class BinProteinDataset(data.Dataset):
    """Get protein dataset to get drug interaction score with given amino sequence"""
    def __init__(self, protein_fname, drug):
        "Initialization"

        with open(protein_fname, "r") as f:
            self.prots = f.read().split("\n")

        self.d_v, self.input_mask_d = drug2emb_encoder(drug)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.smiles)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        p = self.prots[index]

        d_v, input_mask_d, d_t = self.d_v, self.input_mask_d
        p_v, input_mask_p, p_t = protein2emb_encoder(p)

        return d_v, p_v, input_mask_d, input_mask_p, d_t, p_t, p
