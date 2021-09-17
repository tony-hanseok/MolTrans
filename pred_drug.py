from argparse import ArgumentParser
import os
import torch
from torch.utils import data
import numpy as np
from config import BIN_config_DBPE
from stream import BinDrugDataset


def pred(data_generator, model):
    model.eval()
    pred_score = []
    smiles_list = []
    for i, (d, p, d_mask, p_mask, smiles) in enumerate(data_generator):
        print(i, len(data_generator))
        score = model(
            d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda()
        )

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        logits = logits.detach().cpu().numpy().tolist()
        
        pred_score.extend(logits)
        smiles_list.extend(smiles)
    return pred_score, smiles_list


if __name__ == "__main__":
    parser = ArgumentParser(description="MolTrans prediction")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 16), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--model_dir",
        "-m",
        default="./output/biosnap_16_1e-05",
        type=str,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--task",
        choices=["biosnap", "bindingdb", "davis"],
        default="biosnap",
        type=str,
        metavar="TASK",
        help="Task name. Could be biosnap, bindingdb and davis.",
    )
    parser.add_argument(
        "--prot_path",
        "-s",
        default="./input/protein/pdk4.txt",
        type=str,
        help="File of amino acid sequence",
    )

    config = BIN_config_DBPE()
    args = parser.parse_args()
    config["batch_size"] = args.batch_size

    if args.task.lower() == "biosnap":
        task_path = "./dataset/BIOSNAP/full_data"
    elif args.task.lower() == "bindingdb":
        task_path = "./dataset/BindingDB"
    elif args.task.lower() == "davis":
        task_path = "./dataset/DAVIS"

    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }
    with open(args.prot_path, "r") as f:
        protein = f.read()
    dataset = BinDrugDataset(
        drug_fname=os.path.join(task_path, "drugs.txt"), protein=protein
    )
    generator = data.DataLoader(dataset, **params)

    model = torch.load(os.path.join(args.model_dir, "model.pt"))
    with torch.set_grad_enabled(False):
        scores, smiles_list = pred(generator, model)

    model_name = os.path.split(args.model_dir)[-1]
    prot_name = os.path.split(args.prot_path)[-1][:-4]
    output_path = args.prot_path[:-4]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, f"{prot_name}_{model_name}.txt"), "w") as f:
        for score, smiles in zip(scores, smiles_list):
            f.write(f"{smiles}\t{score}\n")
