from argparse import ArgumentParser
import copy
from time import time
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    auc,
)
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BinDataset

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_task(task_name):
    if task_name.lower() == "biosnap":
        return "./dataset/BIOSNAP/full_data"
    elif task_name.lower() == "bindingdb":
        return "./dataset/BindingDB"
    elif task_name.lower() == "davis":
        return "./dataset/DAVIS"


def get_dataloaders(args):
    print("--- Data Preparation ---")
    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }
    dataFolder = get_task(args.task)

    df_train = pd.read_csv(dataFolder + "/train.csv")
    df_val = pd.read_csv(dataFolder + "/val.csv")
    df_test = pd.read_csv(dataFolder + "/test.csv")

    training_set = BinDataset(
        df_train.index.values, df_train.Label.values, df_train
    )
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BinDataset(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BinDataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    return training_generator, validation_generator, testing_generator


def get_model(config):
    model = BIN_Interaction_Flat(**config)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, opt


def train(model, opt):

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    print("--- Go for Training ---")
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(
                d=d.long().cuda(),
                p=p.long().cuda(),
                d_mask=d_mask.long().cuda(),
                p_mask=p_mask.long().cuda(),
            )

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 1000 == 0:
                print(
                    f"Training at Epoch {epo + 1} iteration {i} with loss {loss.cpu().detach().numpy()}"
                )

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, *_ = test(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc

            print(
                f"Validation at Epoch {epo+1} , AUROC: {auc} , AUPRC: {auprc} , F1 : {f1}"
            )

    return model_max


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(
            d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda()
        )

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to("cpu").numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print("Confusion Matrix : \n", cm1)
    print("Recall : ", recall_score(y_label, y_pred_s))
    print("Precision : ", precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print("Specificity : ", specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return (
        roc_auc_score(y_label, y_pred),
        average_precision_score(y_label, y_pred),
        f1_score(y_label, outputs),
        y_pred,
        loss.item(),
        y_label,
        y_pred,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="MolTrans Training.")
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
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--task",
        choices=["biosnap", "bindingdb", "davis"],
        default="bindingdb",
        type=str,
        metavar="TASK",
        help="Task name. Could be biosnap, bindingdb and davis.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-6,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--output", "-o", default="output/", type=str, help="Directory to save outputs"
    )

    config = BIN_config_DBPE()
    args = parser.parse_args()
    config["batch_size"] = args.batch_size
    args.output = os.path.join(args.output, f"{args.task}_{args.batch_size}_{args.lr}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    loss_history = []
    training_generator, validation_generator, testing_generator = get_dataloaders(
        args=args
    )
    model, opt = get_model(config=config)
    with torch.set_grad_enabled(False):
        auroc, auprc, f1, logits, loss, *_ = test(
            testing_generator, copy.deepcopy(model)
        )
        print(
            f"Initial Testing AUROC: {auroc} , AUPRC: {auprc}, F1: {f1}, Test loss: {loss}"
        )

    s = time()
    model_max = train(model=model, opt=opt)
    e = time()

    print("--- Go for Testing ---")
    try:
        with torch.set_grad_enabled(False):
            auroc, auprc, f1, logits, loss, y_label, y_pred = test(
                testing_generator, model_max
            )
            print(
                f"Testing AUROC: {auroc} , AUPRC: {auprc}, F1: {f1}, Test loss: {loss}"
            )

            torch.save(model_max, os.path.join(args.output, "model.pt"))
            with open(os.path.join(args.output, "preds.txt"), "w") as f:
                for label, pred in zip(y_label, y_pred):
                    f.write(f"{label}\t{pred}\n")
    except:
        print("testing failed")

    print(e - s)
