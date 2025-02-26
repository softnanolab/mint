import argparse
import os

import torch
from tasks import get_task_datasets, join_sequences


def main(args):

    devices = [int(s) for s in args.devices.split(",")]

    model_name_clean = args.model_name.split("/")[-1]

    if args.sep_embed:
        model_name_clean = model_name_clean + "_sep"

    save_dir = f"./embeddings/{args.task}/{model_name_clean}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Running {model_name_clean} on {args.task}\n")

    train_dataset, val_dataset, test_dataset = get_task_datasets(args.task, args.test_run)

    if "t5" in args.model_name:
        from baselines import ProtT5

        model = ProtT5(
            model_name=args.model_name,
            layers="last",
            devices=devices,
            batch_size=args.bs,
            max_seq_length=args.max_seq_length,
        )

    elif "progen" in args.model_name:
        from baselines import ProGen

        model = ProGen(
            model_name=args.model_name,
            layers="last",
            devices=devices,
            batch_size=args.bs,
            max_seq_length=args.max_seq_length,
        )

    elif "esm" in args.model_name:
        from baselines import ESM

        model = ESM(
            model_name=args.model_name,
            layers="last",
            devices=devices,
            batch_size=args.bs,
            max_seq_length=args.max_seq_length,
        )
    else:
        print("No model found")
        return 0

    train_emb_file_name = f"{save_dir}/train.pt"
    if not os.path.isfile(train_emb_file_name):
        if args.task == "SKEMPI" or args.task == "BSA":
            if args.sep_embed:
                wt_inputs = model.encode_two(train_dataset.seqs1, train_dataset.seqs2, how="cat")
                mut_inputs = model.encode_two(train_dataset.seqs3, train_dataset.seqs4, how="cat")
                train_inputs = wt_inputs - mut_inputs
            else:
                train_seqs1 = join_sequences(train_dataset.seqs1, train_dataset.seqs2)
                train_seqs2 = join_sequences(train_dataset.seqs3, train_dataset.seqs4)
                train_inputs = model.encode_two(train_seqs1, train_seqs2)
        elif args.task == "Pdb-bind":
            train_seqs = train_dataset.seqs
            train_inputs = model.encode(train_seqs)
        else:
            if args.sep_embed:
                train_inputs = model.encode_two(
                    train_dataset.seqs1, train_dataset.seqs2, how="cat"
                )
            else:
                train_seqs = join_sequences(train_dataset.seqs1, train_dataset.seqs2)
                train_inputs = model.encode(train_seqs)
        torch.save(train_inputs, train_emb_file_name)

    if val_dataset is not None:
        val_emb_file_name = f"{save_dir}/val.pt"
        if not os.path.isfile(val_emb_file_name):
            if args.sep_embed:
                val_inputs = model.encode_two(val_dataset.seqs1, val_dataset.seqs2, how="cat")
            else:
                val_seqs = join_sequences(val_dataset.seqs1, val_dataset.seqs2)
                val_inputs = model.encode(val_seqs)
            torch.save(val_inputs, val_emb_file_name)

    if test_dataset is not None:
        test_emb_file_name = f"{save_dir}/test.pt"
        if not os.path.isfile(test_emb_file_name):
            if args.sep_embed:
                test_inputs = model.encode_two(test_dataset.seqs1, test_dataset.seqs2, how="cat")
            else:
                test_seqs = join_sequences(test_dataset.seqs1, test_dataset.seqs2)
                test_inputs = model.encode(test_seqs)
            torch.save(test_inputs, test_emb_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--task", type=str, default="HumanPPI")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--test_run", action="store_true", default=False)
    parser.add_argument("--sep_embed", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
