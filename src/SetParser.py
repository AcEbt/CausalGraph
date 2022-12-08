import argparse


def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="spmotif_0.5", help="Dataset name") # ba_2motifs, spmotif_0.5,
    parser.add_argument("--dataset_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="result", help="Data directory")

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_layer", type=int, default=2, help="Hidden layer dimension")
    # parser.add_argument("--hid_dim", type=int, default=10, help="Hidden layer dimension")
    # parser.add_argument("--out_dim", type=int, default=5, help="Hidden layer dimension")
    # parser.add_argument("--dropout", type=float, default=0.9, help="Learning rate of optimizer")

    parser.add_argument("--hid_units", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument('--itr', type=int, default=5, help="Number of fine-tuning")
    parser.add_argument('--epoch', type=int, default=20, help="Number of training epochs per itr")
    parser.add_argument('--ft_epoch', type=int, default=10, help="Number of training epochs per itr")
    parser.add_argument('--patience', type=int, default=20, help="Number of waiting epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of optimizer")
    parser.add_argument("--p_train", type=float, default=0.6, help="percentage of validation set")
    parser.add_argument("--p_test", type=float, default=0.2, help="percentage of test set")

    parser.add_argument("--aug_num", type=int, default=10, help="# augmentation")
    return parser.parse_args()