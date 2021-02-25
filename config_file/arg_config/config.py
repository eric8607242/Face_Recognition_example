import os
import argparse

def get_init_config():
    parser = argparse.ArgumentParser(description="Searching configuration")
    parser.add_argument("--title",                 type=str,                           help="Experiment title", required=True)
    parser.add_argument("--resume",                type=str,                           help="Resume path")
    parser.add_argument("--random-seed",           type=int,    default=42,            help="Random seed")
    parser.add_argument("--device",                type=str,    default="cuda")
    parser.add_argument("--ngpu",                  type=int,    default=1)

    # Model setting
    parser.add_argument("--model-name",            type=str,    default="mobileface",  help="Model name for backbone")
    parser.add_argument("--margin-module-name",         type=str,    default="arcface",     help="Margin module for model training")
    parser.add_argument("--embeddings-size",       type=int,    default=128)
    parser.add_argument("--margin",                type=int,    default=128)
    parser.add_argument("--s",                     type=int,    default=128)


    # Datset config
    parser.add_argument("--dataset",               type=str,    default="casia",    help="Name of dataset")
    parser.add_argument("--dataset-path",          type=str,    default="/workspace/CASIA-maxpy-clean",     help="Path to dataset")

    parser.add_argument("--classes",               type=int,    default=10575,           help="Class number for classification")
    parser.add_argument("--input-size",            type=int,    default=32,            help="Input size of dataset")

    parser.add_argument("--batch-size",            type=int,    default=512,           help="Batch size")
    parser.add_argument("--num-workers",           type=int,    default=4)

    # Model training config
    parser.add_argument("-epochs",                 type=int,    default=40,           help="The epochs for supernet training")

    parser.add_argument("--optimizer",             type=str,    default="sgd",         help="Optimizer for supernet training")
    parser.add_argument("--lr",                    type=float,  default=0.1)
    parser.add_argument("--weight-decay",          type=float,  default=0.00005)
    parser.add_argument("--momentum",              type=float,  default=0.9)

    parser.add_argument("--lr-scheduler",          type=str,    default="step")
    parser.add_argument("--decay-step",            type=int,    default=12)
    parser.add_argument("--decay-ratio",           type=float,  default=0.1)

    parser.add_argument("--alpha",                 type=float)
    parser.add_argument("--beta",                  type=float)

    # Path config
    parser.add_argument("--root-path",             type=str,    default="./logs/")
    parser.add_argument("--logger-path",           type=str,    default="./logs/")
    parser.add_argument("--writer-path",           type=str,    default="./logs/tb/")

    args = parser.parse_args()
    args = setting_path_config(args)

    return args

def setting_path_config(args):
    """
    Concatenate root path to each path argument
    """
    if not os.path.exists(os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))):
        args.root_path = os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))
        os.makedirs(args.root_path)

    return args

    
    
