import torch
import torch.nn as nn

from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config()

    if args.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    set_random_seed(args.random_seed)

    model_config = get_model_config(args.model_name)
    margin_module = get_margin_module(args.margin_module_name, args.embeddings_size, args.classes, args.margin, args.s)
    model = Model(model_config, margin_module)

    if (device.type == "cuda" and args.ngpu >= 1):
        model = nn.DataParallel(model, list(range(args.ngpu)))

    train_loader, val_loader = get_train_loader(args.dataset, args.dataset_path, args.batch_size, args.num_workers, train_portion=args.train_portion)

    optimizer = get_optimizer(model.parameters(), args.optimizer, 
                                  learning_rate=args.lr, 
                                  weight_decay=args.weight_decay, 
                                  logger=logger,
                                  momentum=args.momentum, 
                                  alpha=args.alpha, 
                                  beta=args.beta)

    lr_scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, logger, 
                                    step_per_epoch=len(train_loader), 
                                    step_size=args.decay_step, 
                                    decay_ratio=args.decay_ratio,
                                    total_epochs=args.epochs)

    criterion = nn.CrossEntropy()

    trainer = Trainer(criterion, optimizer, args.epochs, writer, logger, args.device, training_strategy)
    trainer.train_loop(model, train_loader, val_loader)

