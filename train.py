import torch
import torch.nn as nn

from lib import *

if __name__ == "__main__":
    args = get_init_config()

    device = torch.device(args.device)

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    set_random_seed(args.random_seed)

    model_config = get_model_config(args.model_name)
    margin_module = get_margin_module(
        args.margin_module_name,
        args.embeddings_size,
        args.classes,
        args.margin,
        args.s)
    model = Model(model_config)

    margin_module = margin_module.to(device)
    model = model.to(device)

    if (device.type == "cuda" and args.ngpu >= 1):
        model = nn.DataParallel(model, list(range(args.ngpu)))

    train_loader = get_train_loader(
        args.dataset,
        args.dataset_path,
        args.batch_size,
        args.num_workers)

    optimizer = get_optimizer([{"params": model.parameters(),
                                "params": margin_module.identity_weights}],
                              args.optimizer,
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

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        criterion,
        optimizer,
        lr_scheduler,
        args.epochs,
        writer,
        logger,
        device,
        args.embeddings_size)
    trainer.train_loop(model, margin_module, train_loader, train_loader)
