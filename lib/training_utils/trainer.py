import time
from ..utils import AverageMeter, accuracy
from .evaluate import evaluate


class Trainer:
    def __init__(
            self,
            criterion,
            optimizer,
            lr_scheduler,
            epochs,
            writer,
            logger,
            device,
            embeddings_size):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.writer = writer
        self.logger = logger

        self.epochs = epochs
        self.embeddings_size = embeddings_size

    def train_loop(self, model, margin_module, train_loader, val_loader):
        best_top1_acc = 0.0

        for epoch in range(self.epochs):
            self._training_step(model, margin_module, train_loader, epoch)
            self.validate(model, val_loader)

            self.lr_scheduler.step()

            # if val_top1 > best_top1_acc:
            #self.logger.info("Best validation top1-acc : {}!".format(val_top1))
            #best_top1_acc = val_top1

    def _training_step(self, model, margin_module, train_loader, epoch):
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()

            outs = model(X)
            outs = margin_module(outs, y)

            loss = self.criterion(outs, y)
            loss.backward()

            self.optimizer.step()
            self._intermediate_stats_logging(
                outs,
                y,
                loss,
                step,
                epoch,
                N,
                len_loader=len(train_loader),
                val_or_train="Train")

        self._epoch_stats_logging(start_time, epoch, val_or_train="Train")
        self._reset_average_tracker()

    def validate(self, model, test_dataset):
        model.eval()
        start_time = time.time()

        for k, dataset in test_dataset.items():
            self.logger("Evaluating on {}".format(k))

            data = dataset["data"]
            label = dataset["label"]

            idx = 0
            embeddings = np.array(len(data), self.args.embeddings_size)
            with torch.no_grad():
                while idx + self.args.batch_size <= len(data):
                    X = torch.tensor(data[idx:idx + self.args.batch_size])
                    X = X.to(self.device, non_blocking=True)

                    outs = model(X)
                    embeddings[idx:idx +
                               self.args.batch_size] = outs.cpu().numpy()

                    idx += self.args.batch_size

                if idx < len(data):
                    X = torch.tensor(data[idx:])
                    X = X.to(self.device, non_blocking=True)

                    outs = model(X)
                    embeddings[idx:] = outs.cpu().numpy()

            true_positive_rate, false_positive_rate, accuracy, best_threshold = evaluate(
                embeddings, label, n_folds=5)

            self.logger.info(
                "Valid : [{}] Accuracy  {} True Positive Rate {} False Positive Rate {} Best Threshold {}".format(
                    accuracy, true_positive_rate, false_positive_rate, best_threshold))

    def _epoch_stats_logging(
            self,
            start_time,
            epoch,
            val_or_train,
            info_for_writer=""):
        """
        Logging training information and record information in Tensorboard for each epoch
        """
        self.writer.add_scalar(
            "{}/_loss/{}".format(val_or_train, info_for_writer), self.losses.get_avg(), epoch)
        self.writer.add_scalar(
            "{}/_top1/{}".format(val_or_train, info_for_writer), self.top1.get_avg(), epoch)
        self.writer.add_scalar(
            "{}/_loss/{}".format(val_or_train, info_for_writer), self.top5.get_avg(), epoch)

        self.logger.info(
            "{} : [{:3d}/{}] Final Loss {:.3f} Final Prec@(1, 5) ({:.1%}, {:.1%}) Time {:.2f}" .format(
                val_or_train,
                epoch + 1,
                self.epochs,
                self.losses.get_avg(),
                self.top1.get_avg(),
                self.top5.get_avg(),
                time.time() - start_time))

    def _intermediate_stats_logging(
            self,
            outs,
            y,
            loss,
            step,
            epoch,
            N,
            len_loader,
            val_or_train,
            print_freq=100):
        """
        Logging training infomation at each print_freq iteration
        """
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))

        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if (step > 1 and step % print_freq == 0) or step == len_loader - 1:
            self.logger.info(
                "{} : [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Prec@(1, 5) ({:.1%}, {:.1%})" .format(
                    val_or_train,
                    epoch + 1,
                    self.epochs,
                    step,
                    len_loader - 1,
                    self.losses.get_avg(),
                    self.top1.get_avg(),
                    self.top5.get_avg()))

    def _reset_average_tracker(self):
        for tracker in [self.top1, self.top5, self.losses]:
            tracker.reset()
