import numpy as np

from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics=None, log_plots=False):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        labels = batch["notes"]
        targ_excerpt = labels.detach().permute(0, 2, 1)
        loss = self.criterion(
            outputs=outputs, targ_excerpt=targ_excerpt, device=self.device
        )
        batch.update({"loss": loss})
        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if log_plots:
            self.log_plots(
                batch=batch, target=targ_excerpt, outputs=outputs, count_samples=4
            )
        if metrics:
            TP, FP, FN = metrics(outputs, labels)
            batch.update({"TP": TP})
            batch.update({"FP": FP})
            batch.update({"FN": FN})
        return batch
