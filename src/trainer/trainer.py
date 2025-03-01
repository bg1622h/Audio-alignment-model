from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import numpy as np
import torch
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def create_log_plot(self, input, target, predict):
        """
        lengths of all the data are the same
        """
        fig, axes = plt.subplots(len(input), 4, figsize=(12, 12))
        axes[0,0].set_title("Input")
        for i in range(len(input)):
            axes[i,0].imshow(input[i], cmap='viridis', interpolation='nearest')
        axes[0,1].set_title("Target")
        for i in range(len(target)):
            axes[i,1].imshow(target[i], cmap='viridis', interpolation='nearest')
        axes[0,2].set_title("Predict")
        for i in range(len(predict)):
            im = axes[i,2].imshow(np.exp(predict[i]), cmap='viridis', interpolation='nearest')
            fig.colorbar(im,ax=axes[i,2],label="Value")
        for i in range(len(predict)):
            im = axes[i,3].imshow((np.exp(predict[i]) > 0.5).int(), cmap='viridis', interpolation='nearest')
            fig.colorbar(im,ax=axes[i,3],label="Value")
        fig.suptitle("Input, Target, Predict, exp Predict")
        return fig

    def process_batch(self, batch, log_plots = False):#, metrics: MetricTracker):
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

        #metric_funcs = self.metrics["inference"]
        if self.is_train:
            #metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        labels = batch['notes']
        targ_excerpt = labels.detach().numpy().transpose(0,2,1)
        """
        The code below leaves unique columns in the sense of removing duplicates throughout the array, 
        and leaves only those columns where there is a change in values from the previous column.
        It then adds 0 column corresponding to silence
        """
        targets_array = []
        all_losses = 0
        for y_pred,batch_target in zip(outputs,targ_excerpt):
            inds = np.concatenate((np.array([0]), 1+np.where((batch_target[:, 1:]!=batch_target[:, :-1]).any(axis=0))[0]))
            target_np = batch_target[:, inds]
            target_blank = np.zeros((target_np.shape[0]+1, target_np.shape[1]+1))
            target_blank[1:, 1:] = target_np
            target_blank[0, 0] = 1
            targets = torch.tensor(target_blank, dtype=torch.float32).to(self.device)
            #targets = torch.tensor(batch_target.T)
            log_probs = y_pred.squeeze().transpose(1,2)
            input_lengths = torch.tensor(log_probs.size(-1), dtype=torch.long).to(self.device)
            target_lengths = torch.tensor(targets.size(-1), dtype=torch.long).to(self.device)
            loss_input = {
                'targets': targets,
                'log_probs': log_probs,
                'input_lengths': input_lengths,
                'target_lengths': target_lengths
            }
            all_losses = all_losses + self.criterion(**loss_input)/ (input_lengths*target_lengths)
        all_losses/=len(targ_excerpt)
        batch.update({"loss": all_losses})

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if log_plots:
            self.writer.add_image("Visualization",
                                  self.create_log_plot
                                  (
                                    batch['audio'][:4],
                                    targ_excerpt[:4], 
                                    np.transpose(outputs[:4,1,:,1:], (0,2,1))
                                  )
                                )

        # update metrics for each loss (in case of multiple losses)
        #for loss_name in self.config.writer.loss_names:
        #    metrics.update(loss_name, batch[loss_name].item())

        #for met in metric_funcs:
        #    metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
