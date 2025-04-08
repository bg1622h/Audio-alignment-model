import numpy as np
import torch


class PredictMetric:
    """
    Class for calculating the predict metric
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
            threshold: Pitch probability threshold
        """
        self.name = name if name is not None else type(self).__name__
        self.threshold = kwargs.get("threshold", 0.5)

    def __call__(self, outputs, targets):
        """
        Calculate predict metric
        Args:
            outputs - model outputs (logits), shape = (B,2,T,P + blank)
            targets - model targets, shape = (B,T,P)
        Returns:
            TP,FP,FN
        """
        TP = 0
        FP = 0
        FN = 0
        outputs = outputs.cpu()
        targets = targets.cpu()
        for output, target in zip(outputs, targets):
            predictions = np.exp(output)
            predictions = predictions[1, :, 1:]
            predictions = predictions.detach().numpy()
            fn_annot_pitch = target.numpy()
            predictions = (predictions > self.threshold).astype(int)
            P_1, R_1, F1_1, TP_1, FP_1, FN_1 = self.compute_eval_measures(
                fn_annot_pitch.T, predictions
            )
            TP += TP_1
            FP += FP_1
            FN += FN_1
        return TP, FP, FN

    def compute_eval_measures(self, I_ref, I_est):
        """Compute evaluation measures including precision, recall, and F-measure
        Args:
            I_ref, I_est: Sets of positive items for reference and estimation
        Returns:
            P, R, F: Precsion, recall, and F-measure
            TP, FP, FN: Number of true positives, false positives, and false negatives
        """
        assert I_ref.shape == I_est.shape, "Dimension of input matrices must agree"
        TP = np.sum(np.logical_and(I_ref, I_est))
        FP = np.sum(I_est > 0, axis=None) - TP
        FN = np.sum(I_ref > 0, axis=None) - TP
        P = 0
        R = 0
        F = 0
        if TP > 0:
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F = 2.0 * P * R / (P + R)
        return P, R, F, TP, FP, FN
