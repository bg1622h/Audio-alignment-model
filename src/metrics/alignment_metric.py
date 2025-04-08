import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


class AlignmentMetric:
    """
    Class for calculating the alignment metric
    """

    def __init__(self, name=None, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
            threshold: Sound probability threshold
            save_alignment: if true saves the alignment
        """
        self.name = name if name is not None else type(self).__name__
        self.threshold = kwargs.get("threshold", 0.05)
        self.save_alignment = kwargs.get("save_alignment", False)
        self.save_A = None

    def __call__(self, outputs, targets):
        """
        Calculate alignment metric
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
            fn_annot_pitch = fn_annot_pitch.T
            f_annot_pitch_unique = self.unique_rows_preserve_order(fn_annot_pitch.T)
            C, path = self.dtw_multilabel_with_pauses(predictions, f_annot_pitch_unique)
            empty = [0 for _ in range(72)]
            A = []
            for step in path:
                if step[1] != -1:
                    A.append(f_annot_pitch_unique[step[1]])
                else:
                    A.append(empty)
            A = np.array(A)
            A = A[: fn_annot_pitch.shape[1]]
            if self.save_alignment:
                self.save_A = A
            self.save_annot_unique = f_annot_pitch_unique.T
            P_1, R_1, F1_1, TP_1, FP_1, FN_1 = self.compute_eval_measures(
                fn_annot_pitch.T, A
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

    def unique_rows_preserve_order(self, matrix):
        """
        Returns the rows of the matrix so that neighbouring rows are not repeated and are in the same order as in the input matrix
        Args:
            matrix
        Returns:
            matrix with unique rows
        """
        unique_rows = []
        last_row = np.zeros((1, matrix.shape[1]))
        for row in matrix:
            if np.all(row == last_row):
                continue
            unique_rows.append(row)
            last_row = row
        unique_matrix = np.array(unique_rows)
        return unique_matrix

    def dtw_multilabel_with_pauses(self, Y, P):
        """
        Calculate DTW for Y,P
        Args:
            Y - prediction note probability matrix
            P - A matrix in which rows contain encoded sets of notes, 1 - note is present, 0 - not present
        Returns:
            C - Matrix of alignment costs
            path - path corresponding to the alignment, each element of the form (i,j) - at the moment of time i the jth set of notes or silence (j=-1) is sounded
        """
        N, M = len(Y), len(P)
        D = 1 - cosine_similarity(Y, P)
        C = np.zeros((N, M))
        C[0, 0] = D[0, 0]

        # Fill first row and column
        for i in range(1, N):
            C[i, 0] = C[i - 1, 0] + D[i, 0]
        for j in range(1, M):
            C[0, j] = C[0, j - 1] + D[0, j]

        # Fill the rest of the matrix
        for i in range(1, N):
            for j in range(1, M):
                if self._is_pause(Y[i]):
                    C[i, j] = C[i - 1, j]
                else:
                    C[i, j] = D[i, j] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
        # Find the optimal path
        path = []
        i, j = N - 1, M - 1
        while i > 0 or j > 0:
            if (i > 0) and (self._is_pause(Y[i])):
                path.append((i, -1))  # Silence
                i -= 1
            else:
                path.append((i, j))
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    min_cost = min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
                    if min_cost == C[i - 1, j]:
                        i -= 1
                    elif min_cost == C[i, j - 1]:
                        j -= 1
                    else:
                        i -= 1
                        j -= 1
        path.append((0, 0 if not self._is_pause(Y[0]) else -1))
        path.reverse()
        return C, path

    def _is_pause(self, predict):
        """
        Checks if there is sound according to the prediction
        Args:
            predict - prediction note probability row
        Returns:
            1 - There's a sound
            0 - No sound
        """
        return np.all(predict < self.threshold)

    def get_alignment(self):
        """
        Returns:
            Returns the stored alignment with shape = (T,P)
        """
        return self.save_A
