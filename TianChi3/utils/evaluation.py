"""
Evaluation.

=====

Evaluate the final result.

"""

import warnings

class Evaluation(object):

    def __init__(self, l_pred, l_ref):
        """
        Class initialization.

        Parameters
        ----------
        l_pred : predicted list. Structure: [[uid, iid], ...]
        l_ref : reference (ground truth) list. Structure: [[uid, iid], ...]

        """
        self.l_pred = l_pred
        self.l_ref = l_ref

    def intersection(self):
        """
        Number of same elements between l_pred and l_ref.

        """
        num = 0
        for i in self.l_pred:
            if i in self.l_ref:
                num += 1
        return num

    def precision(self):
        """
        Precision.

        """
        #num = self.intersection()
        return self.intersection() / (len(self.l_pred) + 0.0)

    def recall(self):
        """
        Recall.
        
        """
        #num = self.intersection()
        return self.intersection() / (len(self.l_ref) + 0.0)

    def F1(self):
        """
        F1.  
        
        """
        f1 = 0
        try:
            f1 = 2*self.precision()*self.recall() / (self.precision()+self.recall())
        except:
            warnings.warn('precision + recall = 0')
        return f1

    def eval_metrics(self):
        return self.F1(), self.precision(), self.recall()
