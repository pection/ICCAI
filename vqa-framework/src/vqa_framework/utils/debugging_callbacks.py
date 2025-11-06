"""
Folder full of callbacks I've made at some point or another for debugging
"""

from pytorch_lightning.callbacks import Callback
import logging
from vqa_framework.utils.vocab import ClosureVocab


class PrintBatch(Callback):
    """
    Simple callback that prints SHAPES or CLEVR question & program
    """

    def __init__(self):
        super().__init__()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ) -> None:
        vocab: ClosureVocab = trainer.model.vocab
        questions = batch[0]
        programs = batch[5]
        assert questions.size(0) == programs.size(0)

        for i in range(questions.size(0)):
            logging.critical("======================================")
            logging.critical("QUESTION")
            logging.critical(
                " ".join([vocab.question_idx_to_token(x.item()) for x in questions[i]])
            )
            logging.critical("PROGRAM")
            logging.critical(
                " ".join([vocab.program_idx_to_token(x.item()) for x in programs[i]])
            )


class PrintGlobalStep(Callback):
    """
    Simple callback that prints SHAPES or CLEVR question & program
    """

    def __init__(self):
        super().__init__()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ) -> None:
        print(trainer.global_step)
