import argparse
import os
import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from dmask.model_t5_4_classification import T5ForConditionalGeneration
from transformers import (
    AdamW,
    
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from utils import Rcv1Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, decoder_paths_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_paths_mask=decoder_paths_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        # lm_labels = batch["target_ids"]
        # {"texts_tokens": texts_tokens, "texts_mask": texts_mask, "labels_tokens": labels_tokens, "labels_mask": labels_mask}
        lm_labels = batch["labels_tokens"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["texts_tokens"],
            attention_mask=batch["texts_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['labels_mask'],
            decoder_paths_mask=batch['paths_mask']
        )

        loss = outputs[0]
        self_attn_loss=outputs[-1]
        return loss + 100 * self_attn_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = Rcv1Dataset(tokenizer=self.tokenizer, data_dir=self.hparams.data_dir, type_path="rcv1_train",
                                    max_len=self.hparams.max_seq_length)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = Rcv1Dataset(tokenizer=self.tokenizer, data_dir=self.hparams.data_dir, type_path="rcv1_val",
                                  max_len=self.hparams.max_seq_length)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    set_seed(42)
    logger = logging.getLogger(__name__)

    args_dict = dict(
        data_dir="./data/rcv1_data", # path for data files
        output_dir="./ckpt_rcv1/rcv1_dmask_ckpt", # path to save the checkpoints
        ckpt_dir='./ckpt_rcv1/rcv1_dmask_ckpt', # path to save the checkpoints
        model_name_or_path='./pretrain_model/t5-base',
        tokenizer_name_or_path='./pretrain_model/t5-base',
        max_seq_length=300,
        learning_rate=5e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=10,
        eval_batch_size=10,
        num_train_epochs=20,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args = argparse.Namespace(**args_dict)

    train_path = './data/rcv1_data/rcv1_train.txt'
    val_path = './data/rcv1_data/rcv1_val.txt'
    test_path = './data/rcv1_data/rcv1_test.txt'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    )

    train_params = dict(
        default_root_dir=args.ckpt_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision= 16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    print("best k models are:\t%s" %checkpoint_callback.best_k_models)