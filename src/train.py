import torch
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from lightning.pytorch import Trainer, seed_everything
from dataset.sampling_dataset import SamplingDataset, MyCollate
from dataset.validation_dataset import ValidationDataset
from models.reranker import ReRanker
from logger import HistoryLogger

seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
LOGGER.addHandler(console)

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bert_base = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = BertTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[START]', '[END]']})

    train_dataset = SamplingDataset(data_dir='data/processed/MedMention/st21pv/train_dev',
                                    dictionary_file='data/processed/MedMention/umls/dictionary.txt',
                                    candidate_file='output/candidates/st21_train_dev/sparse_candidate2.txt',
                                    tokenizer=tokenizer,
                                    device=device,
                                    logger=LOGGER)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  collate_fn=MyCollate(),
                                  batch_size=32,
                                  shuffle=True)
    val_dataset = ValidationDataset(data_dir='data/processed/MedMention/st21pv/test',
                                    dictionary_file='data/processed/MedMention/umls/dictionary.txt',
                                    candidate_file='output/candidates/st21_test/sparse_candidate2.txt',
                                    tokenizer=tokenizer,
                                    device=device,
                                    logger=LOGGER)

    model = ReRanker(bert_base=bert_base,
                     tokenizer=tokenizer)

    # train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=24)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32)
    logger = HistoryLogger()
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[0],
                         min_epochs=1,
                         max_epochs=3,
                         logger=logger,
                         log_every_n_steps=0,
                         enable_checkpointing=False,
                         deterministic=True)

    trainer.fit(model, train_dataloader, val_dataloader)
