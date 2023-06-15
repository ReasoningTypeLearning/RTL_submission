from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import RobertaForMultipleChoice
from data_util import *
from pytorch_lightning.callbacks import ModelCheckpoint
import re, random, json, math
import numpy as np

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def recurrent_layers(total_steps, num_layers):
    out = []
    for _ in range(math.ceil(total_steps / num_layers)):
        for j in range(1, num_layers+1):
            out.append(j)
    return out[:total_steps]
            
def incremental_layers(total_steps, num_layers):
    out = []
    for i in range(1, num_layers+1):
        for _ in range(math.ceil(total_steps / num_layers)):
            out.append(i)
    return out[:total_steps]

def random_layers(total_steps, num_layers):
    random.seed(42)
    out = []
    for i in range(total_steps):
        out.append(random.randint(1, num_layers))
    return out[:total_steps]


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.relu = nn.ReLU()
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate, model_name, total_steps, filename, n_clusters, co, layer_schedule, num_layers, warmup_ratio, task):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.roberta = RobertaForMultipleChoice.from_pretrained(model_name)
        self.linears = nn.ModuleList([ClassificationHead(self.roberta.config, n_clusters) for i in range(num_layers)])
        self.loss_func = nn.CrossEntropyLoss()
        self.co = co
        self.best_dev_acc = 0
        self.best_dev_loss = 999
        self.filename = filename
        self.this_step = 0
        self.layer_schedule = layer_schedule
        self.warmup_ratio = warmup_ratio
        self.task = task
    
    def training_step(self, batch, batch_idx):
        out_type = self.roberta.roberta(input_ids=batch['ca_input_ids'], 
                        attention_mask=batch['ca_attention_mask'],
                        output_hidden_states=True)
        out_qa = self.roberta(input_ids=batch['cqa_input_ids'], 
                        attention_mask=batch['cqa_attention_mask'],
                        labels=batch['tgt'])

        this_layer = self.layer_schedule[self.this_step]
        this_linear_layer = self.linears[this_layer-1]
        q_type_emb = out_type.hidden_states[this_layer]
        self.this_step += 1

        q_type_loss = self.loss_func(this_linear_layer(q_type_emb), batch['q_type'])
        loss = out_qa.loss + self.co * q_type_loss
        self.log("q_type_loss", q_type_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.roberta(input_ids=batch['cqa_input_ids'], 
                        attention_mask=batch['cqa_attention_mask'],
                        labels=batch['tgt'])
        labels = batch['tgt']
        predicted = torch.argmax(out.logits, dim=-1)
        correct = (predicted == labels).sum().item()
        return correct, labels.shape[-1], out.loss
    
    def validation_epoch_end(self, results):
        correct, total, loss = 0, 0, 0
        for batch_result in results:
            correct += batch_result[0]
            total += batch_result[1]
            loss += batch_result[2]
        accuracy = 100 * correct / total
        loss = loss / total
        print("Accuracy: ", accuracy)
        print('This epoch step: ', self.this_step)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        if accuracy > self.best_dev_acc:
            self.best_dev_acc = accuracy
        if loss < self.best_dev_loss:
            self.best_dev_loss = loss

    def test_step(self, batch, batch_idx):
        if self.task == 'reclor':
            out = self.roberta(input_ids=batch['cqa_input_ids'], 
                        attention_mask=batch['cqa_attention_mask'])
            predicted = torch.argmax(out.logits, dim=-1).cpu().detach().numpy()
            return predicted
        else:
            out = self.roberta(input_ids=batch['cqa_input_ids'], 
                            attention_mask=batch['cqa_attention_mask'],
                            labels=batch['tgt'])
            labels = batch['tgt']
            predicted = torch.argmax(out.logits, dim=-1)
            correct = (predicted == labels).sum().item()
            return correct, labels.shape[-1], out.loss
    
    def test_epoch_end(self, results):
        if self.task == 'reclor':
            out = np.concatenate(results, axis=-1)
            with open(self.filename, 'wb') as f:
                print('Saving result file to %s' % self.filename)
                np.save(f, out)
        else:
            correct, total, loss = 0, 0, 0
            for batch_result in results:
                correct += batch_result[0]
                total += batch_result[1]
                loss += batch_result[2]
            accuracy = 100 * correct / total
            self.log('test_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
            # self.log('test_loss', loss / total, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.98),
                                    eps=1e-6, 
                                    weight_decay=0.01,
                                    )
        lr_scheduler = {'scheduler':get_linear_schedule_with_warmup(optimizer, self.total_steps*self.warmup_ratio, self.total_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
def main(hparams):
    fix_seed(42)
    n_clusters, co_q = set_hyerparameters(hparams.model_name, hparams.task, hparams.layer_schedule)

    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    litData = QuestionAnswerDataModule(tokenizer, hparams.batch_size, hparams.path, n_clusters, hparams.max_length)

    total_steps = math.ceil(len(litData) / hparams.batch_size) * hparams.epoch
    print("Total steps:", total_steps)

    if hparams.layer_schedule == 'recurrent':
        layer_schedule = recurrent_layers(total_steps, hparams.num_layers)
    elif hparams.layer_schedule == 'increment':
        layer_schedule = incremental_layers(total_steps, hparams.num_layers)
    else:
        layer_schedule = random_layers(total_steps, hparams.num_layers)
    assert len(layer_schedule) == total_steps

    # print(layer_schedule)

    model = LitModel(learning_rate=hparams.learning_rate, 
                    model_name=hparams.model_name, 
                    total_steps=total_steps,
                    filename=hparams.filename,
                    n_clusters=n_clusters,
                    co=co_q,
                    layer_schedule=layer_schedule,
                    num_layers=hparams.num_layers,
                    warmup_ratio=hparams.warmup_ratio,
                    task=hparams.task,
                    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.path_checkpoint,
        filename=hparams.name_checkpoint,
        monitor='val_acc',
        mode='max',
    )

    trainer = pl.Trainer(accelerator=hparams.accelerator, 
                            devices=hparams.devices, 
                            max_epochs=hparams.epoch,
                            accumulate_grad_batches=hparams.grad_accumulate,
                            amp_backend='native',
                            precision=16,
                            callbacks=[checkpoint_callback],
                            enable_checkpointing=True,
                            num_sanity_val_steps=0,
                            )
    if not hparams.is_test:
        trainer.fit(model, litData)

    print('Best dev acc:', model.best_dev_acc)
    print('Best dev loss:', model.best_dev_loss)

    trainer.test(ckpt_path="best", datamodule=litData)
    
    # torch.save(model.roberta.roberta.state_dict(), 'lightning_logs/pretrain/roberta_pretrain.pt')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--grad_accumulate", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--model_name", default="roberta-large", type=str)
    # parser.add_argument("--model_name", default="chitanda/merit-roberta-large-v2", type=str)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--path", default="dataset/reclor_data/", type=str)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--co_q", default=0.1, type=float)
    parser.add_argument("--filename", default="output/test_pred.npy", type=str)
    parser.add_argument("--layer_schedule", default="recurrent", type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--path_checkpoint", default="lightning_logs/reclor/", type=str)
    parser.add_argument("--name_checkpoint", default="finetuned_roberta", type=str)
    parser.add_argument("--num_layers", default=24, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--task", default="reclor", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
