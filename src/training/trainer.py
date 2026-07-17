import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.arithmetic import PAD_IDX
from src.models.coalition import CoalitionModel
from src.training.losses import (
    coalition_load_balance_loss,
    coalition_size_loss,
    moe_load_balance_loss,
)


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device, save_dir: str = 'checkpoints'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        tc = config['training']
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc['learning_rate'], weight_decay=tc['weight_decay']
        )
        self.epochs = tc['epochs']
        self.grad_clip = tc['grad_clip']

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc['epochs'], eta_min=tc['learning_rate'] * 0.1
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'epoch_time': []}

        self._is_coalition = isinstance(model, CoalitionModel)
        self._model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        if self._is_coalition:
            return 'coalition'
        for layer in self.model.layers:
            if hasattr(layer.ffn, 'gate'):
                return 'moe'
        return 'dense'

    def _compute_aux_loss(self, aux_data_list: list[dict]) -> torch.Tensor:
        aux_loss = torch.tensor(0.0, device=self.device)

        for aux in aux_data_list:
            if aux['type'] == 'moe':
                moe_cfg = self.config['moe']
                loss = moe_load_balance_loss(
                    aux['gate_probs'], aux['top_k_indices'], moe_cfg['n_experts']
                )
                aux_loss = aux_loss + moe_cfg['balance_coef'] * loss

            elif aux['type'] == 'coalition':
                cc = self.config['coalition']
                bal_loss = coalition_load_balance_loss(aux['node_activation'])
                size_loss = coalition_size_loss(aux['node_activation'], cc['target_coalition_size'])
                aux_loss = aux_loss + cc['balance_coef'] * bal_loss + cc['size_coef'] * size_loss

        return aux_loss

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            result = self.model(input_ids)
            logits = result['logits']

            task_loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            aux_loss = torch.tensor(0.0, device=self.device)
            if 'aux_data' in result:
                aux_loss = self._compute_aux_loss(result['aux_data'])

            loss = task_loss + aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            result = self.model(input_ids)
            logits = result['logits']

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            match = (preds == target_ids) | (target_ids == PAD_IDX)
            correct += match.all(dim=-1).sum().item()
            total += target_ids.size(0)

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        print(f"Training {self._model_type} model | {self.model.count_parameters():,} parameters")

        for epoch in range(self.epochs):
            start = time.time()

            if self._is_coalition:
                temp = self.model.compute_temperature(epoch, self.epochs)
                self.model.set_temperature(temp)

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - start
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['epoch_time'].append(elapsed)

            temp_str = ''
            if self._is_coalition:
                temp_str = f' | temp={self.model.get_temperature():.3f}'

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | time={elapsed:.1f}s{temp_str}"
                )

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(self.epochs)
        return self.history

    def save_checkpoint(self, epoch: int):
        path = self.save_dir / f'{self._model_type}_epoch{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']
