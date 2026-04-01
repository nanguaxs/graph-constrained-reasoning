"""SFT 训练器。"""
import torch
from tqdm import tqdm

from logging_utils import get_logger


logger = get_logger("sft_trainer")


class SFTTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = model.device

    def _move_batch_to_device(self, batch):
        moved_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def train(self, train_loader, num_epochs):
        """训练循环。"""
        self.model.train()
        accumulation_steps = max(1, self.config.gradient_accumulation_steps)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            skipped_batches = 0
            effective_batches = 0
            total_batches = len(train_loader)
            accumulated_batches = 0

            self.optimizer.zero_grad()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                if batch is None:
                    skipped_batches += 1
                    logger.debug("epoch=%s 遇到空 batch，已跳过", epoch + 1)
                    continue

                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                (loss / accumulation_steps).backward()

                epoch_loss += loss.item()
                effective_batches += 1
                accumulated_batches += 1

                if accumulated_batches >= accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_batches = 0

                logger.debug(
                    "epoch=%s batch=%s loss=%.4f batch_size=%s",
                    epoch + 1,
                    effective_batches,
                    loss.item(),
                    batch["input_ids"].shape[0],
                )

            if accumulated_batches > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss = epoch_loss / effective_batches if effective_batches > 0 else 0.0
            logger.info(
                "epoch=%s avg_loss=%.4f total_batches=%s skipped=%s effective=%s accumulation=%s",
                epoch + 1,
                avg_loss,
                total_batches,
                skipped_batches,
                effective_batches,
                accumulation_steps,
            )
