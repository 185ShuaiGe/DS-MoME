
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.plaa_mllm import PLAAMLLM
from data.dataset_loader import AIGIDataset
from utils.log_utils import Logger
from utils.device_utils import DeviceManager


class PLAAMLLMTrainer:
    """
    PLAA-MLLM 三阶段训练器
    """
    
    def __init__(
        self,
        model: PLAAMLLM,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        path_config: PathConfig,
        stage: int = 1
    ):
        """
        初始化训练器
        
        Args:
            model: PLAA-MLLM 模型
            model_config: 模型配置
            device_config: 设备配置
            path_config: 路径配置
            stage: 训练阶段 (1, 2, 3)
        """
        self.model = model
        self.model_config = model_config
        self.device_config = device_config
        self.path_config = path_config
        self.stage = stage
        
        self.logger = Logger(name=f"Trainer_Stage{stage}")
        self.device_manager = DeviceManager(device_config)
        self.device = device_config.get_device()
        
        self.best_metric = -float('inf')
        self.global_step = 0
        self.epoch = 0
        
        self._setup_stage()
    
    def _setup_stage(self) -&gt; None:
        """
        根据训练阶段设置参数冻结策略
        """
        self.logger.info(f"Setting up training stage {self.stage}")
        
        if self.stage == 1:
            self._freeze_clip_llm()
            self._unfreeze_artifact_adapter()
        elif self.stage == 2:
            self._freeze_visual_streams()
            self._apply_lora()
        elif self.stage == 3:
            self.logger.info("Stage 3: DPO training, all parameters trainable")
    
    def _freeze_clip_llm(self) -&gt; None:
        """
        冻结 CLIP 语义流和 LLM 主干网络 (Stage 1)
        """
        self.logger.info("Freezing CLIP semantic stream and LLM backbone")
        pass
    
    def _unfreeze_artifact_adapter(self) -&gt; None:
        """
        解冻伪影流和交叉注意力适配器 (Stage 1)
        """
        self.logger.info("Unfreezing artifact stream and cross-attention adapter")
        pass
    
    def _freeze_visual_streams(self) -&gt; None:
        """
        冻结所有视觉流 (Stage 2)
        """
        self.logger.info("Freezing all visual streams")
        pass
    
    def _apply_lora(self) -&gt; None:
        """
        为 LLM 应用 LoRA (Stage 2)
        """
        self.logger.info(f"Applying LoRA with rank={self.model_config.lora_rank}")
        pass
    
    def compute_loss_stage1(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -&gt; Dict[str, torch.Tensor]:
        """
        计算 Stage 1 损失：BCE Loss + DICE Loss
        
        Args:
            outputs: 模型输出
            labels: 分类标签
            masks: 可选的定位掩码
        
        Returns:
            Dict: 损失字典
        """
        loss_dict = {}
        
        detection_logits = outputs.get('detection_logits', None)
        if detection_logits is not None:
            bce_loss = F.binary_cross_entropy_with_logits(detection_logits, labels.float())
            loss_dict['bce_loss'] = bce_loss
        
        if masks is not None:
            pred_mask = outputs.get('pred_mask', None)
            if pred_mask is not None:
                dice_loss = self._dice_loss(pred_mask, masks)
                loss_dict['dice_loss'] = dice_loss
        
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -&gt; torch.Tensor:
        """
        计算 DICE Loss
        
        Args:
            pred: 预测掩码
            target: 真实掩码
            smooth: 平滑项
        
        Returns:
            torch.Tensor: DICE 损失
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def compute_loss_stage2(
        self,
        outputs: Dict[str, Any],
        labels: Dict[str, torch.Tensor]
    ) -&gt; Dict[str, torch.Tensor]:
        """
        计算 Stage 2 损失：Causal Language Modeling Loss
        
        Args:
            outputs: 模型输出
            labels: 标签字典
        
        Returns:
            Dict: 损失字典
        """
        logits = outputs.get('logits', None)
        target_ids = labels.get('input_ids', None)
        
        if logits is not None and target_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            clm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return {'clm_loss': clm_loss, 'total_loss': clm_loss}
        
        return {'total_loss': torch.tensor(0.0, device=self.device)}
    
    def compute_loss_stage3(
        self,
        outputs: Dict[str, Any],
        labels: Dict[str, Any]
    ) -&gt; Dict[str, torch.Tensor]:
        """
        计算 Stage 3 损失：DPO Loss
        
        Args:
            outputs: 模型输出
            labels: 标签字典 (包含 winner 和 loser)
        
        Returns:
            Dict: 损失字典
        """
        winner_log_probs = outputs.get('winner_log_probs', None)
        loser_log_probs = outputs.get('loser_log_probs', None)
        
        if winner_log_probs is not None and loser_log_probs is not None:
            beta = 0.1
            dpo_loss = -F.logsigmoid(beta * (winner_log_probs - loser_log_probs)).mean()
            return {'dpo_loss': dpo_loss, 'total_loss': dpo_loss}
        
        return {'total_loss': torch.tensor(0.0, device=self.device)}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        checkpoint_path: Optional[str] = None
    ) -&gt; None:
        """
        训练主循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            checkpoint_path: 断点续训路径
        """
        self.logger.info(f"Starting training stage {self.stage}")
        
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path, optimizer, scheduler)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self._train_epoch(train_loader, optimizer)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                
                if val_loss &lt; self.best_metric:
                    self.best_metric = val_loss
                    self._save_checkpoint(optimizer, scheduler, is_best=True)
            
            scheduler.step()
            self._save_checkpoint(optimizer, scheduler, is_best=False)
    
    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -&gt; float:
        """
        训练一个 epoch
        
        Args:
            loader: 数据加载器
            optimizer: 优化器
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Training Stage {self.stage}")
        
        for batch in progress_bar:
            images, labels, annotation_info, text_prompts = batch
            images = images.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(images, text_prompts[0])
            
            if self.stage == 1:
                loss_dict = self.compute_loss_stage1(outputs, labels)
            elif self.stage == 2:
                loss_dict = self.compute_loss_stage2(outputs, annotation_info)
            elif self.stage == 3:
                loss_dict = self.compute_loss_stage3(outputs, annotation_info)
            
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader: DataLoader) -&gt; float:
        """
        验证一个 epoch
        
        Args:
            loader: 数据加载器
        
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                images, labels, annotation_info, text_prompts = batch
                images = images.to(self.device)
                
                outputs = self.model(images, text_prompts[0])
                
                if self.stage == 1:
                    loss_dict = self.compute_loss_stage1(outputs, labels)
                elif self.stage == 2:
                    loss_dict = self.compute_loss_stage2(outputs, annotation_info)
                elif self.stage == 3:
                    loss_dict = self.compute_loss_stage3(outputs, annotation_info)
                
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(loader)
    
    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        is_best: bool = False
    ) -&gt; None:
        """
        保存检查点
        
        Args:
            optimizer: 优化器
            scheduler: 学习率调度器
            is_best: 是否为最优模型
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': self.best_metric,
            'stage': self.stage
        }
        
        save_path = os.path.join(
            self.path_config.weights_dir,
            f'checkpoint_stage{self.stage}_latest.pt'
        )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
        
        if is_best:
            best_path = os.path.join(
                self.path_config.weights_dir,
                f'checkpoint_stage{self.stage}_best.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved to {best_path}")
    
    def _load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -&gt; None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            optimizer: 优化器
            scheduler: 学习率调度器
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
