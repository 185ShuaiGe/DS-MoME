
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
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
        
        self.tokenizer = None
        self._init_tokenizer()
        
        self.best_metric = -float('inf')
        self.global_step = 0
        self.epoch = 0
        
        self._setup_stage()
    
    def _init_tokenizer(self) -&gt; None:
        """
        初始化 Tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}")
    
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
            self.logger.info("Stage 3: DPO training, keeping previous setup")
    
    def _freeze_clip_llm(self) -&gt; None:
        """
        冻结 CLIP 语义流和 LLM 主干网络 (Stage 1)
        """
        self.logger.info("Freezing CLIP semantic stream and LLM backbone")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            if hasattr(self.model.dual_stream_encoder, 'semantic_stream'):
                for param in self.model.dual_stream_encoder.semantic_stream.parameters():
                    param.requires_grad = False
        
        if hasattr(self.model, 'llm_infer'):
            if hasattr(self.model.llm_infer, 'llm_model'):
                for param in self.model.llm_infer.llm_model.parameters():
                    param.requires_grad = False
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after freezing: {trainable_count:,}")
    
    def _unfreeze_artifact_adapter(self) -&gt; None:
        """
        解冻伪影流和交叉注意力适配器 (Stage 1)
        """
        self.logger.info("Unfreezing artifact stream and cross-attention adapter")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            if hasattr(self.model.dual_stream_encoder, 'artifact_stream'):
                for param in self.model.dual_stream_encoder.artifact_stream.parameters():
                    param.requires_grad = True
        
        if hasattr(self.model, 'forensic_cross_attention'):
            for param in self.model.forensic_cross_attention.parameters():
                param.requires_grad = True
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after unfreezing: {trainable_count:,}")
    
    def _freeze_visual_streams(self) -&gt; None:
        """
        冻结所有视觉流 (Stage 2)
        """
        self.logger.info("Freezing all visual streams")
        
        if hasattr(self.model, 'dual_stream_encoder'):
            for param in self.model.dual_stream_encoder.parameters():
                param.requires_grad = False
        
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters after freezing visual streams: {trainable_count:,}")
    
    def _apply_lora(self) -&gt; None:
        """
        为 LLM 应用 LoRA (Stage 2)
        """
        self.logger.info(f"Applying LoRA with rank={self.model_config.lora_rank}")
        
        if hasattr(self.model, 'llm_infer') and hasattr(self.model.llm_infer, 'llm_model'):
            llm_model = self.model.llm_infer.llm_model
            
            if llm_model is not None:
                lora_config = LoraConfig(
                    r=self.model_config.lora_rank,
                    lora_alpha=self.model_config.lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                try:
                    self.model.llm_infer.llm_model = get_peft_model(llm_model, lora_config)
                    self.model.llm_infer.lora_config = lora_config
                    self.logger.info("LoRA applied successfully")
                    
                    trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    self.logger.info(f"Trainable parameters with LoRA: {trainable_count:,}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply LoRA: {e}")
    
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
            labels = labels.to(self.device).float()
            if detection_logits.dim() &gt; 1:
                detection_logits = detection_logits.view(-1)
            bce_loss = F.binary_cross_entropy_with_logits(detection_logits, labels)
            loss_dict['bce_loss'] = bce_loss
        
        if masks is not None:
            pred_mask = outputs.get('pred_mask', None)
            if pred_mask is not None:
                dice_loss = self._dice_loss(pred_mask, masks.to(self.device))
                loss_dict['dice_loss'] = dice_loss
        
        if not loss_dict:
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict['dummy_loss'] = dummy_loss
        
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
        labels: Dict[str, Any],
        tokenizer: Optional[Any] = None
    ) -&gt; Dict[str, torch.Tensor]:
        """
        计算 Stage 2 损失：Causal Language Modeling Loss
        
        Args:
            outputs: 模型输出
            labels: 标签字典
            tokenizer: 可选的 tokenizer
        
        Returns:
            Dict: 损失字典
        """
        logits = outputs.get('logits', None)
        
        if logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            
            expert_text = labels.get('expert_explanation', '')
            if tokenizer is not None and expert_text:
                tokenized = tokenizer(
                    expert_text,
                    max_length=self.model_config.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                target_ids = tokenized['input_ids'].to(self.device)
                shift_labels = target_ids[..., 1:].contiguous()
                
                clm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                return {'clm_loss': clm_loss, 'total_loss': clm_loss}
        
        return {'total_loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
    
    def compute_loss_stage3(
        self,
        outputs_winner: Dict[str, Any],
        outputs_loser: Dict[str, Any],
        beta: float = 0.1
    ) -&gt; Dict[str, torch.Tensor]:
        """
        计算 Stage 3 损失：DPO Loss
        
        Args:
            outputs_winner: Winner 回答的模型输出
            outputs_loser: Loser 回答的模型输出
            beta: DPO beta 参数
        
        Returns:
            Dict: 损失字典
        """
        winner_log_probs = outputs_winner.get('log_probs', None)
        loser_log_probs = outputs_loser.get('log_probs', None)
        
        if winner_log_probs is not None and loser_log_probs is not None:
            dpo_loss = -F.logsigmoid(beta * (winner_log_probs - loser_log_probs)).mean()
            return {'dpo_loss': dpo_loss, 'total_loss': dpo_loss}
        
        return {'total_loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
    
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
                
                if val_loss &lt; self.best_metric or self.best_metric == -float('inf'):
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
            
            batch_size_local = images.size(0)
            batch_outputs = []
            batch_losses = []
            
            for i in range(batch_size_local):
                single_image = images[i:i+1]
                single_prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts
                
                outputs = self.model(single_image, single_prompt)
                batch_outputs.append(outputs)
                
                if self.stage == 1:
                    single_label = labels[i:i+1] if isinstance(labels, torch.Tensor) else [labels[i]]
                    single_label_tensor = torch.tensor([single_label], device=self.device) if not isinstance(labels, torch.Tensor) else single_label
                    loss_dict = self.compute_loss_stage1(outputs, single_label_tensor)
                elif self.stage == 2:
                    single_info = {k: v[i] if isinstance(v, (list, torch.Tensor)) else v for k, v in annotation_info.items()}
                    loss_dict = self.compute_loss_stage2(outputs, single_info, self.tokenizer)
                elif self.stage == 3:
                    single_info = {k: v[i] if isinstance(v, (list, torch.Tensor)) else v for k, v in annotation_info.items()}
                    winner_text = single_info.get('winner', '')
                    loser_text = single_info.get('loser', '')
                    
                    outputs_winner = self.model(single_image, winner_text)
                    outputs_loser = self.model(single_image, loser_text)
                    loss_dict = self.compute_loss_stage3(outputs_winner, outputs_loser)
                
                batch_losses.append(loss_dict['total_loss'])
            
            loss = torch.mean(torch.stack(batch_losses))
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
                
                batch_size_local = images.size(0)
                batch_losses = []
                
                for i in range(batch_size_local):
                    single_image = images[i:i+1]
                    single_prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts
                    
                    outputs = self.model(single_image, single_prompt)
                    
                    if self.stage == 1:
                        single_label = labels[i:i+1] if isinstance(labels, torch.Tensor) else [labels[i]]
                        single_label_tensor = torch.tensor([single_label], device=self.device) if not isinstance(labels, torch.Tensor) else single_label
                        loss_dict = self.compute_loss_stage1(outputs, single_label_tensor)
                    elif self.stage == 2:
                        single_info = {k: v[i] if isinstance(v, (list, torch.Tensor)) else v for k, v in annotation_info.items()}
                        loss_dict = self.compute_loss_stage2(outputs, single_info, self.tokenizer)
                    else:
                        loss_dict = {'total_loss': torch.tensor(0.0, device=self.device)}
                    
                    batch_losses.append(loss_dict['total_loss'])
                
                loss = torch.mean(torch.stack(batch_losses))
                total_loss += loss.item()
        
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
