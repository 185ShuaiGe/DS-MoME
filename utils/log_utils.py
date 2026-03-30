import logging
import os
import sys
from datetime import datetime

class Logger:
    # 定义一个类变量，用于全局共享同一个文件写入句柄
    _shared_file_handler = None  

    # 类变量，用于标记是否已经挂载了全局异常拦截器
    _excepthook_registered = False

    def __init__(self, name, base_log_dir=None, mode='train', exp_id='final', checkpoint_path=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # 阻止日志向上一级传递，避免在控制台重复打印两次
        self.logger.propagate = False 

        # 确保每个命名的 logger 只添加一次 handlers
        if not self.logger.handlers:
            # 定义统一的日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            # 1. 挂载控制台 Handler（所有 logger 都在控制台输出）
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 2. 只有在传入了 base_log_dir 且尚未创建全局文件句柄时，才进行路径和文件名的分配
            if base_log_dir is not None and Logger._shared_file_handler is None:
                
                # ==== 1：根据模式动态分配日志文件夹 ====
                if mode == 'train':
                    target_dir = os.path.join(base_log_dir, 'Train')
                elif mode == 'val':
                    target_dir = os.path.join(base_log_dir, 'Test')
                elif mode == 'inference':
                    target_dir = os.path.join(base_log_dir, 'Inference')
                else:
                    target_dir = base_log_dir
                    
                os.makedirs(target_dir, exist_ok=True)
                
                # ==== 2：根据模式动态命名日志文件 ====
                if mode == 'train':
                    # 格式：消融模式组别 - 时间戳（只需要月、日、时、分）
                    timestamp = datetime.now().strftime("%m%d-%H%M")
                    filename = f"{exp_id}-{timestamp}.log"
                elif mode == 'val':
                    # 格式：test - 加载的权重名称.log
                    if checkpoint_path:
                        # os.path.basename 会提取文件名，例如 'final_0314-1030-100-0.001.pt'
                        ckpt_basename = os.path.basename(checkpoint_path)
                        ckpt_name, _ = os.path.splitext(ckpt_basename)  # 去掉 .pt 后缀
                    else:
                        ckpt_name = "unknown_weights"
                    filename = f"test-{ckpt_name}.log"
                elif mode == 'inference':
                    timestamp = datetime.now().strftime("%m%d-%H%M")
                    filename = f"inference-{timestamp}.log"
                else:
                    timestamp = datetime.now().strftime("%m%d-%H%M")
                    filename = f"run-{timestamp}.log"
                    
                log_file = os.path.join(target_dir, filename) 
                
                # 创建文件处理器
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(formatter)
                
                # 赋值给类变量，供全局共享
                Logger._shared_file_handler = file_handler

                absolute_log_path = os.path.abspath(log_file)
                print(f"📄 [日志系统初始化] 当前实验日志将实时写入: {absolute_log_path}")

            # 3. 只要全局共享的 File Handler 存在，就挂载到当前实例上
            if Logger._shared_file_handler is not None:
                self.logger.addHandler(Logger._shared_file_handler)

            if not Logger._excepthook_registered:
                self._register_excepthook()
                Logger._excepthook_registered = True

    # 【新增】：定义异常捕获方法
    def _register_excepthook(self):
        def handle_exception(exc_type, exc_value, exc_traceback):
            # 允许键盘中断 (Ctrl+C) 正常退出，不记录为异常报错
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # 使用 logger 的 CRITICAL 级别记录异常，exc_info 会自动抓取完整的 traceback 栈
            self.logger.critical("🚨 程序发生崩溃 (Uncaught Exception):", exc_info=(exc_type, exc_value, exc_traceback))

        # 将自定义的捕获函数赋值给系统默认的异常钩子
        sys.excepthook = handle_exception

    # 代理 logging 的基本方法
    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)