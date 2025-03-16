#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import psycopg2
import numpy as np
import argparse
import socket
import json
import pickle
import logging
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from typing import List, Dict, Tuple, Any, Optional, Generator
from multiprocessing import Pool, cpu_count, current_process
from itertools import chain
from contextlib import contextmanager
import gc
import queue
from threading import Thread, Event, RLock
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import re
import math
import random
import psutil
import zmq  # 添加ZeroMQ用于节点间通信

# 设置环境变量以禁用进度条
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"  # 禁用所有 tqdm 进度条

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 分布式角色定义
COORDINATOR = 'coordinator'  # 协调节点
WORKER = 'worker'  # 工作节点

# M4系统优化函数保持不变
def optimize_system_for_m4():
    """Optimize system settings for M4 chip"""
    import platform
    
    # Only apply these optimizations on macOS with Apple Silicon
    if platform.system() != 'Darwin' or 'arm64' not in platform.machine():
        print("⚠️ 非 Apple Silicon Mac，跳过系统优化")
        return
    
    print("🔧 应用 M4 系统优化...")
    
    # 1. Set process priority (nice) if running with proper permissions
    try:
        os.nice(-10)  # Higher priority (lower nice value)
        print("  ✓ 进程优先级已提高")
    except:
        print("  ⚠️ 无法设置进程优先级")
    
    # 2. Optimize NumPy configuration
    try:
        # Use Apple Accelerate framework for NumPy
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
        print(f"  ✓ NumPy 配置优化，使用 Accelerate 框架，最大线程: {os.cpu_count()}")
    except:
        pass
    
    # 3. Memory allocation tuning
    try:
        os.environ["PYTHONMALLOC"] = "malloc"
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"  # 128KB
        print("  ✓ 内存分配器已优化")
    except:
        pass
    
    # 4. Configure Python GC for high-performance computing
    try:
        import gc
        gc.set_threshold(100000, 5, 5)  # Less frequent collections
        print("  ✓ 垃圾回收器已优化")
    except:
        pass
        
    print("✅ 系统优化完成")

# MPS优化函数保持不变
def enhance_mps_performance():
    """增强M4芯片MPS的性能优化"""
    import torch
    if not torch.backends.mps.is_available():
        print("⚠️ MPS不可用，跳过MPS优化")
        return False
        
    print("🚀 正在应用增强的MPS优化...")
    
    # 1. 设置环境变量以优化MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 主动内存管理
    os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "1"        # 更优的分配策略
    
    # 2. 启用异步MPS执行
    if hasattr(torch.mps, 'enable_async'):
        torch.mps.enable_async(True)
        print("  ✓ 启用MPS异步执行")
    elif hasattr(torch.mps, 'set_async_execution'):
        torch.mps.set_async_execution(True)
        print("  ✓ 启用MPS异步执行")
    
    # 3. 配置Metal编译器优化
    os.environ["MTL_SHADER_VALIDATION"] = "0"    # 禁用shader验证提高性能
    os.environ["MTL_SHADER_CACHE_MODE"] = "11"   # 优化shader缓存
    
    # 4. 预热MPS设备
    try:
        # 创建随机数据来预热图形管道
        for size in [128, 384, 768, 1024]:
            dummy = torch.rand(size, size, device="mps")
            _ = dummy @ dummy.T  # 矩阵乘法预热
            del dummy
            
        # 对于嵌入模型的关键操作预热
        seq_len, dim = 128, 384
        q = torch.rand(1, seq_len, dim, device="mps")
        k = torch.rand(1, seq_len, dim, device="mps")
        v = torch.rand(1, seq_len, dim, device="mps")
        
        # 预热注意力计算
        qk = torch.matmul(q, k.transpose(1, 2))
        qk = qk / math.sqrt(dim)
        attn = torch.nn.functional.softmax(qk, dim=-1)
        _ = torch.matmul(attn, v)
        
        # 清理
        del q, k, v, qk, attn
        torch.mps.empty_cache()
        
        print("  ✓ MPS图形管道预热完成")
    except Exception as e:
        print(f"  ⚠️ MPS预热过程中出错: {e}")
    
    # 5. 配置MPS内存优化
    try:
        # 主动释放内存并尝试保持较低的内存占用
        torch.mps.empty_cache()
        
        # 建议使用的最大内存大小
        # 这需要PyTorch支持，模拟版本
        available_memory = psutil.virtual_memory().available
        suggested_max = min(int(available_memory * 0.7), 8 * 1024 * 1024 * 1024)  # 最多8GB
        
        print(f"  ✓ MPS内存优化配置完成，建议最大使用: {suggested_max/(1024*1024*1024):.1f}GB")
        
        # 设置自定义缓存释放监视器
        class MPSMemoryMonitor:
            def __init__(self, threshold_mb=2000):
                self.threshold_mb = threshold_mb
                self.last_check = time.time()
                self.check_interval = 5  # 秒
                
            def check(self):
                current_time = time.time()
                if current_time - self.last_check < self.check_interval:
                    return
                
                self.last_check = current_time
                try:
                    # 获取MPS分配的内存
                    allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                    if allocated > self.threshold_mb:
                        print(f"  ⚠️ MPS内存使用较高: {allocated:.0f}MB，执行缓存清理")
                        torch.mps.empty_cache()
                except:
                    pass
        
        # 创建监视器
        global mps_memory_monitor
        mps_memory_monitor = MPSMemoryMonitor(threshold_mb=4000)  # 4GB阈值
        print("  ✓ MPS内存监视器已启动")
    except Exception as e:
        print(f"  ⚠️ 配置MPS内存优化时出错: {e}")
    
    return True

# 初始化设备
import torch
if torch.backends.mps.is_available():
    device = "mps"
    print(f"✅ 使用 Apple MPS (M4 GPU) 加速")
    # 应用 MPS 增强优化
    enhance_mps_performance()
    # 配置 PyTorch 线程
    torch.set_num_threads(os.cpu_count() // 2)
    print(f"✓ PyTorch 线程配置: {os.cpu_count() // 2}")
else:
    device = "cpu"
    print("⚠️ MPS 不可用，使用 CPU 计算")

# 内存清理函数保持不变
def clean_memory(force_gc=False):
    """Efficiently clean up memory"""
    import torch
    if device == "mps":
        # Synchronize MPS stream before cleaning
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        torch.mps.empty_cache()
        
        # 检查MPS内存监视器
        if 'mps_memory_monitor' in globals():
            mps_memory_monitor.check()
    
    if force_gc:
        # More aggressive cleanup
        for _ in range(2):  # Multiple collections can help with fragmentation
            gc.collect()

# 优化流水线并发度函数保持不变
def optimize_pipeline_concurrency():
    """根据M4芯片特性优化流水线并发度"""
    import platform
    import multiprocessing as mp
    
    # 检测可用物理核心数
    physical_cores = os.cpu_count()
    
    # 检测是否为Apple Silicon
    is_apple_silicon = (
        platform.system() == 'Darwin' and 
        platform.machine() == 'arm64'
    )
    
    # 对M4芯片特别优化
    if is_apple_silicon:
        # 检测具体芯片型号
        try:
            import subprocess
            chip_info = subprocess.check_output(
                "sysctl -n machdep.cpu.brand_string", 
                shell=True
            ).decode().strip()
            
            is_m4 = "M4" in chip_info
            
            if is_m4:
                print(f"检测到M4芯片: {chip_info}")
                # M4芯片优化配置
                # 1. 文本切分 - CPU密集型，但不需要太多内存
                num_chunk_processes = max(2, physical_cores - 2)
                
                # 2. 嵌入计算 - MPS (GPU) 瓶颈任务
                # M4 MPS支持异步执行但并行性有限
                # 通常2-3个进程能达到最佳平衡
                num_embedding_processes = 2  
                
                # 3. 队列大小优化
                chunk_queue_size = 10000  # 足够大的缓冲区
                result_queue_size = 15000  # 稍大的结果缓冲区
                
                # 4. 数据库并发写入连接
                db_pool_size = min(physical_cores, 6)
                
                # 5. 设置嵌入批处理大小
                embedding_batch_size = 1024  # M4默认为较大批次
            else:
                # 其他Apple Silicon芯片
                print(f"检测到其他Apple Silicon芯片: {chip_info}")
                num_chunk_processes = max(2, physical_cores // 2)
                num_embedding_processes = min(3, physical_cores // 4) 
                chunk_queue_size = 8000
                result_queue_size = 10000
                db_pool_size = min(physical_cores // 2, 4)
                embedding_batch_size = 512
        except:
            # 无法检测具体型号，使用通用配置
            print("使用通用Apple Silicon优化")
            num_chunk_processes = max(2, physical_cores // 2)
            num_embedding_processes = 2
            chunk_queue_size = 8000
            result_queue_size = 10000
            db_pool_size = 4
            embedding_batch_size = 512
    else:
        # 非Apple Silicon优化
        print("使用通用优化配置")
        num_chunk_processes = max(2, physical_cores // 2)
        num_embedding_processes = min(2, max(1, physical_cores // 4))
        chunk_queue_size = 5000
        result_queue_size = 8000
        db_pool_size = min(physical_cores // 2, 8)
        embedding_batch_size = 512
    
    # 打印优化配置
    print(f"📊 优化配置:")
    print(f"  → 文本切分进程数: {num_chunk_processes}")
    print(f"  → 嵌入计算进程数: {num_embedding_processes}")
    print(f"  → 数据库连接池大小: {db_pool_size}")
    print(f"  → 嵌入批处理大小: {embedding_batch_size}")
    
    return {
        'num_chunk_processes': num_chunk_processes,
        'num_embedding_processes': num_embedding_processes,
        'chunk_queue_size': chunk_queue_size,
        'result_queue_size': result_queue_size,
        'db_pool_size': db_pool_size,
        'embedding_batch_size': embedding_batch_size
    }

# 嵌入缓存优化类保持不变
class EmbeddingCache:
    """为常见文本片段提供嵌入缓存以减少计算"""
    
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, text):
        """从缓存获取嵌入"""
        text_hash = hash(text)
        with self.lock:
            if text_hash in self.cache:
                self.hits += 1
                return self.cache[text_hash]
            self.misses += 1
            return None
    
    def put(self, text, embedding):
        """向缓存添加嵌入"""
        text_hash = hash(text)
        with self.lock:
            # 如果缓存满，移除随机项
            if len(self.cache) >= self.max_size:
                # 随机移除10%的缓存以腾出空间
                keys_to_remove = random.sample(
                    list(self.cache.keys()), 
                    k=max(1, int(self.max_size * 0.1))
                )
                for key in keys_to_remove:
                    del self.cache[key]
                    
            self.cache[text_hash] = embedding
    
    def stats(self):
        """返回缓存统计信息"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }
            
    def clear(self):
        """清除缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

# 导入 huggingface 相关库
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# 导入 SentenceTransformer 并指定使用 MPS 设备
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# 优化版本的模型加载函数保持不变
def load_optimized_model(model_name, device):
    """使用半精度加载模型以减少内存占用并提高速度"""
    import torch
    
    print(f"正在优化加载模型到 {device} 设备...")
    
    # 使用fp16加载模型
    model = SentenceTransformer(model_name, device=device)
    
    # 如果设备支持，转换为半精度
    if device == "mps":
        # 确保MPS支持fp16
        if hasattr(torch.ops.mps, '_has_fp16_support') and torch.ops.mps._has_fp16_support():
            # 将模型转换为fp16
            model.half()
            print("✅ 模型已转换为半精度(fp16)")
        
        # 如果支持特定的MPS图像集优化
        if hasattr(torch.mps, 'optimize_for'):
            # 指定为转换器优化
            torch.mps.optimize_for(model, category="transformer_inference")
            print("✅ 使用MPS图优化器优化")
    
    return model

# 添加分布式通信工具类
class DistributedComm:
    """分布式节点间通信类"""
    
    def __init__(self, role, coordinator_ip='localhost', coordinator_port=5555, worker_port=5556):
        self.role = role
        self.coordinator_ip = coordinator_ip
        self.coordinator_port = coordinator_port
        self.worker_port = worker_port
        self.context = zmq.Context()
        self.sockets = {}
        
        # 初始化通信
        self._init_communication()
        
    def _init_communication(self):
        """根据节点角色初始化通信套接字"""
        if self.role == COORDINATOR:
            # 协调器需要一个ROUTER套接字来接收工作节点的连接
            self.sockets['workers'] = self.context.socket(zmq.ROUTER)
            self.sockets['workers'].bind(f"tcp://*:{self.coordinator_port}")
            
            # 协调器还需要一个DEALER套接字向工作节点发送任务
            self.sockets['tasks'] = self.context.socket(zmq.DEALER)
            self.sockets['tasks'].bind(f"tcp://*:{self.worker_port}")
            
            print(f"协调器监听在端口 {self.coordinator_port}(workers) 和 {self.worker_port}(tasks)")
        else:
            # 工作节点需要连接到协调器
            self.sockets['coordinator'] = self.context.socket(zmq.DEALER)
            # 为这个工作节点生成唯一标识
            identity = f"worker-{socket.gethostname()}-{os.getpid()}"
            self.sockets['coordinator'].setsockopt_string(zmq.IDENTITY, identity)
            self.sockets['coordinator'].connect(f"tcp://{self.coordinator_ip}:{self.coordinator_port}")
            
            # 工作节点还需要一个套接字来接收任务
            self.sockets['tasks'] = self.context.socket(zmq.DEALER)
            self.sockets['tasks'].connect(f"tcp://{self.coordinator_ip}:{self.worker_port}")
            
            print(f"工作节点 {identity} 已连接到协调器 {self.coordinator_ip}")

    def send_message(self, socket_name, message):
        """发送消息到指定套接字"""
        try:
            # Convert to JSON string
            json_data = json.dumps(message)
            #print(f"Sending message to {socket_name}: {json_data[:100]}...")  # Only print first 100 chars
            self.sockets[socket_name].send_string(json_data)
            return True
        except Exception as e:
            print(f"发送消息时出错: {e}")
            return False

    def receive_message(self, socket_name, timeout=1000):
        """从指定套接字接收消息，可设置超时(毫秒)"""
        try:
            # 设置接收超时
            self.sockets[socket_name].setsockopt(zmq.RCVTIMEO, timeout)
            
            # Check if this is a ROUTER socket
            if self.role == COORDINATOR and socket_name == 'workers':
                # For ROUTER sockets, receive both identity and message parts
                identity = self.sockets[socket_name].recv_string()
                message = self.sockets[socket_name].recv_string()
                print(f"Received message from identity {identity}")
                
                if not message or message.strip() == "":
                    return None
                    
                return json.loads(message)
            else:
                # For non-ROUTER sockets, just receive the message
                message = self.sockets[socket_name].recv_string()
                
                if not message or message.strip() == "":
                    return None
                    
                return json.loads(message)
        except zmq.error.Again:
            # 超时，没有收到消息
            return None
        except Exception as e:
            print(f"接收消息时出错: {e}")
            if 'message' in locals():
                print(f"Raw message content (first 100 chars): {repr(message[:100] if message else 'None')}")
            return None
    def close(self):
        """关闭所有套接字"""
        for socket in self.sockets.values():
            socket.close()
        self.context.term()
        print("通信资源已释放")

# 修改向量处理器类以支持分布式
class PgVectorProcessor:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.pool = None
        self.vector_dim = 384
        self.chunk_size = 128  # 较大的文本块以减少总块数
        self.batch_size = 1024  # 增加批处理大小以提高 M4 性能
        
        print(f"正在加载模型到 {device} 设备...")
        
        # 使用指定的模型 - sentence-transformers/all-MiniLM-L6-v2
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        try:
            # 使用优化的模型加载
            self.model = load_optimized_model(model_name, device)
            print(f"✅ 成功加载模型: {model_name}")
            
            # 尝试使用 MPS 加速
            if device == "mps":
                print(f"✅ 模型成功加载到 MPS 设备")
                
                # 尝试预热模型
                dummy_texts = ["This is a warm up text to initialize the model pipeline."] * 4
                _ = self.model.encode(dummy_texts, convert_to_numpy=True)
                print(f"✅ 模型预热完成")
        except Exception as e:
            print(f"⚠️ 模型加载到 {device} 失败: {e}，回退到 CPU")
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✅ 成功加载模型到 CPU: {model_name}")
            except Exception as e:
                print(f"⚠️ 模型加载完全失败: {e}")
                raise
        
        # 加载对应的 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ 成功加载对应 tokenizer")
        except Exception as e:
            print(f"⚠️ 加载 tokenizer 失败: {e}，使用备用方法")
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # 启用 MPS 图捕获 (如果可用)
        if device == "mps" and hasattr(torch.mps, 'enable_graph_capture'):
            try:
                torch.mps.enable_graph_capture()
                print(f"✅ 启用 MPS 图捕获以提高性能")
            except:
                pass
        
        print(f"模型加载完成，使用的设备: {self.model.device}")
        
        # 初始化连接池
        self.init_connection_pool()
        
        # 创建嵌入缓存
        self.embedding_cache = EmbeddingCache(max_size=50000)  # 缓存最多50,000个嵌入

    def init_connection_pool(self, min_conn=2, max_conn=10):
        """Initialize a connection pool for better database performance"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                **self.db_config
            )
            logger.info(f"数据库连接池初始化成功 (min={min_conn}, max={max_conn})")
            return True
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            return False

    def process_text(self, text: str) -> List[str]:
        """Improved text chunking with better semantic boundaries"""
        if not text or text.strip() == "":
            return []
        
        # 1. 清理和规范化文本
        text = text.replace('\r', ' ').replace('\t', ' ')
        
        # 2. 尝试使用自然句子边界进行分块
        
        # 按句子边界分割（原则上）
        # 此正则表达式匹配句子结尾 - 句号、问号、感叹号后跟空格
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        
        # 分割为句子
        sentences = sentence_endings.split(text)
        
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 3. 根据令牌计数将句子分组成块
        chunks = []
        current_chunk = []
        current_length = 0
        target_chunk_size = self.chunk_size
        overlap_tokens = 20  # 较小的重叠以保持上下文连续性
        
        for sentence in sentences:
            # 对句子进行分词
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_length = len(sentence_tokens)
            
            # 通过分割来跳过极长的单句
            if sentence_length > target_chunk_size:
                if current_chunk:
                    # 编码和解码以获得适当的标记边界
                    chunk_text = self.tokenizer.decode(
                        self.tokenizer.encode(' '.join(current_chunk))
                    )
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                # 将长句处理成具有重叠的多个块
                for i in range(0, sentence_length, target_chunk_size - overlap_tokens):
                    sub_tokens = sentence_tokens[i:i + target_chunk_size]
                    sub_text = self.tokenizer.decode(sub_tokens)
                    chunks.append(sub_text)
                
                continue
            
            # 检查添加此句是否会超出块大小
            if current_length + sentence_length > target_chunk_size:
                # 当前块已满，将其添加到块中并开始新块
                if current_chunk:
                    chunk_text = self.tokenizer.decode(
                        self.tokenizer.encode(' '.join(current_chunk))
                    )
                    chunks.append(chunk_text)
                
                # 用此句开始新块
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # 添加到当前块
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 添加剩余的块
        if current_chunk:
            chunk_text = self.tokenizer.decode(
                self.tokenizer.encode(' '.join(current_chunk))
            )
            chunks.append(chunk_text)
        
        # 确保即使对于很短的文本也至少有一个块
        if not chunks and text.strip():
            chunks.append(text.strip())
        
        return chunks

    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            if self.conn is not None and not self.conn.closed:
                logger.info("已存在有效的数据库连接")
                return True
                
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
                logger.info("数据库连接已关闭")
            
            # 关闭连接池
            if self.pool:
                self.pool.closeall()
                logger.info("数据库连接池已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")

    @contextmanager
    def get_connection(self):
        """使用上下文管理器管理数据库连接"""
        connection = None
        connection_created = False
        
        try:
            # 尝试从连接池获取连接
            if self.pool:
                connection = self.pool.getconn()
                yield connection
            else:
                # 回退到常规连接
                if not self.conn or self.conn.closed:
                    if not self.connect():
                        raise RuntimeError("无法建立数据库连接")
                    connection_created = True
                
                yield self.conn
        finally:
            # 归还连接到池
            if connection and self.pool:
                self.pool.putconn(connection)
            elif connection_created:
                self.close()

    def check_vector_extension(self) -> bool:
        """检查 PostgreSQL vector 扩展是否已安装"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                    if cur.fetchone() is None:
                        logger.error("PostgreSQL 'vector' 扩展未安装")
                        return False
                    logger.info("PostgreSQL 'vector' 扩展已安装")
                    return True
        except Exception as e:
            logger.error(f"检查 vector 扩展时出错: {e}")
            return False

    def setup_vector_db(self, table_name: str) -> bool:
        """设置向量数据库表及索引"""
        if not self.check_vector_extension():
            logger.error("未找到必要的 vector 扩展，无法创建向量表")
            return False
            
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 检查 schema 是否存在
                    cur.execute("""
                    SELECT schema_name FROM information_schema.schemata 
                    WHERE schema_name = 'device'
                    """)
                    if cur.fetchone() is None:
                        logger.error("device schema 不存在")
                        return False
                        
                    cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        report_number TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        text_chunk TEXT NOT NULL,
                        embedding public.vector({self.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(chunk_id)
                    );
                    """)
                    cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                    ON {table_name} USING ivfflat (embedding public.vector_cosine_ops)
                    WITH (lists = 100);
                    """)
                    conn.commit()
                    logger.info(f"向量表 {table_name} 创建成功")
                    return True
        except Exception as e:
            logger.error(f"设置向量数据库失败: {e}")
            return False

    def get_embeddings_batch_optimized(self, texts: List[str]) -> List[List[float]]:
        """优化的批量嵌入生成，自适应批处理大小"""
        if not texts:
            return []
            
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.vector_dim for _ in range(len(texts))]
        
        try:
            # 跟踪原始索引以便对齐
            indices = [i for i, t in enumerate(texts) if t and t.strip()]
            
            # 准备规范化输入
            normalized_texts = []
            for t in valid_texts:
                # 更高效的文本规范化
                t = re.sub(r'\s+', ' ', t).strip()
                normalized_texts.append(t)
            
            # 动态批处理大小 - 基于文本长度和可用内存
            avg_token_len = sum(len(t.split()) for t in normalized_texts) / len(normalized_texts)
            
            # 针对M4芯片优化的批处理大小
            if device == "mps":
                # 根据平均文本长度动态调整批大小
                if avg_token_len < 50:
                    batch_size = 2048  # 短文本使用更大批次
                elif avg_token_len < 100:
                    batch_size = 1024  # 中等长度文本
                else:
                    batch_size = 512   # 长文本使用小批次
            else:
                # CPU处理使用较小批次避免内存问题
                batch_size = min(512, max(64, int(10000 / avg_token_len)))
                
            # 使用固定长度填充优化
            # 按长度分组，相似长度文本一起处理可减少填充开销
            length_groups = {}
            for i, text in enumerate(normalized_texts):
                text_len = len(text.split())
                length_bucket = text_len // 20 * 20  # 分组为20词一个桶
                if length_bucket not in length_groups:
                    length_groups[length_bucket] = []
                length_groups[length_bucket].append((i, text))
            
            # 按长度分组处理
            all_embeddings = [None] * len(normalized_texts)
            for length_bucket in sorted(length_groups.keys()):
                bucket_indices, bucket_texts = zip(*length_groups[length_bucket])
                
                # 使用normalize_embeddings=True自动归一化,提高性能
                # convert_to_tensor=True避免CPU-GPU转换开销
                bucket_embeddings = self.model.encode(
                    list(bucket_texts),
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                
                # 转换为numpy以便进一步处理(如果尚未转换)
                if hasattr(bucket_embeddings, 'cpu'):
                    bucket_embeddings = bucket_embeddings.cpu().numpy()
                    
                # 将组内嵌入放回相应位置
                for idx, emb in zip(bucket_indices, bucket_embeddings):
                    all_embeddings[idx] = emb
                    
            # 构建最终结果
            result = [[0.0] * self.vector_dim for _ in range(len(texts))]
            for orig_idx, emb_idx in enumerate(indices):
                # 确保是列表格式
                emb = all_embeddings[orig_idx]
                emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                # 处理NaN值
                if any(np.isnan(x) for x in emb_list):
                    emb_list = [0.0 if np.isnan(x) else x for x in emb_list]
                result[emb_idx] = emb_list
                
            # 手动触发MPS缓存清理
            if device == "mps":
                torch.mps.empty_cache()
                
            return result
        except Exception as e:
            logger.error(f"批量获取嵌入向量时出错: {e}")
            # 出错时返回零向量
            return [[0.0] * self.vector_dim for _ in range(len(texts))]
    
    # 向后兼容的API
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """优化的批量嵌入生成，兼容性更好"""
        # 先检查缓存
        cached_results = []
        uncached_indices = []
        uncached_texts = []
        
        # 跟踪哪些文本需要计算
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_results.append([0.0] * self.vector_dim)
                continue
                
            cached_emb = self.embedding_cache.get(text)
            if cached_emb is not None:
                cached_results.append(cached_emb)
            else:
                cached_results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # 如果所有文本都在缓存中，直接返回
        if not uncached_texts:
            return cached_results
            
        # 处理未缓存的嵌入
        uncached_embeddings = self.get_embeddings_batch_optimized(uncached_texts)
        
        # 将未缓存的嵌入添加到缓存
        for i, (idx, emb) in enumerate(zip(uncached_indices, uncached_embeddings)):
            if i < len(uncached_texts):
                self.embedding_cache.put(uncached_texts[i], emb)
                cached_results[idx] = emb
        
        return cached_results

    def store_vectors(self, table_name: str, data: List[Tuple]) -> bool:
        """使用 COPY 命令优化向量存储"""
        if not data:
            logger.warning("没有向量数据需要存储")
            return True
            
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 优化数据库写入
                    
                    # 1. 使用 COPY 代替 INSERT 进行批量操作
                    from io import StringIO
                    
                    # 创建具有相同结构的临时表
                    temp_table = f"temp_{table_name}_{int(time.time())}"
                    cur.execute(f"""
                    CREATE TEMPORARY TABLE {temp_table} (
                        report_number TEXT,
                        chunk_id TEXT,
                        text_chunk TEXT,
                        embedding public.vector({self.vector_dim})
                    ) ON COMMIT DROP;
                    """)
                    
                    # 准备数据进行 COPY
                    buffer = StringIO()
                    for report_number, chunk_id, text_chunk, embedding in data:
                        # 将嵌入格式化为 PostgreSQL 向量字符串
                        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                        # 转义分隔符和引号
                        text = text_chunk.replace('\t', ' ').replace('\n', ' ').replace('\\', '\\\\')
                        buffer.write(f"{report_number}\t{chunk_id}\t{text}\t{embedding_str}\n")
                    
                    buffer.seek(0)
                    
                    # 使用 COPY 进行更快的插入
                    cur.copy_expert(f"COPY {temp_table} FROM STDIN", buffer)
                    
                    # 从临时表插入到主表，处理冲突
                    cur.execute(f"""
                    INSERT INTO {table_name} (report_number, chunk_id, text_chunk, embedding)
                    SELECT report_number, chunk_id, text_chunk, embedding 
                    FROM {temp_table}
                    ON CONFLICT (chunk_id) DO NOTHING;
                    """)
                    
                    conn.commit()
                return True
        except Exception as e:
            logger.error(f"存储向量失败: {e}")
            # 如果优化版本失败，回退到原始方法
            try:
                # 回退到使用 execute_values
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        # 使用更小的批次执行插入操作
                        batch_size = 5000
                        for i in range(0, len(data), batch_size):
                            batch = data[i:i + batch_size]
                            execute_values(cur, f"""
                            INSERT INTO {table_name} 
                            (report_number, chunk_id, text_chunk, embedding)
                            VALUES %s
                            ON CONFLICT (chunk_id) DO NOTHING;
                            """, batch)
                            
                            # 提交当前批次，减少事务大小
                            conn.commit()
                return True
            except:
                return False

# 文本切分的工作函数 - 提前定义以供调用
def chunk_process_worker(args: Tuple[List[Tuple[str, str]], dict]) -> List[Tuple[str, int, str]]:
    """处理文本切分的工作函数"""
    batch, db_config = args
    processor = PgVectorProcessor(db_config)
    results = []
    
    try:
        for report_number, text in batch:
            if not text:
                continue
            chunks = processor.process_text(text)
            for i, chunk in enumerate(chunks):
                if not chunk or chunk.strip() == "":
                    continue
                # 只返回必要的数据：报告号、块索引、文本内容
                results.append((str(report_number), i+1, chunk))
                
                # 频繁释放内存
                if len(results) % 10000 == 0:
                    # 强制清理内存
                    clean_memory(force_gc=False)
    except Exception as e:
        print(f"文本切分处理时出错: {e}")
    
    # 清理资源
    del processor
    clean_memory(force_gc=True)
    
    return results

# 修改的数据获取函数，支持分区
def fetch_partitioned_data(conn, partition_id, total_partitions, limit=500000, batch_size=20000) -> Generator[List[Tuple[str, str]], None, None]:
    """分区获取测试数据"""
    try:
        with conn.cursor() as cur:
            # 创建临时表进行分区
            cur.execute("""
            CREATE TEMPORARY TABLE temp_partitioned_event_texts AS
            SELECT id as report_number, text,
                   ROW_NUMBER() OVER (ORDER BY id) as row_num
            FROM device.event_texts 
            WHERE text IS NOT NULL 
            LIMIT %s
            """, (limit,))
            
            # 获取总行数
            cur.execute("SELECT COUNT(*) FROM temp_partitioned_event_texts")
            total_rows = cur.fetchone()[0]
            
            # 计算每个分区的大小
            partition_size = total_rows // total_partitions
            if total_rows % total_partitions > 0:
                partition_size += 1
            
            # 计算当前分区的开始和结束
            start_row = (partition_id - 1) * partition_size + 1
            end_row = min(partition_id * partition_size, total_rows)
            
            print(f"分区 {partition_id}/{total_partitions}: 处理行 {start_row} 到 {end_row} (共 {end_row-start_row+1} 行)")
            
            # 查询当前分区的数据
            cur.execute("""
            SELECT report_number, text
            FROM temp_partitioned_event_texts
            WHERE row_num BETWEEN %s AND %s
            ORDER BY row_num
            """, (start_row, end_row))
            
            while True:
                records = cur.fetchmany(batch_size)
                if not records:
                    break
                yield records
                
            # 删除临时表
            cur.execute("DROP TABLE temp_partitioned_event_texts")
            
    except Exception as e:
        logger.error(f"获取分区数据时出错: {e}")
        yield []

# 工作节点嵌入处理函数
def worker_embedding_processor(chunk_queue, result_queue, chunk_done, embedding_done, db_config, device, batch_size=1024):
    """工作节点的嵌入处理函数，与原来的优化嵌入处理器基本相同"""
    # 创建处理器和缓存
    embed_processor = PgVectorProcessor(db_config)
    
    # 跟踪统计信息
    processed_chunks = 0
    cache_hits = 0
    start_time = time.time()
    
    # 状态报告频率
    status_interval = 5000
    
    # 创建批次容器
    batch_texts = []
    batch_data = []
    
    # 允许更大的批处理累积
    max_batch_wait = 0.1  # 秒
    last_process_time = time.time()
    
    # 内存使用监控
    last_memory_check = time.time()
    memory_check_interval = 30  # 秒
    
    # 动态调整批处理大小
    current_batch_size = batch_size
    processing_times = []  # 跟踪处理时间
    
    while not (chunk_done.is_set() and chunk_queue.empty()):
        try:
            # 轮询，超时以检查完成标志
            try:
                chunk = chunk_queue.get(timeout=0.05)
                report_number, chunk_idx, text = chunk
                
                # 1. 缓存查找
                cached_embedding = embed_processor.embedding_cache.get(text)
                if cached_embedding is not None:
                    # 缓存命中，直接使用
                    chunk_id = f"{report_number}_{chunk_idx}"
                    result_queue.put((report_number, chunk_id, text, cached_embedding))
                    cache_hits += 1
                    processed_chunks += 1
                    continue
                
                # 2. 添加到批处理
                batch_texts.append(text)
                batch_data.append((report_number, chunk_idx, text))
            except queue.Empty:
                # 队列为空，但不是结束 - 处理任何已累积的批次
                pass
            
            # 决定是否处理当前批次 (满足任一条件):
            # 1. 批次大小达到目标
            # 2. 自上次处理以来已经过去max_batch_wait秒
            # 3. 已经收到完成信号并且队列几乎为空
            current_time = time.time()
            should_process = (
                len(batch_texts) >= current_batch_size or
                (len(batch_texts) > 0 and current_time - last_process_time >= max_batch_wait) or
                (len(batch_texts) > 0 and chunk_done.is_set() and chunk_queue.qsize() < 10)
            )
            
            if should_process:
                # 处理时间测量
                process_start = time.time()
                
                # 处理批次
                if batch_texts:
                    # 使用优化的批处理
                    embeddings = embed_processor.get_embeddings_batch(batch_texts)
                    
                    # 将结果发送到队列并更新缓存
                    for i, (r_num, c_idx, txt) in enumerate(batch_data):
                        if i < len(embeddings):
                            chunk_id = f"{r_num}_{c_idx}"
                            result_queue.put((r_num, chunk_id, txt, embeddings[i]))
                    
                    # 更新统计信息
                    processed_chunks += len(batch_texts)
                    
                    # 清空批次
                    batch_texts = []
                    batch_data = []
                    
                    # 更新处理时间
                    process_time = time.time() - process_start
                    processing_times.append(process_time)
                    
                    # 更新上次处理时间
                    last_process_time = time.time()
                    
                    # 动态批处理大小调整 (保留最新的5次处理)
                    if len(processing_times) > 5:
                        processing_times = processing_times[-5:]
                    
                    # 如果有足够的数据，调整批处理大小
                    if len(processing_times) >= 3:
                        avg_time = sum(processing_times) / len(processing_times)
                        
                        # 根据平均处理时间调整批处理大小
                        if avg_time < 0.5 and current_batch_size < 2048:
                            # 处理速度快，增加批处理大小
                            current_batch_size = min(current_batch_size * 1.2, 2048)
                            current_batch_size = int(current_batch_size)
                        elif avg_time > 2.0 and current_batch_size > 512:
                            # 处理速度慢，减小批处理大小
                            current_batch_size = max(current_batch_size * 0.8, 512)
                            current_batch_size = int(current_batch_size)
                
                # 周期性地清理内存
                if processed_chunks % 5000 == 0 and processed_chunks > 0:
                    # 获取当前内存使用状态
                    if time.time() - last_memory_check >= memory_check_interval:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        
                        print(f"  → 进程内存使用: {memory_mb:.1f}MB, 批处理大小: {current_batch_size}")
                        last_memory_check = time.time()
                        
                        # 如果内存占用过高，清理
                        if memory_mb > 4000:  # 4GB
                            clean_memory(force_gc=True)
                    else:
                        clean_memory(force_gc=False)
            
            # 报告进度
            if processed_chunks > 0 and processed_chunks % status_interval == 0:
                elapsed = time.time() - start_time
                chunks_per_sec = processed_chunks / elapsed
                
                # 缓存统计
                cache_stats = embed_processor.embedding_cache.stats()
                
                print(f"  → 已处理 {processed_chunks} 个嵌入向量 "
                      f"({chunks_per_sec:.1f} 个/秒), "
                      f"队列中还有 {chunk_queue.qsize()} 个文本块")
                print(f"  → 缓存命中率: {cache_stats['hit_rate']*100:.1f}%, "
                      f"命中/未命中: {cache_stats['hits']}/{cache_stats['misses']}, "
                      f"批处理大小: {current_batch_size}")
                
        except Exception as e:
            print(f"嵌入处理错误: {e}")
            # 错误后短暂暂停，避免快速失败循环
            time.sleep(0.5)
            continue
    
    # 处理剩余的批次
    if batch_texts:
        embeddings = embed_processor.get_embeddings_batch(batch_texts)
        for i, (r_num, c_idx, txt) in enumerate(batch_data):
            if i < len(embeddings):
                chunk_id = f"{r_num}_{c_idx}"
                result_queue.put((r_num, chunk_id, txt, embeddings[i]))
    
    # 记录最终统计信息
    elapsed = time.time() - start_time
    if processed_chunks > 0:
        chunks_per_sec = processed_chunks / elapsed
        cache_stats = embed_processor.embedding_cache.stats()
        
        print(f"嵌入处理完成:")
        print(f"  → 总处理量: {processed_chunks} 个嵌入向量")
        print(f"  → 处理速度: {chunks_per_sec:.1f} 个/秒")
        print(f"  → 总耗时: {elapsed:.1f} 秒")
        print(f"  → 缓存命中率: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  → 使用的设备: {device}")
    
    # 标记完成
    embedding_done.set()
    
    # 清理
    del embed_processor
    clean_memory(force_gc=True)

# 协调器节点主函数
def run_coordinator(args):
    """协调器节点主函数"""
    print("=" * 80)
    print(f"启动协调器节点，监听地址: {args.host}:{args.coordinator_port}")
    print("=" * 80)
    
    # 应用M4优化
    optimize_system_for_m4()
    
    # 创建分布式通信对象
    comm = DistributedComm(
        role=COORDINATOR,
        coordinator_ip=args.host,
        coordinator_port=args.coordinator_port,
        worker_port=args.worker_port
    )
    
    # 数据库配置
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'dbname': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'options': '-c search_path=device'
    }
    
    processor = PgVectorProcessor(db_config)
    
    try:
        # 连接数据库
        if not processor.connect():
            logger.error("无法连接到数据库，程序退出")
            return
            
        # 设置向量表
        table_name = args.table_name
        if not processor.setup_vector_db(table_name):
            logger.error("无法设置向量数据库，程序退出")
            return

        print("=" * 80)
        print("开始分布式数据处理...")
        print("=" * 80)
        start_time = time.time()
        
        # 获取优化配置
        pipeline_config = optimize_pipeline_concurrency()
        num_chunk_processes = pipeline_config['num_chunk_processes']
        num_embedding_processes = 1  # 在协调器节点只运行1个嵌入进程
        chunk_queue_size = pipeline_config['chunk_queue_size']
        result_queue_size = pipeline_config['result_queue_size']
        embedding_batch_size = pipeline_config['embedding_batch_size']
        
        # 创建队列用于流水线
        chunk_queue = queue.Queue(maxsize=chunk_queue_size)  # 块创建和嵌入之间的缓冲区
        result_queue = queue.Queue(maxsize=result_queue_size)  # 用于数据库存储的缓冲区
        
        # 用于表示完成的标志
        chunk_done = Event()
        embedding_done = Event()
        
        # 等待工作节点连接
        print("等待工作节点连接...(10秒超时)")
        worker_info = None
        start_wait = time.time()
        while time.time() - start_wait < 10:
            worker_info = comm.receive_message('workers', timeout=1000)
            if worker_info:
                print(f"工作节点已连接: {worker_info}")
                break
            print(".", end="", flush=True)
            time.sleep(1)
        
        if not worker_info:
            print("\n没有工作节点连接，将以单机模式运行")
            # 继续执行单机模式...
        else:
            print(f"\n{len(worker_info) if isinstance(worker_info, list) else 1}个工作节点已连接")
        
        # 获取总数据量
        with processor.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM (
                    SELECT 1 
                    FROM device.event_texts 
                    WHERE text IS NOT NULL 
                    LIMIT %s
                ) t
            """, (args.limit,))
            total_count = cur.fetchone()[0]
        print(f"总共需要处理 {total_count} 条记录\n")
        
        # Producer: 在单独的线程池中切分文本
        def chunk_producer():
            total_records = 0
            total_chunks = 0
            
            # 确定分区，协调器处理第一个分区
            partition_id = 1
            total_partitions = 2  # 假设只有一个工作节点
            
            with ProcessPoolExecutor(max_workers=num_chunk_processes) as chunk_pool:
                for batch_idx, batch in enumerate(fetch_partitioned_data(
                    processor.conn, partition_id, total_partitions, limit=args.limit, batch_size=20000)):
                    if not batch:
                        continue
                    
                    batch_start = time.time()
                    batch_size = len(batch)
                    total_records += batch_size
                    
                    print(f"批次 {batch_idx+1}: 处理 {batch_size} 条记录")
                    
                    # 并行处理块
                    futures = []
                    sub_batch_size = max(1, batch_size // num_chunk_processes)
                    sub_batches = [batch[i:i + sub_batch_size] for i in range(0, batch_size, sub_batch_size)]
                    
                    for sub_batch in sub_batches:
                        future = chunk_pool.submit(chunk_process_worker, (sub_batch, db_config))
                        futures.append(future)
                    
                    # 处理完成的结果（不等待全部完成）
                    for future in futures:
                        chunks = future.result()
                        total_chunks += len(chunks)
                        
                        # 将块添加到队列
                        for chunk in chunks:
                            chunk_queue.put(chunk)
                    
                    # 跟踪进度
                    chunk_time = time.time() - batch_start
                    print(f"  → 批次 {batch_idx+1} 文本切分完成，耗时 {chunk_time:.2f}秒")
                    
                    # 显示总体进度
                    elapsed = time.time() - start_time
                    partition_progress = total_records / (total_count / total_partitions)
                    print(f"\n分区进度: 已处理 {total_records} 条记录 ({partition_progress*100:.1f}%)")
                    if total_records > 0:
                        est_total = elapsed / partition_progress
                        remaining = est_total - elapsed
                        print(f"  → 已用时间: {elapsed/60:.1f}分钟，预计剩余: {remaining/60:.1f}分钟")
                    print(f"  → 已生成 {total_chunks} 个文本块，队列大小: {chunk_queue.qsize()}")
                
                # 标记完成
                chunk_done.set()
                print("所有文本切分任务完成")
        
        # Consumer: 将结果高效存储在数据库中
        def db_consumer():
            # 用于统计的计数器
            stored_count = 0
            batch = []
            
            while not (embedding_done.is_set() and result_queue.empty()):
                try:
                    # 收集批次进行数据库插入
                    while len(batch) < 5000:
                        try:
                            item = result_queue.get(timeout=0.1)
                            batch.append(item)
                            
                            # 如果在清空最后的项目，中断
                            if embedding_done.is_set() and result_queue.empty():
                                break
                        except queue.Empty:
                            # 队列暂时为空但还未结束
                            break
                    
                    # 存储批次
                    if batch:
                        store_start = time.time()
                        if processor.store_vectors(table_name, batch):
                            stored_count += len(batch)
                            print(f"  → 成功存储 {len(batch)} 个向量，总计: {stored_count}，耗时 {time.time() - store_start:.2f}秒")
                        batch = []
                except Exception as e:
                    print(f"存储向量时出错: {e}")
                    time.sleep(0.1)
                    continue
            
            # 存储剩余批次
            if batch:
                if processor.store_vectors(table_name, batch):
                    stored_count += len(batch)
                    print(f"  → 成功存储最后 {len(batch)} 个向量，总计: {stored_count}")
            
            print(f"所有向量存储任务完成，共存储 {stored_count} 个向量")
        
        # 启动流水线线程
        try:
            print("启动分布式流水线处理...")
            # 创建并启动线程
            chunk_thread = Thread(target=chunk_producer)
            db_thread = Thread(target=db_consumer)
            
            chunk_thread.start()
            # 小延迟让块开始积累
            time.sleep(1)
            
            # 创建嵌入处理线程(本地)
            embedding_thread = Thread(
                target=worker_embedding_processor,
                args=(chunk_queue, result_queue, chunk_done, embedding_done, db_config, device, embedding_batch_size)
            )
            embedding_thread.start()
            
            # 小延迟让嵌入开始积累
            time.sleep(1)
            db_thread.start()
            
            # 等待所有线程完成
            chunk_thread.join()
            embedding_thread.join()
            db_thread.join()
            
            # 统计信息输出
            end_time = time.time()
            processing_time = end_time - start_time
            print("\n处理完成:")
            print("=" * 80)
            print(f"总耗时: {processing_time:.2f} 秒 ({processing_time/60:.2f} 分钟)")
            print(f"使用的设备: {device.upper()}")
            print(f"文本切分进程数: {num_chunk_processes}")
            print(f"嵌入计算进程数: {num_embedding_processes}")
            print(f"批处理大小: {embedding_batch_size}")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"流水线处理出错: {e}")
    except Exception as e:
        logger.error(f"程序执行过程中出错: {e}")
    finally:
        processor.close()
        comm.close()

# 工作节点主函数
def run_worker(args):
    """工作节点主函数"""
    print("=" * 80)
    print(f"启动工作节点，连接到协调器: {args.coordinator_host}:{args.coordinator_port}")
    print("=" * 80)
    
    # 应用M4优化
    optimize_system_for_m4()
    
    # 创建分布式通信对象
    comm = DistributedComm(
        role=WORKER,
        coordinator_ip=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        worker_port=args.worker_port
    )
    
    # 数据库配置
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'dbname': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'options': '-c search_path=device'
    }
    
    processor = PgVectorProcessor(db_config)
    
    try:
        # 连接数据库
        if not processor.connect():
            logger.error("无法连接到数据库，程序退出")
            return
        
        # 向协调器注册
        worker_info = {
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'device': device,
            'cores': os.cpu_count(),
            'timestamp': time.time()
        }
        comm.send_message('coordinator', worker_info)
        
        print("向协调器注册成功，等待任务分配...")
        
        # 获取优化配置
        pipeline_config = optimize_pipeline_concurrency()
        num_chunk_processes = pipeline_config['num_chunk_processes']
        num_embedding_processes = pipeline_config['num_embedding_processes']
        chunk_queue_size = pipeline_config['chunk_queue_size']
        result_queue_size = pipeline_config['result_queue_size']
        embedding_batch_size = pipeline_config['embedding_batch_size']
        
        # 创建队列用于流水线
        chunk_queue = queue.Queue(maxsize=chunk_queue_size)
        result_queue = queue.Queue(maxsize=result_queue_size)
        
        # 用于表示完成的标志
        chunk_done = Event()
        embedding_done = Event()
        
        # 工作节点处理第二个分区
        partition_id = 2
        total_partitions = 2
        
        # Producer: 在单独的线程池中切分文本
        def chunk_producer():
            total_records = 0
            total_chunks = 0
            
            with ProcessPoolExecutor(max_workers=num_chunk_processes) as chunk_pool:
                for batch_idx, batch in enumerate(fetch_partitioned_data(
                    processor.conn, partition_id, total_partitions, limit=args.limit, batch_size=20000)):
                    if not batch:
                        continue
                    
                    batch_start = time.time()
                    batch_size = len(batch)
                    total_records += batch_size
                    
                    print(f"批次 {batch_idx+1}: 处理 {batch_size} 条记录")
                    
                    # 并行处理块
                    futures = []
                    sub_batch_size = max(1, batch_size // num_chunk_processes)
                    sub_batches = [batch[i:i + sub_batch_size] for i in range(0, batch_size, sub_batch_size)]
                    
                    for sub_batch in sub_batches:
                        future = chunk_pool.submit(chunk_process_worker, (sub_batch, db_config))
                        futures.append(future)
                    
                    # 处理完成的结果（不等待全部完成）
                    for future in futures:
                        chunks = future.result()
                        total_chunks += len(chunks)
                        
                        # 将块添加到队列
                        for chunk in chunks:
                            chunk_queue.put(chunk)
                    
                    # 跟踪进度
                    chunk_time = time.time() - batch_start
                    print(f"  → 批次 {batch_idx+1} 文本切分完成，耗时 {chunk_time:.2f}秒")
                    print(f"  → 已生成 {total_chunks} 个文本块，队列大小: {chunk_queue.qsize()}")
                
                # 标记完成
                chunk_done.set()
                print("所有文本切分任务完成")
        
        # 工作节点无需数据库写入，结果直接发送到协调器
        def result_sender():
            sent_count = 0
            batch = []
            batch_size = 100  # 发送批次大小
            
            while not (embedding_done.is_set() and result_queue.empty()):
                try:
                    # 收集批次
                    while len(batch) < batch_size:
                        try:
                            item = result_queue.get(timeout=0.1)
                            batch.append(item)
                            
                            # 如果在清空最后的项目，中断
                            if embedding_done.is_set() and result_queue.empty():
                                break
                        except queue.Empty:
                            # 队列暂时为空但还未结束
                            break
                    
                    # 发送批次到协调器
                    if batch:
                        # 使用ZeroMQ发送结果
                        if comm.send_message('tasks', batch):
                            sent_count += len(batch)
                            if sent_count % 1000 == 0:
                                print(f"  → 已发送 {sent_count} 个向量结果到协调器")
                        batch = []
                except Exception as e:
                    print(f"发送结果时出错: {e}")
                    time.sleep(0.1)
                    continue
            
            # 发送剩余批次
            if batch:
                if comm.send_message('tasks', batch):
                    sent_count += len(batch)
            
            print(f"所有结果发送完成，共发送 {sent_count} 个向量结果")
            
            # 通知协调器本节点处理完成
            comm.send_message('coordinator', {'status': 'complete', 'count': sent_count})
        
        # 启动流水线线程
        try:
            print("启动工作节点处理流水线...")
            # 创建并启动线程
            chunk_thread = Thread(target=chunk_producer)
            result_thread = Thread(target=result_sender)
            
            chunk_thread.start()
            # 小延迟让块开始积累
            time.sleep(1)
            
            # 创建嵌入处理线程
            embedding_threads = []
            for i in range(num_embedding_processes):
                thread = Thread(
                    target=worker_embedding_processor,
                    args=(chunk_queue, result_queue, chunk_done, embedding_done, db_config, device, embedding_batch_size)
                )
                thread.start()
                embedding_threads.append(thread)
                # 为每个嵌入线程添加小延迟，避免同时启动造成的资源竞争
                time.sleep(0.5)
            
            # 小延迟让嵌入开始积累
            time.sleep(1)
            result_thread.start()
            
            # 等待所有线程完成
            chunk_thread.join()
            for thread in embedding_threads:
                thread.join()
            result_thread.join()
            
            print("工作节点所有处理完成")
            
        except Exception as e:
            logger.error(f"工作节点处理出错: {e}")
    except Exception as e:
        logger.error(f"程序执行过程中出错: {e}")
    finally:
        processor.close()
        comm.close()

def main():
    parser = argparse.ArgumentParser(description='分布式向量嵌入处理系统')
    parser.add_argument('--role', choices=[COORDINATOR, WORKER], required=True, help='节点角色 (coordinator 或 worker)')
    
    # 数据库参数
    parser.add_argument('--db-host', default='169.254.22.165', help='数据库主机地址')
    parser.add_argument('--db-port', type=int, default=5432, help='数据库端口')
    parser.add_argument('--db-name', default='fda_device', help='数据库名称')
    parser.add_argument('--db-user', default='postgres', help='数据库用户名')
    parser.add_argument('--db-password', default='12345687', help='数据库密码')
    parser.add_argument('--table-name', default='event_text_vectors', help='向量表名称')
    
    # 分布式通信参数
    parser.add_argument('--host', default='0.0.0.0', help='本机IP，协调器节点使用')
    parser.add_argument('--coordinator-host', default='localhost', help='协调器IP，工作节点使用')
    parser.add_argument('--coordinator-port', type=int, default=5555, help='协调器端口')
    parser.add_argument('--worker-port', type=int, default=5556, help='工作节点通信端口')
    
    # 任务参数
    parser.add_argument('--limit', type=int, default=500000, help='处理的最大记录数')
    
    args = parser.parse_args()
    
    if args.role == COORDINATOR:
        run_coordinator(args)
    else:
        run_worker(args)

if __name__ == "__main__":
    main()