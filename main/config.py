#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模块 - 系统全局配置和常量定义
"""

import os
import logging
import multiprocessing as mp

# 设置环境变量以禁用进度条
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"  # 禁用所有 tqdm 进度条

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 分布式角色定义
COORDINATOR = 'coordinator'  # 协调节点
WORKER = 'worker'  # 工作节点

# 向量维度配置
VECTOR_DIM = 384  # 使用的向量嵌入维度

# 默认模型配置
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Replace the existing DB_TABLE_CONFIG in config.py with this:

# 数据库表配置 - 可由管理员手动修改以适应不同需求
DB_TABLE_CONFIG = {
    # 主键字段，自动递增
    'id_field': 'id',
    'id_type': 'SERIAL PRIMARY KEY',
    
    # 报告编号字段，用于标识数据来源
    'report_field': 'report_number',
    'report_type': 'TEXT NOT NULL',
    
    # 块ID字段，用于唯一标识文本块
    'chunk_id_field': 'chunk_id',
    'chunk_id_type': 'TEXT NOT NULL',
    'chunk_id_unique': True,  # 是否设置为唯一键
    
    # 文本块字段，存储原始文本
    'text_field': 'text_chunk',
    'text_type': 'TEXT NOT NULL',
    
    # 向量字段，存储嵌入向量
    'vector_field': 'embedding',
    'vector_type': f'public.vector({VECTOR_DIM})',
    
    # 时间戳字段，记录创建时间
    'timestamp_field': 'created_at',
    'timestamp_type': 'TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP',
}

# 块切分配置
CHUNK_CONFIG = {
    'chunk_size': 128,  # 目标令牌数
    'overlap_tokens': 20,  # 块之间的重叠令牌数
}

# 全局状态标志
LAST_PROGRESS_TIME = None
SYSTEM_HEALTHY = True

# 自动检测可用CPU核心数
CPU_CORES = mp.cpu_count()

# 初始化设备变量 - 将在system_utils.py中被设置为实际值
device = "cpu"  # 默认值