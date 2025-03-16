#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口 - 分布式向量嵌入处理系统
"""

import os
import sys
import time
import argparse
import logging
import traceback

from config import logger, COORDINATOR, WORKER
from coordinator_node import run_coordinator
from worker_node import run_worker


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='分布式向量嵌入处理系统')
    parser.add_argument('--role', choices=[COORDINATOR, WORKER], required=True, help='节点角色 (coordinator 或 worker)')
    
    # 数据库参数
    parser.add_argument('--db-host', default='169.254.22.165', help='数据库主机地址')
    parser.add_argument('--db-port', type=int, default=5432, help='数据库端口')
    parser.add_argument('--db-name', default='fda_device', help='数据库名称')
    parser.add_argument('--db-user', default='postgres', help='数据库用户名')
    parser.add_argument('--db-password', default='12345687', help='数据库密码')
    parser.add_argument('--db-schema', default='device', help='数据库模式')
    
    # 表参数
    parser.add_argument('--source-table', default='event_texts', help='源数据表名')
    parser.add_argument('--target-table', default='event_text_vectors', help='向量表名称')
    parser.add_argument('--id-field', default='id', help='ID字段名')
    parser.add_argument('--text-field', default='text', help='文本字段名')
    
    # 分布式通信参数
    parser.add_argument('--host', default='0.0.0.0', help='本机IP，协调器节点使用')
    parser.add_argument('--coordinator-host', default='localhost', help='协调器IP，工作节点使用')
    parser.add_argument('--coordinator-port', type=int, default=5555, help='协调器端口')
    parser.add_argument('--worker-port', type=int, default=5556, help='工作节点通信端口')
    
    # 任务参数
    parser.add_argument('--limit', type=int, default=500000, help='处理的最大记录数')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
    
    # 高级参数
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='使用的嵌入模型')
    parser.add_argument('--log-file', default='vector_processor.log', help='日志文件路径')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    # 配置文件日志
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 根据角色运行相应的函数
    start_time = time.time()
    
    try:
        if args.role == COORDINATOR:
            logger.info(f"启动协调器模式，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_coordinator(args)
        else:
            logger.info(f"启动工作节点模式，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_worker(args)
    except KeyboardInterrupt:
        logger.warning("收到中断信号，程序正在安全关闭...")
    except Exception as e:
        logger.critical(f"程序执行出现严重错误: {e}")
        logger.critical(traceback.format_exc())
    finally:
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(f"程序运行结束，总耗时: {run_time/60:.2f} 分钟 ({run_time:.2f} 秒)")
        logger.info(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"程序崩溃: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)