#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工作节点模块 - 负责在分布式系统中处理文本切分和嵌入生成任务
"""

import os
import time
import socket  # 确保socket模块在最顶层导入
import queue
import threading
import traceback
import psutil
from threading import Thread, Event
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Any

from config import logger, SYSTEM_HEALTHY
from system_utils import device, clean_memory, optimize_system_for_m4, SystemMonitor, optimize_pipeline_concurrency
from db_processor import DatabaseProcessor
from embedding_utils import TextProcessor
from comm_utils import DistributedComm

# 工作节点文本切分处理函数
def chunk_process_worker(args: Tuple[List[Tuple[str, str]], dict]) -> List[Tuple[str, int, str]]:
    """
    处理文本切分的工作函数
    
    参数:
        args: 包含(批次数据, 数据库配置)的元组
        
    返回:
        处理后的(报告号, 块索引, 文本内容)元组列表
    """
    batch, db_config = args
    processor = None
    results = []
    
    try:
        processor = TextProcessor()
        
        for report_number, text in batch:
            if not text:
                continue
                
            try:
                chunks = processor.process_text(text)
                for i, chunk in enumerate(chunks):
                    if not chunk or chunk.strip() == "":
                        continue
                    # 只返回必要的数据：报告号、块索引、文本内容
                    results.append((str(report_number), i+1, chunk))
                    
                    # 频繁释放内存
                    if len(results) % 5000 == 0:
                        # 清理内存
                        clean_memory(force_gc=False)
            except Exception as e:
                logger.error(f"处理报告 {report_number} 时出错: {e}")
                logger.error(traceback.format_exc())
                continue
                
    except Exception as e:
        logger.error(f"文本切分处理时出错: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 清理资源
        clean_memory(force_gc=True)
    
    return results

# 嵌入处理函数
def worker_embedding_processor(chunk_queue, result_queue, chunk_done, embedding_done, db_config, batch_size=512):
    """
    工作节点的嵌入处理函数
    
    参数:
        chunk_queue: 文本块队列
        result_queue: 结果队列
        chunk_done: 文本切分完成事件
        embedding_done: 嵌入处理完成事件
        db_config: 数据库配置
        batch_size: 嵌入批处理大小
    """
    # 创建处理器
    embed_processor = None
    monitor = None
    
    try:
        embed_processor = TextProcessor()
        
        # 跟踪统计信息
        processed_chunks = 0
        cache_hits = 0
        start_time = time.time()
        
        # 状态报告频率
        status_interval = 2000
        
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
        
        # 错误计数
        error_count = 0
        max_consecutive_errors = 5
        
        # 健康监控
        monitor = SystemMonitor(timeout=180)  # 3分钟无进度报警
        monitor.start()
        
        while not (chunk_done.is_set() and chunk_queue.empty()):
            try:
                # 检查系统是否健康
                global SYSTEM_HEALTHY
                if not SYSTEM_HEALTHY:
                    logger.warning("系统处于不健康状态，可能需要人工干预")
                    time.sleep(5)  # 等待可能的恢复
                    SYSTEM_HEALTHY = True  # 尝试恢复并继续
                
                # 检查结果队列是否快满
                while result_queue.qsize() > result_queue.maxsize * 0.9:
                    logger.warning(f"结果队列接近满载 ({result_queue.qsize()}/{result_queue.maxsize})，等待空间...")
                    time.sleep(0.5)
                
                # 轮询，超时以检查完成标志
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                    report_number, chunk_idx, text = chunk
                    
                    # 更新进度
                    monitor.update_progress()
                    
                    # 1. 缓存查找
                    cached_embedding = embed_processor.embedding_cache.get(text)
                    if cached_embedding is not None:
                        # 缓存命中，直接使用
                        chunk_id = f"{report_number}_{chunk_idx}"
                        try:
                            result_queue.put((report_number, chunk_id, text, cached_embedding), timeout=1)
                            cache_hits += 1
                            processed_chunks += 1
                        except queue.Full:
                            logger.warning("结果队列已满，等待空间...")
                            # 放回chunk队列，避免丢失
                            try:
                                chunk_queue.put((report_number, chunk_idx, text), timeout=1)
                            except queue.Full:
                                logger.error("无法放回chunk队列，数据丢失!")
                            time.sleep(1)  # 等待队列空间
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
                        try:
                            # 使用优化的批处理
                            embeddings = embed_processor.get_embeddings_batch(batch_texts)
                            
                            # 重置错误计数 - 成功处理
                            error_count = 0
                            
                            # 将结果发送到队列并更新缓存
                            for i, (r_num, c_idx, txt) in enumerate(batch_data):
                                if i < len(embeddings):
                                    chunk_id = f"{r_num}_{c_idx}"
                                    
                                    # 尝试放入结果队列，避免满队列阻塞
                                    try:
                                        result_queue.put((r_num, chunk_id, txt, embeddings[i]), timeout=0.5)
                                    except queue.Full:
                                        logger.warning("结果队列已满，暂时跳过，将重试...")
                                        time.sleep(0.1)
                                        try:
                                            result_queue.put((r_num, chunk_id, txt, embeddings[i]), timeout=1)
                                        except queue.Full:
                                            logger.error(f"无法添加结果到队列: {r_num}, {c_idx}")
                            
                            # 更新统计信息
                            processed_chunks += len(batch_texts)
                            
                        except Exception as e:
                            logger.error(f"处理嵌入批次时出错: {e}")
                            logger.error(traceback.format_exc())
                            
                            # 增加错误计数
                            error_count += 1
                            
                            # 检查连续错误是否超过阈值
                            if error_count >= max_consecutive_errors:
                                logger.critical(f"连续错误超过阈值 ({max_consecutive_errors})，尝试恢复...")
                                # 尝试重新创建嵌入处理器
                                try:
                                    embed_processor = TextProcessor()
                                    # 清理内存
                                    clean_memory(force_gc=True)
                                    logger.info("嵌入处理器已重新创建")
                                    error_count = 0
                                except Exception as re:
                                    logger.critical(f"无法恢复嵌入处理器: {re}")
                        
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
                            if avg_time < 0.5 and current_batch_size < 1024:
                                # 处理速度快，增加批处理大小
                                current_batch_size = min(current_batch_size * 1.2, 1024)
                                current_batch_size = int(current_batch_size)
                            elif avg_time > 2.0 and current_batch_size > 128:
                                # 处理速度慢，减小批处理大小
                                current_batch_size = max(current_batch_size * 0.8, 128)
                                current_batch_size = int(current_batch_size)
                    
                    # 周期性地清理内存
                    if processed_chunks % 5000 == 0 and processed_chunks > 0:
                        # 获取当前内存使用状态
                        if time.time() - last_memory_check >= memory_check_interval:
                            try:
                                process = psutil.Process(os.getpid())
                                memory_mb = process.memory_info().rss / (1024 * 1024)
                                
                                logger.info(f"  → 进程内存使用: {memory_mb:.1f}MB, 批处理大小: {current_batch_size}")
                                last_memory_check = time.time()
                                
                                # 如果内存占用过高，清理
                                if memory_mb > 3500:  # 3.5GB
                                    clean_memory(force_gc=True)
                                    logger.info("执行强制内存清理")
                            except:
                                # 忽略内存检查错误
                                pass
                        else:
                            clean_memory(force_gc=False)
                
                # 报告进度
                if processed_chunks > 0 and processed_chunks % status_interval == 0:
                    elapsed = time.time() - start_time
                    chunks_per_sec = processed_chunks / elapsed
                    
                    # 缓存统计
                    cache_stats = embed_processor.embedding_cache.stats()
                    
                    logger.info(f"  → 已处理 {processed_chunks} 个嵌入向量 "
                          f"({chunks_per_sec:.1f} 个/秒), "
                          f"队列中还有 {chunk_queue.qsize()} 个文本块")
                    logger.info(f"  → 缓存命中率: {cache_stats['hit_rate']*100:.1f}%, "
                          f"命中/未命中: {cache_stats['hits']}/{cache_stats['misses']}, "
                          f"批处理大小: {current_batch_size}")
                    
                    # 更新进度
                    monitor.update_progress()
                
            except Exception as e:
                logger.error(f"嵌入处理主循环错误: {e}")
                logger.error(traceback.format_exc())
                # 错误后短暂暂停，避免快速失败循环
                time.sleep(0.5)
                continue
        
        # 处理剩余的批次
        if batch_texts:
            try:
                embeddings = embed_processor.get_embeddings_batch(batch_texts)
                for i, (r_num, c_idx, txt) in enumerate(batch_data):
                    if i < len(embeddings):
                        chunk_id = f"{r_num}_{c_idx}"
                        try:
                            result_queue.put((r_num, chunk_id, txt, embeddings[i]), timeout=1)
                        except queue.Full:
                            logger.warning("最终处理：结果队列已满，等待空间...")
                            time.sleep(1)
                            result_queue.put((r_num, chunk_id, txt, embeddings[i]))
            except Exception as e:
                logger.error(f"处理最后批次时出错: {e}")
        
        # 记录最终统计信息
        elapsed = time.time() - start_time
        if processed_chunks > 0:
            chunks_per_sec = processed_chunks / elapsed
            cache_stats = embed_processor.embedding_cache.stats()
            
            logger.info(f"嵌入处理完成:")
            logger.info(f"  → 总处理量: {processed_chunks} 个嵌入向量")
            logger.info(f"  → 处理速度: {chunks_per_sec:.1f} 个/秒")
            logger.info(f"  → 总耗时: {elapsed:.1f} 秒")
            logger.info(f"  → 缓存命中率: {cache_stats['hit_rate']*100:.1f}%")
            logger.info(f"  → 使用的设备: {device}")
        
    except Exception as e:
        logger.error(f"嵌入处理器关键错误: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 标记完成
        embedding_done.set()
        
        # 停止监控
        if monitor:
            monitor.stop()
        
        # 清理内存
        clean_memory(force_gc=True)


def run_worker(args):
    """
    工作节点主函数
    
    参数:
        args: 命令行参数对象
    """
    logger.info("=" * 80)
    logger.info(f"启动工作节点，连接到协调器: {args.coordinator_host}:{args.coordinator_port}")
    logger.info("=" * 80)
    
    # 应用系统优化
    optimize_system_for_m4()
    
    # 预先初始化关键变量，以防early exit时仍能正确清理资源
    comm = None
    db_processor = None
    system_monitor = None
    thread_pool = []  # 确保thread_pool变量在finally块之前被定义
    
    try:
        # 数据库配置
        db_config = {
            'host': args.db_host,
            'port': args.db_port,
            'dbname': args.db_name,
            'user': args.db_user,
            'password': args.db_password,
            'options': f'-c search_path={args.db_schema}'
        }
        
        # 初始化数据库处理器
        db_processor = DatabaseProcessor(db_config)
        
        # 连接数据库
        if not db_processor.connect():
            logger.error("无法连接到数据库，程序退出")
            return
        
        # 初始化通信
        comm = DistributedComm(
            role='worker',
            coordinator_ip=args.coordinator_host,
            coordinator_port=args.coordinator_port,
            worker_port=args.worker_port
        )
        
        # 向协调器注册
        worker_info = {
            'hostname': socket.gethostname(),
            'identity': comm.identity,  # 使用ZeroMQ的identity作为工作节点ID
            'pid': os.getpid(),
            'device': device,
            'cores': os.cpu_count(),
            'memory': psutil.virtual_memory().total // (1024*1024*1024),  # GB
            'timestamp': time.time()
        }
        
        if not comm.send_message('coordinator', worker_info):
            logger.error("无法连接到协调器，程序退出")
            return
        
        logger.info("向协调器注册成功，等待分区分配...")
        
        # 等待接收分区分配
        partition_assigned = False
        for _ in range(10):  # 最多尝试10次
            assignment = comm.receive_message('coordinator', timeout=3000)
            if assignment and isinstance(assignment, dict) and assignment.get('type') == 'partition_assignment':
                partition_id = assignment.get('partition_id')
                total_partitions = assignment.get('total_partitions')
                logger.info(f"已接收分区分配: 分区ID={partition_id}, 总分区数={total_partitions}")
                partition_assigned = True
                break
            time.sleep(1)
        
        if not partition_assigned:
            logger.error("未接收到分区分配，使用默认值 (分区ID=2, 总分区数=2)")
            partition_id = 2
            total_partitions = 2
        
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
        
        # 创建健康监控
        system_monitor = SystemMonitor(timeout=300)  # 5分钟无进度报警
        system_monitor.start()
        
        # Producer: 在单独的线程池中切分文本
        def chunk_producer():
            """文本切分线程：从数据库获取文本并切分成块"""
            total_records = 0
            total_chunks = 0
            
            try:
                with ProcessPoolExecutor(max_workers=num_chunk_processes) as chunk_pool:
                    for batch_idx, batch in enumerate(db_processor.fetch_partitioned_data(
                        args.source_table, args.id_field, args.text_field,
                        partition_id, total_partitions, limit=args.limit, batch_size=10000,
                        schema=args.db_schema)):
                        if not batch:
                            continue
                        
                        batch_start = time.time()
                        batch_size = len(batch)
                        total_records += batch_size
                        
                        logger.info(f"批次 {batch_idx+1}: 处理 {batch_size} 条记录")
                        
                        # 更新进度
                        system_monitor.update_progress()
                        
                        # 并行处理块
                        futures = []
                        sub_batch_size = max(1, batch_size // num_chunk_processes)
                        sub_batches = [batch[i:i + sub_batch_size] for i in range(0, batch_size, sub_batch_size)]
                        
                        for sub_batch in sub_batches:
                            future = chunk_pool.submit(chunk_process_worker, (sub_batch, db_config))
                            futures.append(future)
                        
                        # 处理完成的结果（不等待全部完成）
                        for future in futures:
                            try:
                                chunks = future.result()
                                total_chunks += len(chunks)
                                
                                # 将块添加到队列
                                for chunk in chunks:
                                    # 使用超时添加，避免永久阻塞
                                    while True:
                                        try:
                                            chunk_queue.put(chunk, timeout=1)
                                            break
                                        except queue.Full:
                                            logger.warning(f"块队列已满 ({chunk_queue.qsize()}/{chunk_queue_size})，等待空间...")
                                            time.sleep(0.5)
                                            # 检查嵌入处理是否卡住
                                            if embedding_done.is_set():
                                                logger.error("嵌入处理已完成但块队列仍然满！跳过剩余块。")
                                                break
                            except Exception as e:
                                logger.error(f"获取切分结果时出错: {e}")
                                logger.error(traceback.format_exc())
                        
                        # 跟踪进度
                        chunk_time = time.time() - batch_start
                        logger.info(f"  → 批次 {batch_idx+1} 文本切分完成，耗时 {chunk_time:.2f}秒")
                        logger.info(f"  → 已生成 {total_chunks} 个文本块，队列大小: {chunk_queue.qsize()}")
                        
                        # 更新进度
                        system_monitor.update_progress()
                    
                    # 标记完成
                    chunk_done.set()
                    logger.info("所有文本切分任务完成")
            except Exception as e:
                logger.error(f"文本切分生产者出错: {e}")
                logger.error(traceback.format_exc())
                # 确保设置完成标志，即使出错
                chunk_done.set()
        
        # 工作节点无需数据库写入，结果直接发送到协调器
        def result_sender():
            """结果发送线程：将生成的嵌入向量发送给协调器"""
            sent_count = 0
            batch = []
            batch_size = 50  # 减小发送批次大小，提高响应性
            
            try:
                while not (embedding_done.is_set() and result_queue.empty()):
                    try:
                        # 检查系统是否健康
                        global SYSTEM_HEALTHY
                        if not SYSTEM_HEALTHY:
                            logger.warning("系统处于不健康状态，可能需要人工干预")
                            time.sleep(5)  # 等待可能的恢复
                            SYSTEM_HEALTHY = True  # 尝试恢复并继续
                        
                        # 收集批次
                        while len(batch) < batch_size:
                            try:
                                item = result_queue.get(timeout=0.2)
                                batch.append(item)
                                
                                # 如果在清空最后的项目，中断
                                if embedding_done.is_set() and result_queue.empty():
                                    break
                            except queue.Empty:
                                # 队列暂时为空但还未结束
                                break
                        
                        # 发送批次到协调器
                        if batch:
                            # 最多尝试3次发送
                            for attempt in range(3):
                                # 使用ZeroMQ发送结果
                                if comm.send_message('tasks', batch, timeout=5000):  # 5秒超时
                                    sent_count += len(batch)
                                    if sent_count % 1000 == 0:
                                        logger.info(f"  → 已发送 {sent_count} 个向量结果到协调器")
                                    
                                    # 更新进度
                                    system_monitor.update_progress()
                                    break
                                else:
                                    # 发送失败，等待后重试
                                    logger.warning(f"发送批次失败，尝试 {attempt+1}/3")
                                    time.sleep(1)
                            
                            # 清空批次
                            batch = []
                        else:
                            # 避免空轮询消耗CPU
                            time.sleep(0.05)
                            
                    except Exception as e:
                        logger.error(f"发送结果时出错: {e}")
                        logger.error(traceback.format_exc())
                        time.sleep(1)  # 出错后稍长等待
                        continue
                
                # 发送剩余批次
                if batch:
                    # 最后一批次尝试多次，确保发送成功
                    for attempt in range(5):
                        if comm.send_message('tasks', batch, timeout=10000):  # 10秒超时
                            sent_count += len(batch)
                            break
                        else:
                            logger.warning(f"发送最后批次失败，尝试 {attempt+1}/5")
                            time.sleep(2)
                
                logger.info(f"所有结果发送完成，共发送 {sent_count} 个向量结果")
                
                # 通知协调器本节点处理完成 - 尝试多次发送确保成功
                for attempt in range(5):
                    completion_message = {
                        'status': 'complete', 
                        'count': sent_count,
                        'worker_id': comm.identity  # 包含工作节点标识
                    }
                    if comm.send_message('tasks', completion_message):
                        logger.info("成功发送完成通知到协调器")
                        break
                    else:
                        logger.warning(f"发送完成通知失败，尝试 {attempt+1}/5")
                        time.sleep(2)
            except Exception as e:
                logger.error(f"结果发送线程出错: {e}")
                logger.error(traceback.format_exc())
        
        # 启动流水线线程
        try:
            logger.info("启动工作节点处理流水线...")
            
            # 创建所有线程
            thread_pool = []
            
            # 创建并启动文本切分线程
            chunk_thread = Thread(target=chunk_producer, name="ChunkProducer")
            chunk_thread.daemon = True
            thread_pool.append(chunk_thread)
            
            # 创建结果发送线程
            result_thread = Thread(target=result_sender, name="ResultSender") 
            result_thread.daemon = True
            thread_pool.append(result_thread)
            
            # 创建嵌入处理线程
            embedding_threads = []
            for i in range(num_embedding_processes):
                thread = Thread(
                    target=worker_embedding_processor,
                    args=(chunk_queue, result_queue, chunk_done, embedding_done, db_config, embedding_batch_size),
                    name=f"EmbeddingProcessor-{i+1}"
                )
                thread.daemon = True
                embedding_threads.append(thread)
                thread_pool.append(thread)
            
            # 按顺序启动线程
            logger.info("启动文本切分线程...")
            chunk_thread.start()
            
            # 小延迟让块开始积累
            time.sleep(1)
            
            logger.info("启动嵌入处理线程...")
            for i, thread in enumerate(embedding_threads):
                thread.start()
                logger.info(f"嵌入处理线程 {i+1}/{len(embedding_threads)} 已启动")
                # 为每个嵌入线程添加小延迟，避免同时启动造成的资源竞争
                time.sleep(0.5)
            
            # 小延迟让嵌入开始积累
            time.sleep(1)
            
            logger.info("启动结果发送线程...")
            result_thread.start()
            
            # 定期检查进度
            running = True
            check_interval = 60  # 每分钟检查一次
            
            while running:
                try:
                    # 等待一个周期
                    time.sleep(check_interval)
                    
                    # 检查所有线程是否活跃
                    active_threads = [t for t in thread_pool if t.is_alive()]
                    
                    # 检查完成情况
                    if chunk_done.is_set() and embedding_done.is_set() and result_queue.empty():
                        # 主要工作已完成
                        if all(not t.is_alive() for t in thread_pool):
                            logger.info("所有处理线程已完成")
                            running = False
                        elif len(active_threads) < len(thread_pool):
                            logger.warning(f"工作已完成，但仍有 {len(active_threads)}/{len(thread_pool)} 个线程活跃")
                    else:
                        # 输出进度信息
                        logger.info(f"进度检查: {len(active_threads)}/{len(thread_pool)} 个线程活跃")
                        logger.info(f"块队列大小: {chunk_queue.qsize()}, 结果队列大小: {result_queue.qsize()}")
                        logger.info(f"切分完成标志: {chunk_done.is_set()}, 嵌入完成标志: {embedding_done.is_set()}")
                        
                        # 更新进度
                        system_monitor.update_progress()
                except KeyboardInterrupt:
                    logger.warning("收到键盘中断，正在安全关闭...")
                    break
                except Exception as e:
                    logger.error(f"监控线程出错: {e}")
                    logger.error(traceback.format_exc())
            
            # 等待主要线程完成(设置超时避免永久阻塞)
            for thread in thread_pool:
                thread_name = thread.name
                logger.info(f"等待 {thread_name} 线程完成...")
                thread.join(timeout=60)  # 最多等待60秒
                if thread.is_alive():
                    logger.warning(f"{thread_name} 线程未在预期时间内完成")
            
            logger.info("工作节点所有处理完成")
            
        except Exception as e:
            logger.error(f"工作节点处理出错: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"程序执行过程中出错: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 停止监控
        if system_monitor:
            try:
                system_monitor.stop()
            except:
                pass
        
        # 清理线程 - 确保即使在early exit时也能执行
        for thread in thread_pool:
            try:
                if hasattr(thread, 'is_alive') and thread.is_alive():
                    thread.join(timeout=1)
            except:
                pass
        
        # 清理资源
        if db_processor:
            try:
                db_processor.close()
            except:
                pass
        
        if comm:
            try:
                # 先关闭单个socket，再关闭context
                for socket_name in list(comm.sockets.keys()):
                    try:
                        comm.sockets[socket_name].close(0)
                    except:
                        pass
                # 延迟后关闭context
                time.sleep(0.2)
                try:
                    comm.context.term()
                except:
                    pass
            except:
                pass
        
        # 清理MPS资源
        if device == "mps":
            try:
                import torch
                torch.mps.empty_cache()
                time.sleep(0.5)  # 给MPS一些时间完成操作
            except:
                pass