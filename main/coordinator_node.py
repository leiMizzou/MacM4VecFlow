#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
协调器节点模块 - 负责在分布式系统中协调工作节点和处理结果
"""

import os
import time
import queue
import threading
import traceback
from threading import Thread, Event
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any

from config import logger, SYSTEM_HEALTHY
from system_utils import device, clean_memory, optimize_system_for_m4, SystemMonitor, optimize_pipeline_concurrency
from db_processor import DatabaseProcessor
from embedding_utils import TextProcessor
from comm_utils import DistributedComm

# 文本切分处理函数 - 协调器版
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

# 嵌入处理函数 - 协调器版
def coordinator_embedding_processor(chunk_queue, result_queue, chunk_done, embedding_done, db_config, batch_size=512):
    """
    协调器的嵌入处理函数
    
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
        status_interval = 1000  # 更频繁的状态报告
        
        # 创建批次容器
        batch_texts = []
        batch_data = []
        
        # 允许更大的批处理累积
        max_batch_wait = 0.1  # 秒
        last_process_time = time.time()
        
        # 内存使用监控
        last_memory_check = time.time()
        memory_check_interval = 20  # 缩短内存检查间隔(秒)
        
        # 动态调整批处理大小
        current_batch_size = min(batch_size, 256)  # 降低初始批处理大小
        processing_times = []  # 跟踪处理时间
        
        # 错误计数
        error_count = 0
        max_consecutive_errors = 5
        
        # 健康监控
        monitor = SystemMonitor(timeout=180)  # 3分钟无进度报警
        monitor.start()
        
        # 性能监控变量
        queue_full_count = 0
        last_queue_check = time.time()
        
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
                    queue_full_count += 1
                    
                    # 如果队列持续满，强制清理内存
                    if queue_full_count > 10:
                        logger.warning("结果队列持续满载，执行强制内存清理")
                        clean_memory(force_gc=True)
                        queue_full_count = 0
                
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
                
                # 定期检查和报告队列状态
                current_time = time.time()
                if current_time - last_queue_check >= 60:  # 每分钟检查一次
                    chunk_queue_size = chunk_queue.qsize()
                    result_queue_size = result_queue.qsize()
                    logger.info(f"队列状态: 块队列 {chunk_queue_size}, 结果队列 {result_queue_size}")
                    
                    # 检测队列瓶颈
                    if chunk_queue_size > chunk_queue.maxsize * 0.9:
                        logger.warning(f"检测到块队列瓶颈! ({chunk_queue_size}/{chunk_queue.maxsize})")
                        
                        # 如果块队列几乎满且当前批处理大小较小，增加批处理大小以加快消费
                        if current_batch_size < 512:
                            old_batch_size = current_batch_size
                            current_batch_size = min(current_batch_size * 1.5, 512)
                            current_batch_size = int(current_batch_size)
                            logger.info(f"增大批处理大小: {old_batch_size} -> {current_batch_size}")
                    
                    last_queue_check = current_time
                
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
                                    clean_memory(force_gc=True)  # 先进行内存清理
                                    embed_processor = TextProcessor()
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
                            
                            # 根据平均处理时间调整批处理大小 - 更积极地调整
                            if avg_time < 0.5 and current_batch_size < 512 and chunk_queue.qsize() > chunk_queue.maxsize * 0.5:
                                # 处理速度快且队列有积压，增加批处理大小
                                current_batch_size = min(current_batch_size * 1.2, 512)
                                current_batch_size = int(current_batch_size)
                            elif avg_time > 1.5 and current_batch_size > 64:
                                # 处理速度慢，更积极地减小批处理大小
                                current_batch_size = max(current_batch_size * 0.7, 64)
                                current_batch_size = int(current_batch_size)
                    
                    # 周期性地清理内存
                    if processed_chunks % 3000 == 0 and processed_chunks > 0:  # 更频繁的内存检查
                        # 获取当前内存使用状态
                        if time.time() - last_memory_check >= memory_check_interval:
                            try:
                                import psutil
                                process = psutil.Process(os.getpid())
                                memory_mb = process.memory_info().rss / (1024 * 1024)
                                
                                logger.info(f"  → 进程内存使用: {memory_mb:.1f}MB, 批处理大小: {current_batch_size}")
                                last_memory_check = time.time()
                                
                                # 如果内存占用过高，清理
                                if memory_mb > 3000:  # 降低到3GB
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
                    
                    # 获取并输出内存使用状态
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        logger.info(f"  → 内存使用: {memory_mb:.1f}MB, 系统内存: {psutil.virtual_memory().percent}%")
                    except:
                        pass
                    
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


def run_coordinator(args):
    """
    协调器节点主函数
    
    参数:
        args: 命令行参数对象
    """
    logger.info("=" * 80)
    logger.info(f"启动协调器节点，监听地址: {args.host}:{args.coordinator_port}")
    logger.info("=" * 80)
    
    # 应用系统优化
    optimize_system_for_m4()
    
    # 创建通信和数据库处理对象
    comm = None
    db_processor = None
    system_monitor = None  # 初始化为None
    
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
        # 创建健康监控
        system_monitor = SystemMonitor(timeout=300)  # 5分钟无进度报警
        system_monitor.start()
        
        # 初始化数据库处理器
        db_processor = DatabaseProcessor(db_config)

        # 连接数据库
        if not db_processor.connect():
            logger.error("无法连接到数据库，程序退出")
            return
        
        # 创建向量表
        table_name = args.target_table
        if not db_processor.setup_vector_table(table_name, schema=args.db_schema):
            logger.error("无法设置向量数据库表，程序退出")
            return
        
        # 初始化通信
        comm = DistributedComm(
            role='coordinator',
            coordinator_ip=args.host,
            coordinator_port=args.coordinator_port,
            worker_port=args.worker_port
        )

        logger.info("=" * 80)
        logger.info("开始分布式数据处理...")
        logger.info("=" * 80)
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
        logger.info("等待工作节点连接...(10秒超时)")
        worker_info = None
        start_wait = time.time()
        print("等待工作节点连接: ", end="", flush=True)
        while time.time() - start_wait < 10:
            worker_info = comm.receive_message('workers', timeout=1000)
            if worker_info:
                print()  # End the line of dots
                logger.info(f"工作节点已连接: {worker_info}")
                break
            print(".", end="", flush=True)
            time.sleep(1)
        print()  # Make sure to end the line of dots

        if not worker_info:
            logger.info("没有工作节点连接，将以单机模式运行")
            total_partitions = 1
            # 继续执行单机模式...
        else:
            logger.info(f"工作节点已连接")
            total_partitions = 2  # 固定使用两个分区(协调器和一个工作节点)
        
        # 获取总数据量
        total_count = 0
        try:
            # 计算源表中符合条件的记录数
            with db_processor.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM (
                            SELECT 1 
                            FROM {args.db_schema}.{args.source_table} 
                            WHERE {args.text_field} IS NOT NULL 
                            LIMIT %s
                        ) t
                    """, (args.limit,))
                    total_count = cur.fetchone()[0]
        except Exception as e:
            logger.error(f"获取总数据量时出错: {e}")
            logger.error(traceback.format_exc())
            total_count = args.limit  # 使用限制作为估计
                
        logger.info(f"总共需要处理约 {total_count} 条记录\n")
        
        # 健康监控
        system_monitor = SystemMonitor(timeout=300)  # 5分钟无进度报警
        system_monitor.start()
        
        # Producer: 在单独的线程池中切分文本
        def chunk_producer():
            """文本切分线程：从数据库获取文本并切分成块"""
            total_records = 0
            total_chunks = 0
            
            # 协调器处理第一个分区
            partition_id = 1
            
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
                        
                        # 检查块队列是否接近满载 - 实现背压
                        if chunk_queue.qsize() > chunk_queue_size * 0.8:
                            logger.warning(f"块队列接近满载 ({chunk_queue.qsize()}/{chunk_queue_size})，暂停生产...")
                            # 等待队列减少到更低水位线
                            while chunk_queue.qsize() > chunk_queue_size * 0.5:
                                time.sleep(1)
                                # 如果嵌入处理完成，应该跳出等待
                                if embedding_done.is_set():
                                    logger.error("嵌入处理已完成但块队列仍然满！跳过等待。")
                                    break
                            logger.info(f"块队列水位降低 ({chunk_queue.qsize()}/{chunk_queue_size})，恢复生产")
                        
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
                                
                                # 将块添加到队列前，再次检查队列状态
                                if chunk_queue.qsize() > chunk_queue_size * 0.9:
                                    logger.warning(f"块队列高水位 ({chunk_queue.qsize()}/{chunk_queue_size})，暂停添加...")
                                    # 等待队列空间
                                    while chunk_queue.qsize() > chunk_queue_size * 0.7:
                                        time.sleep(0.5)
                                        if embedding_done.is_set():
                                            break
                                
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
                        
                        # 显示总体进度
                        elapsed = time.time() - start_time
                        if total_count > 0:
                            partition_progress = total_records / (total_count / total_partitions)
                            logger.info(f"\n分区进度: 已处理 {total_records} 条记录 ({partition_progress*100:.1f}%)")
                            if total_records > 0:
                                est_total = elapsed / partition_progress
                                remaining = est_total - elapsed
                                logger.info(f"  → 已用时间: {elapsed/60:.1f}分钟，预计剩余: {remaining/60:.1f}分钟")
                        logger.info(f"  → 已生成 {total_chunks} 个文本块，队列大小: {chunk_queue.qsize()}")
                        
                        # 获取内存使用信息
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            logger.info(f"  → 进程内存使用: {memory_mb:.1f}MB")
                        except:
                            pass
                        
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
        
        # Consumer: 将结果高效存储在数据库中
        def db_consumer():
            """数据库存储线程：将生成的嵌入向量存储到数据库"""
            # 用于统计的计数器
            stored_count = 0
            batch = []
            
            try:
                while not (embedding_done.is_set() and result_queue.empty()):
                    try:
                        # 收集批次进行数据库插入
                        while len(batch) < 1000:  # 减小批次大小，提高事务提交频率
                            try:
                                item = result_queue.get(timeout=0.2)
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
                            if db_processor.store_vectors(table_name, batch, schema=args.db_schema):
                                stored_count += len(batch)
                                store_time = time.time() - store_start
                                logger.info(f"  → 成功存储 {len(batch)} 个向量，总计: {stored_count}，耗时 {store_time:.2f}秒 "
                                        f"({len(batch)/store_time:.1f} 向量/秒)")
                            batch = []
                            
                            # 更新进度
                            system_monitor.update_progress()
                            
                            # 定期清理内存
                            if stored_count % 5000 == 0:
                                clean_memory(force_gc=False)
                    except Exception as e:
                        logger.error(f"存储向量时出错: {e}")
                        logger.error(traceback.format_exc())
                        time.sleep(0.1)
                        continue
                
                # 存储剩余批次
                if batch:
                    if db_processor.store_vectors(table_name, batch, schema=args.db_schema):
                        stored_count += len(batch)
                        logger.info(f"  → 成功存储最后 {len(batch)} 个向量，总计: {stored_count}")
                
                logger.info(f"所有向量存储任务完成，共存储 {stored_count} 个向量")
            except Exception as e:
                logger.error(f"数据库消费者线程出错: {e}")
                logger.error(traceback.format_exc())
        
        # 结果接收线程 - 专门接收工作节点结果
        def result_receiver():
            """结果接收线程：接收工作节点发送的嵌入向量结果"""
            received_count = 0
            worker_complete = False
            
            try:
                while not (worker_complete and embedding_done.is_set() and result_queue.empty()):
                    try:
                        # 接收来自工作节点的消息
                        message = comm.receive_message('tasks', timeout=1000)
                        if message:
                            # 检查是否是完成消息
                            if isinstance(message, dict) and message.get('status') == 'complete':
                                logger.info(f"工作节点报告完成，处理了 {message.get('count', 0)} 个向量")
                                worker_complete = True
                                continue
                                
                            # 处理批量向量数据
                            if isinstance(message, list):
                                # 检查结果队列空间
                                if result_queue.qsize() > result_queue.maxsize * 0.9:
                                    logger.warning(f"结果队列接近满载 ({result_queue.qsize()}/{result_queue.maxsize})，暂缓接收...")
                                    # 等待队列有空间
                                    while result_queue.qsize() > result_queue.maxsize * 0.7:
                                        time.sleep(0.5)
                                
                                # 将结果添加到结果队列
                                for item in message:
                                    try:
                                        result_queue.put(item, timeout=0.5)
                                        received_count += 1
                                    except queue.Full:
                                        logger.warning("结果队列已满，等待空间...")
                                        time.sleep(0.5)
                                        # 再次尝试
                                        result_queue.put(item)
                                        received_count += 1
                                
                                if received_count % 1000 == 0:
                                    logger.info(f"  → 已从工作节点接收 {received_count} 个向量结果")
                                    
                                # 更新进度
                                system_monitor.update_progress()
                        else:
                            # 短暂暂停避免CPU高占用
                            time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"接收工作节点结果时出错: {e}")
                        logger.error(traceback.format_exc())
                        time.sleep(1)  # 出错后稍长等待
                
                logger.info(f"结果接收线程完成，共接收 {received_count} 个向量")
                
                # 更新进度
                system_monitor.update_progress()
            except Exception as e:
                logger.error(f"结果接收线程出错: {e}")
                logger.error(traceback.format_exc())
        
        # 启动流水线线程
        try:
            logger.info("启动分布式流水线处理...")
            
            # 创建所有线程
            thread_pool = []
            
            # 创建并启动文本切分线程
            chunk_thread = Thread(target=chunk_producer, name="ChunkProducer")
            chunk_thread.daemon = True
            thread_pool.append(chunk_thread)
            
            # 创建结果存储线程
            db_thread = Thread(target=db_consumer, name="DBConsumer")
            db_thread.daemon = True
            thread_pool.append(db_thread)
            
            # 创建嵌入处理线程(本地)
            embedding_thread = Thread(
                target=coordinator_embedding_processor,
                args=(chunk_queue, result_queue, chunk_done, embedding_done, db_config, embedding_batch_size),
                name="EmbeddingProcessor"
            )
            embedding_thread.daemon = True
            thread_pool.append(embedding_thread)
            
            # 创建结果接收线程
            if worker_info:
                result_thread = Thread(target=result_receiver, name="ResultReceiver")
                result_thread.daemon = True
                thread_pool.append(result_thread)
            
            # 按顺序启动线程
            logger.info("启动文本切分线程...")
            chunk_thread.start()
            
            # 小延迟让块开始积累
            time.sleep(2)  # 增加延迟
            
            logger.info("启动嵌入处理线程...")
            embedding_thread.start()
            
            # 启动工作节点结果接收线程
            if worker_info:
                logger.info("启动工作节点结果接收线程...")
                result_thread.start()
            
            # 小延迟让嵌入开始积累
            time.sleep(2)  # 增加延迟
            
            logger.info("启动数据库存储线程...")
            db_thread.start()
            
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
                        
                        # 获取内存使用信息
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            logger.info(f"进程内存使用: {memory_mb:.1f}MB")
                        except:
                            pass
                        
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
            
            # 统计信息输出
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info("\n处理完成:")
            logger.info("=" * 80)
            logger.info(f"总耗时: {processing_time:.2f} 秒 ({processing_time/60:.2f} 分钟)")
            logger.info(f"使用的设备: {device.upper()}")
            logger.info(f"文本切分进程数: {num_chunk_processes}")
            logger.info(f"嵌入计算进程数: {num_embedding_processes}")
            logger.info(f"批处理大小: {embedding_batch_size}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"流水线处理出错: {e}")
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
        
        # 清理资源
        if db_processor:
            try:
                db_processor.close()
            except:
                pass
        
        if comm:
            try:
                comm.close()
            except:
                pass