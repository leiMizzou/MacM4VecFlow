#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统工具模块 - 提供系统级别的优化和监控功能
"""

import os
import time
import platform
import psutil
import gc
import threading
from threading import Timer, RLock
import traceback
import torch
import logging
from config import logger, SYSTEM_HEALTHY, CPU_CORES

# 将设备变量导出到全局命名空间
device = None

# 初始化设备类型
def init_device():
    """初始化设备并设置为全局变量"""
    global device
    
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"✅ 使用 Apple MPS (M4 GPU) 加速")
        # 应用 MPS 增强优化
        enhance_mps_performance()
        # 配置 PyTorch 线程
        torch.set_num_threads(CPU_CORES // 2)
        logger.info(f"✓ PyTorch 线程配置: {CPU_CORES // 2}")
    else:
        device = "cpu"
        logger.info("⚠️ MPS 不可用，使用 CPU 计算")
    
    # 返回设备类型以供调用者使用
    return device

def optimize_system_for_m4():
    """Optimize system settings for M4 chip"""
    # Only apply these optimizations on macOS with Apple Silicon
    if platform.system() != 'Darwin' or 'arm64' not in platform.machine():
        logger.info("⚠️ 非 Apple Silicon Mac，跳过系统优化")
        return
    
    logger.info("🔧 应用 M4 系统优化...")
    
    # 1. Set process priority (nice) if running with proper permissions
    try:
        os.nice(-10)  # Higher priority (lower nice value)
        logger.info("  ✓ 进程优先级已提高")
    except:
        logger.info("  ⚠️ 无法设置进程优先级")
    
    # 2. Optimize NumPy configuration
    try:
        # Use Apple Accelerate framework for NumPy
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_CORES)
        logger.info(f"  ✓ NumPy 配置优化，使用 Accelerate 框架，最大线程: {CPU_CORES}")
    except:
        pass
    
    # 3. Memory allocation tuning
    try:
        os.environ["PYTHONMALLOC"] = "malloc"
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"  # 128KB
        logger.info("  ✓ 内存分配器已优化")
    except:
        pass
    
    # 4. Configure Python GC for high-performance computing
    try:
        gc.set_threshold(100000, 5, 5)  # Less frequent collections
        logger.info("  ✓ 垃圾回收器已优化")
    except:
        pass
        
    logger.info("✅ 系统优化完成")

def enhance_mps_performance():
    """增强M4芯片MPS的性能优化"""
    if not torch.backends.mps.is_available():
        logger.info("⚠️ MPS不可用，跳过MPS优化")
        return False
        
    logger.info("🚀 正在应用增强的MPS优化...")
    
    # 1. 设置环境变量以优化MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 主动内存管理
    os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "1"        # 更优的分配策略
    
    # 2. 启用异步MPS执行
    if hasattr(torch.mps, 'enable_async'):
        torch.mps.enable_async(True)
        logger.info("  ✓ 启用MPS异步执行")
    elif hasattr(torch.mps, 'set_async_execution'):
        torch.mps.set_async_execution(True)
        logger.info("  ✓ 启用MPS异步执行")
    
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
        qk = qk / (dim ** 0.5)
        attn = torch.nn.functional.softmax(qk, dim=-1)
        _ = torch.matmul(attn, v)
        
        # 清理
        del q, k, v, qk, attn
        torch.mps.empty_cache()
        
        logger.info("  ✓ MPS图形管道预热完成")
    except Exception as e:
        logger.warning(f"  ⚠️ MPS预热过程中出错: {e}")
    
    # 5. 配置MPS内存优化
    try:
        # 主动释放内存并尝试保持较低的内存占用
        torch.mps.empty_cache()
        
        # 建议使用的最大内存大小
        # 这需要PyTorch支持，模拟版本
        available_memory = psutil.virtual_memory().available
        suggested_max = min(int(available_memory * 0.7), 8 * 1024 * 1024 * 1024)  # 最多8GB
        
        logger.info(f"  ✓ MPS内存优化配置完成，建议最大使用: {suggested_max/(1024*1024*1024):.1f}GB")
        
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
                        logger.warning(f"  ⚠️ MPS内存使用较高: {allocated:.0f}MB，执行缓存清理")
                        torch.mps.empty_cache()
                except:
                    pass
        
        # 创建监视器
        global mps_memory_monitor
        mps_memory_monitor = MPSMemoryMonitor(threshold_mb=4000)  # 4GB阈值
        logger.info("  ✓ MPS内存监视器已启动")
    except Exception as e:
        logger.warning(f"  ⚠️ 配置MPS内存优化时出错: {e}")
    
    return True

def clean_memory(force_gc=False):
    """Efficiently clean up memory"""
    # 确保全局变量已初始化
    global device
    if device is None:
        init_device()
        
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

class SystemMonitor:
    """监控系统健康状态并在需要时干预"""
    
    def __init__(self, timeout=180):
        self.timeout = timeout  # 无进度超时时间(秒)
        self.watchdog_timer = None
        self.active = False
        self.lock = threading.RLock()
        self.last_progress = time.time()
        
    def start(self):
        """启动监控"""
        with self.lock:
            self.active = True
            self.last_progress = time.time()
            self._restart_timer()
            logger.info(f"系统监控已启动，无活动超时时间: {self.timeout}秒")
            
    def stop(self):
        """停止监控"""
        with self.lock:
            self.active = False
            if self.watchdog_timer:
                self.watchdog_timer.cancel()
            logger.info("系统监控已停止")
            
    def update_progress(self):
        """更新进度时间戳"""
        with self.lock:
            self.last_progress = time.time()
            self._restart_timer()
            
    def _restart_timer(self):
        """重启监控定时器"""
        if self.watchdog_timer:
            self.watchdog_timer.cancel()
            
        if self.active:
            self.watchdog_timer = Timer(self.timeout, self._check_health)
            self.watchdog_timer.daemon = True
            self.watchdog_timer.start()
            
    def _check_health(self):
        """检查系统健康状态"""
        global SYSTEM_HEALTHY
        
        current_time = time.time()
        elapsed = current_time - self.last_progress
        
        if elapsed > self.timeout:
            SYSTEM_HEALTHY = False
            logger.error(f"⚠️ 系统无响应! {elapsed:.1f}秒无进度更新. 请检查处理流程是否卡住。")
            
            # 输出系统状态信息用于诊断
            mem_info = psutil.virtual_memory()
            logger.error(f"系统内存: {mem_info.percent}% 使用中, 可用: {mem_info.available/(1024*1024*1024):.1f}GB")
            
            # 输出Python线程信息
            logger.error("活动线程:")
            for thread in threading.enumerate():
                logger.error(f"  - {thread.name} ({'活动' if thread.is_alive() else '非活动'})")
                
            # 启动新的定时器
            self._restart_timer()
        else:
            # 一切正常，重启定时器
            self._restart_timer()

def optimize_pipeline_concurrency():
    """根据系统特性优化流水线并发度"""
    # 检测可用物理核心数
    physical_cores = CPU_CORES
    
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
                logger.info(f"检测到M4芯片: {chip_info}")
                # M4芯片优化配置
                # 1. 文本切分 - CPU密集型，但不需要太多内存
                num_chunk_processes = max(2, physical_cores - 2)
                
                # 2. 嵌入计算 - MPS (GPU) 瓶颈任务
                # M4 MPS支持异步执行但并行性有限
                # 通常2-3个进程能达到最佳平衡
                num_embedding_processes = 2  
                
                # 3. 队列大小优化
                chunk_queue_size = 5000  # 较小的缓冲区，防止内存过载
                result_queue_size = 10000  # 适中的结果缓冲区
                
                # 4. 数据库并发写入连接
                db_pool_size = min(physical_cores, 6)
                
                # 5. 设置嵌入批处理大小
                embedding_batch_size = 512  # 降低批处理大小提高响应性
            else:
                # 其他Apple Silicon芯片
                logger.info(f"检测到其他Apple Silicon芯片: {chip_info}")
                num_chunk_processes = max(2, physical_cores // 2)
                num_embedding_processes = min(3, physical_cores // 4) 
                chunk_queue_size = 4000
                result_queue_size = 6000
                db_pool_size = min(physical_cores // 2, 4)
                embedding_batch_size = 256
        except:
            # 无法检测具体型号，使用通用配置
            logger.info("使用通用Apple Silicon优化")
            num_chunk_processes = max(2, physical_cores // 2)
            num_embedding_processes = 2
            chunk_queue_size = 4000
            result_queue_size = 6000
            db_pool_size = 4
            embedding_batch_size = 256
    else:
        # 非Apple Silicon优化
        logger.info("使用通用优化配置")
        num_chunk_processes = max(2, physical_cores // 2)
        num_embedding_processes = min(2, max(1, physical_cores // 4))
        chunk_queue_size = 2500
        result_queue_size = 4000
        db_pool_size = min(physical_cores // 2, 8)
        embedding_batch_size = 256
    
    # 打印优化配置
    logger.info(f"📊 优化配置:")
    logger.info(f"  → 文本切分进程数: {num_chunk_processes}")
    logger.info(f"  → 嵌入计算进程数: {num_embedding_processes}")
    logger.info(f"  → 数据库连接池大小: {db_pool_size}")
    logger.info(f"  → 嵌入批处理大小: {embedding_batch_size}")
    logger.info(f"  → 文本块队列大小: {chunk_queue_size}")
    logger.info(f"  → 结果队列大小: {result_queue_size}")
    
    return {
        'num_chunk_processes': num_chunk_processes,
        'num_embedding_processes': num_embedding_processes,
        'chunk_queue_size': chunk_queue_size,
        'result_queue_size': result_queue_size,
        'db_pool_size': db_pool_size,
        'embedding_batch_size': embedding_batch_size
    }

# 初始化设备
device = init_device()