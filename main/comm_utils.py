#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通信工具模块 - 提供分布式节点间通信功能
"""

import os
import time
import json
import socket
import threading
import traceback
import psutil
import zmq
from config import logger, COORDINATOR, WORKER


class DistributedComm:
    """分布式节点间通信类"""
    
    def __init__(self, role, coordinator_ip='localhost', coordinator_port=5555, worker_port=5556):
        """
        初始化分布式通信
        
        参数:
            role: 节点角色 (coordinator 或 worker)
            coordinator_ip: 协调器IP地址
            coordinator_port: 协调器端口
            worker_port: 工作节点通信端口
        """
        self.role = role
        self.coordinator_ip = coordinator_ip
        self.coordinator_port = coordinator_port
        self.worker_port = worker_port
        self.context = zmq.Context()
        self.sockets = {}
        self.heartbeat_interval = 10  # 心跳间隔(秒)
        self.last_heartbeat = time.time()
        self.active = True
        self.identity = None  # 工作节点的唯一标识
        
        # 初始化通信
        self._init_communication()
        
        # 启动心跳线程 (仅工作节点)
        if self.role == WORKER:
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
        
    def _init_communication(self):
        """根据节点角色初始化通信套接字"""
        if self.role == COORDINATOR:
            try:
                # 协调器需要一个ROUTER套接字来接收工作节点的连接
                self.sockets['workers'] = self.context.socket(zmq.ROUTER)
                self.sockets['workers'].setsockopt(zmq.RCVTIMEO, 5000)  # 5秒接收超时
                self.sockets['workers'].setsockopt(zmq.LINGER, 0)  # 无延迟关闭
                self.sockets['workers'].bind(f"tcp://*:{self.coordinator_port}")
                
                # 协调器还需要一个ROUTER套接字向工作节点发送任务
                self.sockets['tasks'] = self.context.socket(zmq.ROUTER)
                self.sockets['tasks'].setsockopt(zmq.SNDTIMEO, 5000)  # 5秒发送超时
                self.sockets['tasks'].setsockopt(zmq.LINGER, 0)  # 无延迟关闭
                self.sockets['tasks'].bind(f"tcp://*:{self.worker_port}")
                
                logger.info(f"协调器监听在端口 {self.coordinator_port}(workers) 和 {self.worker_port}(tasks)")
            except Exception as e:
                logger.error(f"初始化协调器套接字失败: {e}")
                logger.error(traceback.format_exc())
                raise
        else:
            try:
                # 工作节点需要连接到协调器
                self.sockets['coordinator'] = self.context.socket(zmq.DEALER)
                # 为这个工作节点生成唯一标识
                self.identity = f"worker-{socket.gethostname()}-{os.getpid()}"
                self.sockets['coordinator'].setsockopt_string(zmq.IDENTITY, self.identity)
                self.sockets['coordinator'].setsockopt(zmq.SNDTIMEO, 5000)  # 5秒发送超时
                self.sockets['coordinator'].setsockopt(zmq.RCVTIMEO, 5000)  # 5秒接收超时
                self.sockets['coordinator'].setsockopt(zmq.LINGER, 0)  # 无延迟关闭
                self.sockets['coordinator'].connect(f"tcp://{self.coordinator_ip}:{self.coordinator_port}")
                
                # 工作节点还需要一个套接字来接收任务
                self.sockets['tasks'] = self.context.socket(zmq.DEALER)
                self.sockets['tasks'].setsockopt_string(zmq.IDENTITY, self.identity)
                self.sockets['tasks'].setsockopt(zmq.SNDTIMEO, 5000)  # 5秒发送超时
                self.sockets['tasks'].setsockopt(zmq.RCVTIMEO, 5000)  # 5秒接收超时
                self.sockets['tasks'].setsockopt(zmq.LINGER, 0)  # 无延迟关闭
                self.sockets['tasks'].connect(f"tcp://{self.coordinator_ip}:{self.worker_port}")
                
                logger.info(f"工作节点 {self.identity} 已连接到协调器 {self.coordinator_ip}")
            except Exception as e:
                logger.error(f"初始化工作节点套接字失败: {e}")
                logger.error(traceback.format_exc())
                raise

    def _heartbeat_loop(self):
        """工作节点心跳线程，定期向协调器发送状态信息"""
        while self.active:
            try:
                if time.time() - self.last_heartbeat >= self.heartbeat_interval:
                    heartbeat = {
                        'type': 'heartbeat',
                        'id': self.identity,
                        'time': time.time(),
                        'stats': {
                            'cpu': psutil.cpu_percent(),
                            'memory': psutil.virtual_memory().percent
                        }
                    }
                    self.send_message('coordinator', heartbeat)
                    self.last_heartbeat = time.time()
                
                time.sleep(1)
            except Exception as e:
                logger.warning(f"发送心跳时出错: {e}")
                time.sleep(3)  # 失败后稍长等待

    def send_message(self, socket_name, message, timeout=5000):
        """
        发送消息到指定套接字
        
        参数:
            socket_name: 套接字名称 ('coordinator', 'workers', 'tasks')
            message: 要发送的消息 (将被转换为JSON)
            timeout: 发送超时(毫秒)
            
        返回:
            发送成功返回True，否则返回False
        """
        try:
            # 设置发送超时
            if socket_name in self.sockets:
                self.sockets[socket_name].setsockopt(zmq.SNDTIMEO, timeout)
            
            # 将消息转换为JSON字符串
            json_data = json.dumps(message)
            logger.debug(f"发送消息到 {socket_name}: {json_data[:100]}...")  # 只打印前100个字符
            
            # 发送前检查套接字是否可用
            if socket_name not in self.sockets:
                logger.error(f"尝试发送消息到不存在的套接字: {socket_name}")
                return False
                
            self.sockets[socket_name].send_string(json_data)
            return True
        except zmq.error.Again:
            logger.warning(f"发送消息到 {socket_name} 超时")
            return False
        except Exception as e:
            logger.error(f"发送消息时出错: {e}")
            logger.error(traceback.format_exc())
            return False

    def receive_message(self, socket_name, timeout=2000):
        """
        从指定套接字接收消息
        
        参数:
            socket_name: 套接字名称 ('coordinator', 'workers', 'tasks')
            timeout: 接收超时(毫秒)
            
        返回:
            接收到的消息对象，超时或错误返回None
        """
        try:
            # 设置接收超时
            if socket_name in self.sockets:
                self.sockets[socket_name].setsockopt(zmq.RCVTIMEO, timeout)
            
            # 接收前检查套接字是否可用
            if socket_name not in self.sockets:
                logger.error(f"尝试从不存在的套接字接收: {socket_name}")
                return None
                
            # 检查是否为ROUTER套接字 (协调器接收工作节点消息)
            if self.role == COORDINATOR and socket_name in ['workers', 'tasks']:
                # ROUTER套接字需要接收身份标识和消息内容
                try:
                    identity = self.sockets[socket_name].recv_string()
                    message = self.sockets[socket_name].recv_string()
                    logger.debug(f"从节点 {identity} 接收到消息")
                    
                    if not message or message.strip() == "":
                        return None
                        
                    parsed_message = json.loads(message)
                    
                    # 处理心跳消息 - 记录但不返回
                    if isinstance(parsed_message, dict) and parsed_message.get('type') == 'heartbeat':
                        logger.debug(f"收到工作节点 {identity} 的心跳, 状态: CPU {parsed_message.get('stats', {}).get('cpu', 'N/A')}%, "
                                f"内存 {parsed_message.get('stats', {}).get('memory', 'N/A')}%")
                        return None
                        
                    return parsed_message
                except zmq.error.Again:
                    return None
            else:
                # 普通套接字直接接收消息
                message = self.sockets[socket_name].recv_string()
                
                if not message or message.strip() == "":
                    return None
                    
                return json.loads(message)
        except zmq.error.Again:
            # 超时，没有收到消息
            return None
        except Exception as e:
            logger.error(f"接收消息时出错: {e}")
            logger.error(traceback.format_exc())
            if 'message' in locals():
                logger.error(f"Raw message content (first 100 chars): {repr(message[:100] if message else 'None')}")
            return None       
    def close(self):
        """关闭所有通信资源"""
        self.active = False
        for socket_name, socket in self.sockets.items():
            try:
                socket.close(0)  # 立即关闭
                logger.debug(f"已关闭套接字: {socket_name}")
            except:
                pass
        try:
            self.context.term()
            logger.debug("ZMQ上下文已终止")
        except:
            pass
        logger.info("通信资源已释放")