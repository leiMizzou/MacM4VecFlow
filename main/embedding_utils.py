#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
嵌入处理模块 - 提供文本切分、向量嵌入生成等功能
"""

import re
import time
import random
import numpy as np
import traceback
import threading
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import logger, CHUNK_CONFIG, DEFAULT_MODEL
from system_utils import device, clean_memory


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


def load_optimized_model(model_name, device):
    """使用半精度加载模型以减少内存占用并提高速度"""
    logger.info(f"正在优化加载模型到 {device} 设备...")
    
    # 使用fp16加载模型
    model = SentenceTransformer(model_name, device=device)
    
    # 如果设备支持，转换为半精度
    if device == "mps":
        # 确保MPS支持fp16
        if hasattr(torch.ops.mps, '_has_fp16_support') and torch.ops.mps._has_fp16_support():
            # 将模型转换为fp16
            model.half()
            logger.info("✅ 模型已转换为半精度(fp16)")
        
        # 如果支持特定的MPS图像集优化
        if hasattr(torch.mps, 'optimize_for'):
            # 指定为转换器优化
            torch.mps.optimize_for(model, category="transformer_inference")
            logger.info("✅ 使用MPS图优化器优化")
    
    return model


class TextProcessor:
    """文本处理类，负责文本切分和向量嵌入生成"""
    
    def __init__(self, model_name=DEFAULT_MODEL, vector_dim=384):
        """
        初始化文本处理器
        
        参数:
            model_name: 使用的模型名称
            vector_dim: 向量维度
        """
        self.vector_dim = vector_dim
        self.chunk_size = CHUNK_CONFIG['chunk_size']
        self.overlap_tokens = CHUNK_CONFIG['overlap_tokens']
        self.batch_size = 512  # 默认批处理大小
        
        logger.info(f"正在加载模型到 {device} 设备...")
        
        try:
            # 使用优化的模型加载
            self.model = load_optimized_model(model_name, device)
            logger.info(f"✅ 成功加载模型: {model_name}")
            
            # 尝试使用 MPS 加速
            if device == "mps":
                logger.info(f"✅ 模型成功加载到 MPS 设备")
                
                # 尝试预热模型
                dummy_texts = ["This is a warm up text to initialize the model pipeline."] * 4
                _ = self.model.encode(dummy_texts, convert_to_numpy=True)
                logger.info(f"✅ 模型预热完成")
        except Exception as e:
            logger.error(f"⚠️ 模型加载到 {device} 失败: {e}")
            logger.error(traceback.format_exc())
            try:
                logger.info("尝试回退到 CPU...")
                self.model = SentenceTransformer(model_name)
                logger.info(f"✅ 成功加载模型到 CPU: {model_name}")
            except Exception as e:
                logger.error(f"⚠️ 模型加载完全失败: {e}")
                raise
        
        # 加载对应的 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✅ 成功加载对应 tokenizer")
        except Exception as e:
            logger.warning(f"⚠️ 加载 tokenizer 失败: {e}，使用备用方法")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                logger.info(f"✅ 成功加载备用 tokenizer")
            except Exception as e:
                logger.error(f"⚠️ 加载备用 tokenizer 也失败: {e}")
                raise
        
        # 启用 MPS 图捕获 (如果可用)
        if device == "mps" and hasattr(torch.mps, 'enable_graph_capture'):
            try:
                torch.mps.enable_graph_capture()
                logger.info(f"✅ 启用 MPS 图捕获以提高性能")
            except:
                pass
        
        logger.info(f"模型加载完成，使用的设备: {self.model.device}")
        
        # 创建嵌入缓存
        self.embedding_cache = EmbeddingCache(max_size=20000)

    def process_text(self, text: str) -> List[str]:
        """
        将文本切分为语义合理的块
        
        参数:
            text: 输入文本
            
        返回:
            文本块列表
        """
        if not text or text.strip() == "":
            return []
        
        try:
            # 1. 清理和规范化文本
            text = text.replace('\r', ' ').replace('\t', ' ')
            
            # 2. 尝试使用自然句子边界进行分块
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
            overlap_tokens = self.overlap_tokens
            
            for sentence in sentences:
                # 对句子进行分词
                try:
                    sentence_tokens = self.tokenizer.encode(sentence)
                    sentence_length = len(sentence_tokens)
                except Exception as e:
                    # 处理可能的tokenizer错误
                    logger.warning(f"分词句子时出错: {e}, 句子: {sentence[:50]}...")
                    # 估计长度作为回退
                    sentence_length = len(sentence.split()) + 5  # 额外余量
                
                # 通过分割来跳过极长的单句
                if sentence_length > target_chunk_size:
                    if current_chunk:
                        # 编码和解码以获得适当的标记边界
                        try:
                            chunk_text = self.tokenizer.decode(
                                self.tokenizer.encode(' '.join(current_chunk))
                            )
                            chunks.append(chunk_text)
                        except Exception as e:
                            # 处理编码错误
                            logger.warning(f"编码/解码块时出错: {e}")
                            # 简单拼接作为回退
                            chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # 将长句处理成具有重叠的多个块
                    for i in range(0, sentence_length, target_chunk_size - overlap_tokens):
                        try:
                            # 处理子句片段
                            sub_tokens = sentence_tokens[i:i + target_chunk_size]
                            sub_text = self.tokenizer.decode(sub_tokens)
                            chunks.append(sub_text)
                        except Exception as e:
                            # 处理解码错误
                            logger.warning(f"解码子句时出错: {e}")
                            # 使用原句片段作为回退
                            words = sentence.split()
                            start_idx = i // 2  # 近似字转换为词
                            end_idx = min(len(words), start_idx + (target_chunk_size // 2))
                            sub_text = ' '.join(words[start_idx:end_idx])
                            if sub_text.strip():
                                chunks.append(sub_text)
                    
                    continue
                
                # 检查添加此句是否会超出块大小
                if current_length + sentence_length > target_chunk_size:
                    # 当前块已满，将其添加到块中并开始新块
                    if current_chunk:
                        try:
                            chunk_text = self.tokenizer.decode(
                                self.tokenizer.encode(' '.join(current_chunk))
                            )
                            chunks.append(chunk_text)
                        except Exception as e:
                            # 处理编码错误
                            logger.warning(f"编码/解码块时出错: {e}")
                            # 简单拼接作为回退
                            chunks.append(' '.join(current_chunk))
                    
                    # 用此句开始新块
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # 添加到当前块
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # 添加剩余的块
            if current_chunk:
                try:
                    chunk_text = self.tokenizer.decode(
                        self.tokenizer.encode(' '.join(current_chunk))
                    )
                    chunks.append(chunk_text)
                except Exception as e:
                    # 处理编码错误
                    logger.warning(f"处理最后一个块时出错: {e}")
                    # 简单拼接作为回退
                    chunks.append(' '.join(current_chunk))
            
            # 确保即使对于很短的文本也至少有一个块
            if not chunks and text.strip():
                chunks.append(text.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"处理文本时发生错误: {e}")
            logger.error(traceback.format_exc())
            # 安全回退 - 简单分割
            try:
                # 简单的分段，每段最多200个单词
                words = text.split()
                simple_chunks = []
                for i in range(0, len(words), 200):
                    chunk = ' '.join(words[i:i+200])
                    if chunk.strip():
                        simple_chunks.append(chunk)
                return simple_chunks or [text.strip()]
            except:
                # 最终回退 - 返回整个文本作为单个块
                return [text.strip()] if text.strip() else []

    def get_embeddings_batch_optimized(self, texts: List[str]) -> List[List[float]]:
        """
        优化的批量嵌入生成，自适应批处理大小，更好的错误处理
        
        参数:
            texts: 文本列表
            
        返回:
            嵌入向量列表
        """
        if not texts:
            return []
            
        # 过滤空文本并预处理
        valid_texts = []
        valid_indices = []
        
        for i, t in enumerate(texts):
            if t and t.strip():
                # 规范化文本
                cleaned_text = re.sub(r'\s+', ' ', t).strip()
                if cleaned_text:
                    valid_texts.append(cleaned_text)
                    valid_indices.append(i)
        
        if not valid_texts:
            return [[0.0] * self.vector_dim for _ in range(len(texts))]
        
        try:
            # 动态批处理大小 - 基于文本长度和可用内存
            avg_token_len = sum(len(t.split()) for t in valid_texts) / len(valid_texts)
            
            # 针对设备类型优化的批处理大小
            if device == "mps":
                # 根据平均文本长度动态调整批大小
                if avg_token_len < 50:
                    batch_size = 1024  # 短文本使用更大批次
                elif avg_token_len < 100:
                    batch_size = 512   # 中等长度文本
                else:
                    batch_size = 256   # 长文本使用小批次
            else:
                # CPU处理使用较小批次避免内存问题
                batch_size = min(256, max(32, int(5000 / avg_token_len)))
            
            # 使用错误恢复的嵌入生成
            # 将文本分批处理，提高容错性
            all_embeddings = [None] * len(valid_texts)
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                try:
                    # 使用normalize_embeddings=True自动归一化，提高性能
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        batch_size=batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    
                    # 转换为numpy以便进一步处理(如果尚未转换)
                    if hasattr(batch_embeddings, 'cpu'):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                        
                    # 将批次嵌入放回相应位置
                    for j, emb in enumerate(batch_embeddings):
                        idx = i + j
                        if idx < len(all_embeddings):
                            all_embeddings[idx] = emb
                            
                except Exception as e:
                    logger.error(f"生成批次嵌入时出错 ({i}:{i+batch_size}): {e}")
                    logger.error(traceback.format_exc())
                    # 为这个批次生成零向量
                    for j in range(len(batch_texts)):
                        idx = i + j
                        if idx < len(all_embeddings):
                            all_embeddings[idx] = np.zeros(self.vector_dim)
                
                # 定期清理MPS缓存
                if device == "mps" and i % (batch_size * 3) == 0:
                    torch.mps.empty_cache()
                    
            # 构建最终结果 - 填充原始位置
            result = [[0.0] * self.vector_dim for _ in range(len(texts))]
            
            for orig_idx, valid_idx in enumerate(valid_indices):
                if orig_idx < len(all_embeddings) and all_embeddings[orig_idx] is not None:
                    # 获取嵌入
                    emb = all_embeddings[orig_idx]
                    
                    # 转换为列表
                    emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    
                    # 处理NaN值
                    emb_list = [0.0 if np.isnan(x) else x for x in emb_list]
                    
                    # 存储在原始位置
                    result[valid_idx] = emb_list
                
            # 手动触发MPS缓存清理
            if device == "mps":
                torch.mps.empty_cache()
                
            return result
        except Exception as e:
            logger.error(f"批量获取嵌入向量时出错: {e}")
            logger.error(traceback.format_exc())
            # 出错时返回零向量
            return [[0.0] * self.vector_dim for _ in range(len(texts))]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        带缓存和错误恢复的批量嵌入生成
        
        参数:
            texts: 文本列表
            
        返回:
            嵌入向量列表
        """
        try:
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
        except Exception as e:
            logger.error(f"处理嵌入时出现严重错误: {e}")
            logger.error(traceback.format_exc())
            # 发生严重错误时，返回零向量
            return [[0.0] * self.vector_dim for _ in range(len(texts))]