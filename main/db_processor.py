#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库处理模块 - 负责数据库连接、表创建和向量存储等功能
"""

import time
import random
import traceback
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from typing import List, Dict, Tuple, Any, Optional, Generator
from config import logger, DB_TABLE_CONFIG, VECTOR_DIM
from io import StringIO


class DatabaseProcessor:
    """数据库处理类，负责PostgreSQL数据库操作"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        初始化数据库处理器
        
        参数:
            db_config: 包含数据库连接参数的字典
        """
        self.db_config = db_config
        self.conn = None
        self.pool = None
        self.vector_dim = VECTOR_DIM
        
        # 从配置获取表结构
        self.table_config = DB_TABLE_CONFIG
        
        # 初始化连接池
        self.init_connection_pool()

    def init_connection_pool(self, min_conn=2, max_conn=10):
        """初始化数据库连接池，提高并发性能"""
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
            # 尝试普通连接作为回退
            try:
                self.conn = psycopg2.connect(**self.db_config)
                logger.info("回退到普通数据库连接")
                return True
            except Exception as e:
                logger.error(f"普通数据库连接也失败: {e}")
                return False

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
                try:
                    connection = self.pool.getconn()
                    yield connection
                except Exception as e:
                    logger.warning(f"从连接池获取连接失败: {e}, 尝试直接连接")
                    # 如果从池获取失败，尝试直接连接
                    if not self.conn or self.conn.closed:
                        self.connect()
                    yield self.conn
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
                try:
                    self.pool.putconn(connection)
                except:
                    # 如果归还失败，尝试关闭连接
                    try:
                        connection.close()
                    except:
                        pass
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
            logger.error(traceback.format_exc())
            return False

    def check_schema_exists(self, schema_name="device") -> bool:
        """检查指定的schema是否存在"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    SELECT schema_name FROM information_schema.schemata 
                    WHERE schema_name = %s
                    """, (schema_name,))
                    if cur.fetchone() is None:
                        logger.error(f"{schema_name} schema 不存在")
                        return False
                    logger.info(f"{schema_name} schema 已存在")
                    return True
        except Exception as e:
            logger.error(f"检查 schema 时出错: {e}")
            logger.error(traceback.format_exc())
            return False

    def setup_vector_table(self, table_name: str, schema="device") -> bool:
        """
        设置向量数据库表及索引
        
        参数:
            table_name: 表名
            schema: 模式名
        
        返回:
            创建成功返回True，否则返回False
        """
        if not self.check_vector_extension():
            logger.error("未找到必要的 vector 扩展，无法创建向量表")
            return False
            
        if not self.check_schema_exists(schema):
            logger.error(f"指定的schema '{schema}'不存在，无法创建表")
            return False
            
        full_table_name = f"{schema}.{table_name}"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 使用直接的表结构定义，而非配置字典
                    # 这样更不容易出错，且易于阅读和调试
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {full_table_name} (
                        id SERIAL PRIMARY KEY,
                        report_number TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        text_chunk TEXT NOT NULL,
                        embedding public.vector({self.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(chunk_id)
                    );
                    """
                    
                    # 添加错误处理和重试逻辑
                    max_retries = 3
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            cur.execute(create_table_sql)
                            break
                        except Exception as e:
                            retry_count += 1
                            logger.warning(f"创建表失败(尝试 {retry_count}/{max_retries}): {e}")
                            if retry_count >= max_retries:
                                raise
                            time.sleep(1)
                    
                    # 创建向量索引
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            # 构建索引创建SQL
                            index_name = f"{table_name}_embedding_idx"
                            cur.execute(f"""
                            CREATE INDEX IF NOT EXISTS {index_name} 
                            ON {full_table_name} USING ivfflat (embedding public.vector_cosine_ops)
                            WITH (lists = 100);
                            """)
                            break
                        except Exception as e:
                            retry_count += 1
                            logger.warning(f"创建索引失败(尝试 {retry_count}/{max_retries}): {e}")
                            if retry_count >= max_retries:
                                raise
                            time.sleep(1)
                    
                    conn.commit()
                    logger.info(f"向量表 {full_table_name} 创建成功")
                    return True
        except Exception as e:
            logger.error(f"设置向量数据库表失败: {e}")
            logger.error(traceback.format_exc())
            return False
        
    def store_vectors(self, table_name: str, data: List[Tuple], schema="device") -> bool:
        """
        将向量数据批量存储到数据库
        
        参数:
            table_name: 表名
            data: 包含(report_number, chunk_id, text_chunk, embedding)元组的列表
            schema: 模式名
        
        返回:
            存储成功返回True，否则返回False
        """
        if not data:
            logger.warning("没有向量数据需要存储")
            return True
            
        full_table_name = f"{schema}.{table_name}"
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        # 使用COPY命令优化批量插入
                        from io import StringIO
                        
                        # 创建具有相同结构的临时表
                        temp_table = f"temp_{table_name}_{int(time.time())}_{random.randint(1000, 9999)}"
                        cur.execute(f"""
                        CREATE TEMPORARY TABLE {temp_table} (
                            report_number TEXT,
                            chunk_id TEXT,
                            text_chunk TEXT,
                            embedding public.vector({self.vector_dim})
                        ) ON COMMIT DROP;
                        """)
                        
                        # 准备数据进行COPY
                        buffer = StringIO()
                        for report_number, chunk_id, text_chunk, embedding in data:
                            # 将嵌入格式化为PostgreSQL向量字符串
                            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                            # 转义分隔符和引号
                            text = text_chunk.replace('\t', ' ').replace('\n', ' ').replace('\\', '\\\\')
                            buffer.write(f"{report_number}\t{chunk_id}\t{text}\t{embedding_str}\n")
                        
                        buffer.seek(0)
                        
                        # 使用COPY进行批量导入
                        cur.copy_expert(f"COPY {temp_table} FROM STDIN", buffer)
                        
                        # 将临时表数据插入到正式表，处理冲突
                        cur.execute(f"""
                        INSERT INTO {full_table_name} (
                            report_number, 
                            chunk_id, 
                            text_chunk, 
                            embedding
                        )
                        SELECT 
                            report_number, 
                            chunk_id, 
                            text_chunk, 
                            embedding 
                        FROM {temp_table}
                        ON CONFLICT (chunk_id) DO NOTHING;
                        """)
                        
                        conn.commit()
                    return True
            except Exception as e:
                retry_count += 1
                logger.warning(f"存储向量失败 (尝试 {retry_count}/{max_retries}): {e}")
                logger.warning(traceback.format_exc())
                
                # 如果不是最后一次尝试，等待后重试
                if retry_count < max_retries:
                    time.sleep(2)  # 增加等待时间
                    continue
                
                # 回退到使用execute_values方法
                logger.info("回退到备用存储方法...")
                try:
                    with self.get_connection() as conn:
                        with conn.cursor() as cur:
                            # 使用更小的批次执行插入操作
                            batch_size = 1000  # 减小批次大小
                            
                            for i in range(0, len(data), batch_size):
                                try:
                                    batch = data[i:i + batch_size]
                                    
                                    # 构建SQL语句
                                    sql = f"""
                                    INSERT INTO {full_table_name} 
                                    (report_number, chunk_id, text_chunk, embedding)
                                    VALUES %s
                                    ON CONFLICT (chunk_id) DO NOTHING;
                                    """
                                    
                                    execute_values(cur, sql, batch)
                                    
                                    # 提交当前批次，减少事务大小
                                    conn.commit()
                                except Exception as inner_e:
                                    logger.error(f"存储批次 {i}:{i+batch_size} 时出错: {inner_e}")
                                    conn.rollback()  # 回滚失败的事务
                                    
                                    # 尝试一条一条地插入
                                    for item in batch:
                                        try:
                                            sql = f"""
                                            INSERT INTO {full_table_name}
                                            (report_number, chunk_id, text_chunk, embedding)
                                            VALUES (%s, %s, %s, %s)
                                            ON CONFLICT (chunk_id) DO NOTHING;
                                            """
                                            
                                            cur.execute(sql, item)
                                            conn.commit()
                                        except Exception as item_e:
                                            logger.error(f"存储单个向量时出错: {item_e}")
                                            conn.rollback()
                    return True
                except Exception as backup_e:
                    logger.error(f"备用存储方法也失败: {backup_e}")
                    return False

    def fetch_partitioned_data(self, table_name: str, source_field: str, text_field: str, 
                            partition_id: int, total_partitions: int, 
                            limit=500000, batch_size=10000, schema="device") -> Generator[List[Tuple[str, str]], None, None]:
        """
        分区获取数据，用于分布式处理
        
        参数:
            table_name: 源数据表名
            source_field: 源ID字段名
            text_field: 文本字段名
            partition_id: 当前分区ID (从1开始)
            total_partitions: 总分区数
            limit: 最大处理记录数
            batch_size: 每批次获取的记录数
            schema: 数据库模式名
        
        返回:
            生成器，每次返回一批(id, text)元组列表
        """
        full_table_name = f"{schema}.{table_name}"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        # 创建临时表进行分区
                        # 注意：text_column_name变量用于在临时表中明确指定列名
                        text_column_name = "text_content"  # 使用明确的列名避免混淆
                        
                        cur.execute(f"""
                        CREATE TEMPORARY TABLE temp_partitioned_data AS
                        SELECT {source_field} as report_number, {text_field} as {text_column_name},
                            ROW_NUMBER() OVER (ORDER BY {source_field}) as row_num
                        FROM {full_table_name} 
                        WHERE {text_field} IS NOT NULL 
                        LIMIT %s
                        """, (limit,))
                        
                        # 获取总行数
                        cur.execute("SELECT COUNT(*) FROM temp_partitioned_data")
                        total_rows = cur.fetchone()[0]
                        
                        # 计算每个分区的大小
                        partition_size = total_rows // total_partitions
                        if total_rows % total_partitions > 0:
                            partition_size += 1
                        
                        # 计算当前分区的开始和结束
                        start_row = (partition_id - 1) * partition_size + 1
                        end_row = min(partition_id * partition_size, total_rows)
                        
                        logger.info(f"分区 {partition_id}/{total_partitions}: 处理行 {start_row} 到 {end_row} (共 {end_row-start_row+1} 行)")
                        
                        # 查询当前分区的数据 - 分批获取避免内存压力
                        total_fetched = 0
                        current_start = start_row
                        
                        while current_start <= end_row:
                            current_end = min(current_start + batch_size - 1, end_row)
                            
                            logger.debug(f"获取行 {current_start} 到 {current_end}")
                            
                            # 使用正确的列名从临时表中选择数据
                            cur.execute(f"""
                            SELECT report_number, {text_column_name}
                            FROM temp_partitioned_data
                            WHERE row_num BETWEEN %s AND %s
                            ORDER BY row_num
                            """, (current_start, current_end))
                            
                            records = cur.fetchall()
                            if not records:
                                break
                                
                            total_fetched += len(records)
                            yield records
                            
                            # 更新当前起始位置
                            current_start = current_end + 1
                        
                        logger.info(f"分区 {partition_id} 总共获取了 {total_fetched} 条记录")
                        
                    except Exception as e:
                        logger.error(f"分区数据获取出错: {e}")
                        logger.error(traceback.format_exc())
                        # 尝试备用查询方法
                        logger.info("尝试使用备用查询方法...")
                        
                        try:
                            # 不使用临时表的备用方法
                            total_query = f"""
                            SELECT COUNT(*) 
                            FROM {full_table_name} 
                            WHERE {text_field} IS NOT NULL 
                            LIMIT %s
                            """
                            cur.execute(total_query, (limit,))
                            total_rows = cur.fetchone()[0]
                            
                            # 计算每个分区的大小
                            partition_size = total_rows // total_partitions
                            if total_rows % total_partitions > 0:
                                partition_size += 1
                            
                            # 计算偏移量
                            offset = (partition_id - 1) * partition_size
                            
                            # 直接查询带偏移量
                            data_query = f"""
                            SELECT {source_field} as report_number, {text_field}
                            FROM {full_table_name} 
                            WHERE {text_field} IS NOT NULL 
                            ORDER BY {source_field}
                            LIMIT %s OFFSET %s
                            """
                            
                            # 分批获取
                            total_fetched = 0
                            for i in range(0, partition_size, batch_size):
                                current_limit = min(batch_size, partition_size - i)
                                current_offset = offset + i
                                
                                logger.debug(f"获取 {current_limit} 行，偏移量 {current_offset}")
                                cur.execute(data_query, (current_limit, current_offset))
                                
                                records = cur.fetchall()
                                if not records:
                                    break
                                    
                                total_fetched += len(records)
                                yield records
                            
                            logger.info(f"备用方法: 分区 {partition_id} 总共获取了 {total_fetched} 条记录")
                            
                        except Exception as backup_e:
                            logger.error(f"备用查询方法也失败: {backup_e}")
                            logger.error(traceback.format_exc())
                            yield []
                    finally:
                        try:
                            # 尝试删除临时表
                            cur.execute("DROP TABLE IF EXISTS temp_partitioned_data")
                        except:
                            pass
                    
        except Exception as e:
            logger.error(f"获取分区数据时出错: {e}")
            logger.error(traceback.format_exc())
            yield []