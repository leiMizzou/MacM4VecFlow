# MacM4VecFlow
This project is built on the macA4 architecture, forming a distributed system by connecting multiple A4 devices. It leverages the combined processing power of these machines to run parallel tasks.Specifically, the system targets a PostgreSQL database, where it processes selected records by applying modern vector embedding techniques. The resulting vector representations are then stored in dedicated vector fields within the database, enabling efficient data retrieval and advanced analytics.

# 分布式向量嵌入处理系统

本系统是一个高性能的分布式文本向量化工具，专为大规模文本处理和向量嵌入生成而设计。它能够将文本数据切分成合适的语义块，并利用预训练语言模型生成高质量的向量表示，最终存储在PostgreSQL数据库中。

## 系统特性

* **分布式处理**: 支持协调器-工作节点架构，实现多机并行处理
* **高效文本切分**: 智能切分文本为语义连贯的块
* **向量嵌入缓存**: 减少重复计算，提高处理效率
* **Apple Silicon优化**: 针对M4等芯片特别优化，充分利用MPS加速
* **健壮的错误处理**: 完善的错误恢复和容错机制
* **实时进度监控**: 详细的进度报告和健康监控
* **灵活的配置**: 支持自定义数据库结构和处理参数

## 系统要求

* Python 3.8+
* PostgreSQL 13+ (需安装`vector`扩展)
* 足够的RAM (建议8GB+)
* 推荐使用Apple Silicon Mac (M系列芯片) 获取最佳性能

## 安装

1. 克隆代码库:
```bash
git clone https://github.com/leiMizzou/MacM4VecFlow.git
cd MacM4VecFlow/main
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 准备PostgreSQL数据库:
```sql
-- 安装vector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建Schema (可选)
CREATE SCHEMA IF NOT EXISTS device;
```

## 文件结构

本系统由多个模块组成:

* `config.py` - 系统配置和常量定义
* `system_utils.py` - 系统工具和优化函数
* `db_processor.py` - 数据库处理模块
* `embedding_utils.py` - 文本嵌入处理模块
* `comm_utils.py` - 分布式通信模块
* `coordinator_node.py` - 协调节点实现
* `worker_node.py` - 工作节点实现
* `main.py` - 主程序入口

## 使用方法

### 启动协调器节点

```bash
python main.py --role coordinator \
    --db-host 192.168.2.1\
    --db-port 5432 \
    --db-name fda_database \
    --db-user postgres \
    --db-password 12345687 \
    --db-schema fda_data \
    --source-table event_mdr_text \
    --target-table event_text_vectors \
    --id-field id \
    --text-field text \
    --host 0.0.0.0 \
    --coordinator-port 5555 \
    --worker-port 5556 \
    --limit 500000 \
    --log-level INFO

```

### 启动工作节点

```bash
python main.py --role worker \
    --db-host 192.168.2.1 \
    --db-port 5432 \
    --db-name fda_database \
    --db-user postgres \
    --db-password 12345687 \
    --db-schema fda_data \
    --source-table event_mdr_text \
    --target-table event_text_vectors \
    --id-field id \
    --text-field text \
    --coordinator-host 192.168.2.1 \
    --coordinator-port 5555 \
    --worker-port 5556 \
    --limit 500000 \
    --log-level INFO
```

## 配置说明

### 数据库配置

数据库表结构可以在`config.py`文件中的`DB_TABLE_CONFIG`字典中定义。默认配置如下:

```python
DB_TABLE_CONFIG = {
    # 主键字段，自动递增
    'id_field': 'id',
    'id_type': 'SERIAL PRIMARY KEY',
    
    # 报告编号字段，用于标识数据来源
    'report_field': 'fulltext_id',
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
```

### 文本切分配置

文本切分参数可以在`config.py`文件中的`CHUNK_CONFIG`字典中定义:

```python
CHUNK_CONFIG = {
    'chunk_size': 128,  # 目标令牌数
    'overlap_tokens': 20,  # 块之间的重叠令牌数
}
```

### 命令行参数

主要命令行参数:

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--role` | 节点角色 (coordinator 或 worker) | 必须指定 |
| `--db-host` | 数据库主机地址 | localhost |
| `--db-port` | 数据库端口 | 5432 |
| `--db-name` | 数据库名称 | postgres |
| `--db-user` | 数据库用户名 | postgres |
| `--db-password` | 数据库密码 | postgres |
| `--db-schema` | 数据库模式 | public |
| `--source-table` | 源数据表名 | event_texts |
| `--target-table` | 向量表名称 | event_text_vectors |
| `--id-field` | ID字段名 | id |
| `--text-field` | 文本字段名 | text |
| `--limit` | 处理的最大记录数 | 500000 |
| `--log-level` | 日志级别 | INFO |
| `--model` | 使用的嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |

## 性能调优

系统会自动检测硬件配置并进行优化，但您也可以手动调整以下参数:

1. 在`system_utils.py`中的`optimize_pipeline_concurrency()`函数中调整进程数和队列大小
2. 在`embedding_utils.py`中调整`batch_size`以平衡内存使用和计算效率
3. 对于大型数据集，可以调整`main.py`中的`--limit`参数分批处理

## 错误处理与恢复

系统内置了多级错误处理和恢复机制:

1. 通信错误: 自动重试连接和消息发送
2. 处理错误: 本地恢复或跳过问题数据
3. 数据库错误: 事务回滚和批次拆分
4. 系统监控: 检测处理停滞并尝试恢复

## 常见问题

**Q: 工作节点无法连接到协调器**
A: 确保协调器IP地址正确，并检查防火墙是否开放了相应端口。

**Q: 处理速度较慢**
A: 检查嵌入模型大小、批处理大小和数据库连接数。对于大型数据集，考虑增加工作节点数量。

**Q: 出现内存错误**
A: 减小批处理大小和队列大小，或者考虑增加系统内存。

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

希望这个工具能对您的工作有所帮助！
