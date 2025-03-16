#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版向量搜索工具 - 直接使用全限定的vector类型名称
python vector_search.py --query "clips in different materials in surgical procedure" --threshold 0.2 --limit 20
"""

import psycopg2
import argparse
import time
from sentence_transformers import SentenceTransformer

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='简化版向量搜索工具')
    parser.add_argument('--query', type=str, required=True, help='查询文本')
    parser.add_argument('--db-host', default='169.254.22.165', help='数据库主机地址')
    parser.add_argument('--db-port', type=int, default=5432, help='数据库端口')
    parser.add_argument('--db-name', default='fda_device', help='数据库名称')
    parser.add_argument('--db-user', default='postgres', help='数据库用户名')
    parser.add_argument('--db-password', default='12345687', help='数据库密码')
    parser.add_argument('--db-schema', default='device', help='数据库模式')
    parser.add_argument('--table', default='event_text_vectors', help='向量表名称')
    parser.add_argument('--limit', type=int, default=5, help='返回结果数量')
    parser.add_argument('--threshold', type=float, default=0.6, help='相似度阈值(0-1)')
    parser.add_argument('--vector-schema', default='public', help='vector类型所在的模式')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("简化版向量搜索工具")
    print("="*80)
    
    conn = None
    try:
        # 1. 加载模型
        print("正在加载模型 sentence-transformers/all-MiniLM-L6-v2...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ 模型加载成功")
        
        # 2. 生成查询向量
        query_embedding = model.encode(args.query, convert_to_numpy=True).tolist()
        vector_str = f"[{','.join(str(x) for x in query_embedding)}]"
        
        # 3. 连接数据库
        conn = psycopg2.connect(
            host=args.db_host,
            port=args.db_port,
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
        print(f"✓ 成功连接到数据库 {args.db_name} @ {args.db_host}")
        
        # 开启自动提交，避免事务问题
        conn.autocommit = True
        
        # 4. 执行向量查询 - 使用全限定的vector类型名称
        full_table_name = f"{args.db_schema}.{args.table}"
        vector_type = f"{args.vector_schema}.vector"
        
        print(f"\n🔍 查询文本: \"{args.query}\"")
        print(f"📊 相似度阈值: {args.threshold}, 最大结果数: {args.limit}")
        print("-"*80)
        
        with conn.cursor() as cur:
            # 使用全限定的向量类型名称
            query = f"""
            SELECT 
                id, 
                report_number, 
                chunk_id, 
                text_chunk, 
                1 - (embedding <-> '{vector_str}'::{vector_type}) AS similarity
            FROM {full_table_name}
            WHERE 1 - (embedding <-> '{vector_str}'::{vector_type}) > {args.threshold}
            ORDER BY embedding <-> '{vector_str}'::{vector_type} ASC
            LIMIT {args.limit}
            """
            
            try:
                cur.execute(query)
                results = cur.fetchall()
                
                if not results:
                    print("未找到匹配结果")
                else:
                    print(f"找到 {len(results)} 个相似文本:")
                    print("-"*80)
                    
                    # 获取列名
                    column_names = [desc[0] for desc in cur.description]
                    
                    for i, row in enumerate(results, 1):
                        # 将行转换为字典
                        result = dict(zip(column_names, row))
                        similarity_pct = result['similarity'] * 100
                        print(f"【结果 {i}】相似度: {similarity_pct:.2f}%")
                        print(f"ID: {result['id']}, 报告号: {result['report_number']}, 块ID: {result['chunk_id']}")
                        print(f"文本内容: \"{result['text_chunk']}\"\n")
            except Exception as e:
                print(f"✗ 执行查询失败: {e}")
                print("\n尝试诊断问题...")
                
                try:
                    # 检查vector扩展
                    cur.execute("SELECT extname, extnamespace::regnamespace FROM pg_extension WHERE extname = 'vector'")
                    vector_ext = cur.fetchone()
                    if vector_ext:
                        print(f"✓ pgvector扩展已安装在模式 {vector_ext[1]} 中")
                    else:
                        print("✗ pgvector扩展未安装")
                        
                    # 检查搜索路径
                    cur.execute("SHOW search_path")
                    path = cur.fetchone()
                    print(f"✓ 当前搜索路径: {path[0]}")
                    
                    # 检查表结构
                    cur.execute(f"SELECT column_name, data_type, udt_schema, udt_name FROM information_schema.columns WHERE table_schema = '{args.db_schema}' AND table_name = '{args.table}' AND column_name = 'embedding'")
                    col_info = cur.fetchone()
                    if col_info:
                        print(f"✓ embedding列信息: {col_info}")
                    else:
                        print(f"✗ 未找到embedding列")
                except Exception as diag_e:
                    print(f"✗ 诊断失败: {diag_e}")
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if conn:
            conn.close()
            print("数据库连接已关闭")
        print("="*80)


if __name__ == "__main__":
    main()