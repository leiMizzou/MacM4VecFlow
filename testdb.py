#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PostgreSQL数据库连接测试脚本

这个脚本用于测试PostgreSQL数据库连接，并尝试验证数据库架构和表。
它可以帮助诊断分布式向量处理系统的数据库连接问题。
"""

import sys
import time
import argparse
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def test_connection(host, port, dbname, user, password, schema):
    """测试数据库连接并验证架构"""
    print("\n===== PostgreSQL连接测试 =====\n")
    
    # 连接信息
    print(f"连接信息:")
    print(f"  主机: {host}")
    print(f"  端口: {port}")
    print(f"  数据库: {dbname}")
    print(f"  用户: {user}")
    print(f"  Schema: {schema}")
    print(f"  密码: {'*' * len(password)}")
    
    # 准备连接字符串
    conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    
    # 步骤1: 测试基本连接
    print("\n步骤1: 测试基本连接")
    try:
        print("  尝试连接...")
        conn = psycopg2.connect(conn_string)
        print("  ✓ 连接成功!")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        print("\n可能的解决方案:")
        print("  - 确认数据库服务器是否运行")
        print("  - 检查IP地址和端口是否正确")
        print("  - 验证用户名和密码是否正确")
        print("  - 确认PostgreSQL是否允许远程连接 (pg_hba.conf)")
        print("  - 检查防火墙规则是否允许连接")
        return
    
    # 步骤2: 测试数据库版本和连接信息
    print("\n步骤2: 获取数据库信息")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"  ✓ 数据库版本: {version}")
        
        cursor.execute("SELECT current_user, current_database();")
        user, db = cursor.fetchone()
        print(f"  ✓ 当前用户: {user}, 数据库: {db}")
    except Exception as e:
        print(f"  ✗ 无法获取数据库信息: {e}")
    
    # 步骤3: 测试Schema存在
    print(f"\n步骤3: 检查Schema '{schema}'")
    try:
        cursor.execute("""
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name = %s;
        """, (schema,))
        
        if cursor.fetchone():
            print(f"  ✓ Schema '{schema}' 存在")
        else:
            print(f"  ✗ Schema '{schema}' 不存在")
            print("    提示: 可能需要创建这个schema:")
            print(f"    CREATE SCHEMA {schema};")
    except Exception as e:
        print(f"  ✗ 检查Schema时出错: {e}")
    
    # 步骤4: 测试Vector扩展
    print("\n步骤4: 检查Vector扩展")
    try:
        cursor.execute("""
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname = 'vector';
        """)
        
        result = cursor.fetchone()
        if result:
            extname, extversion = result
            print(f"  ✓ Vector扩展已安装 (版本: {extversion})")
        else:
            print(f"  ✗ Vector扩展未安装")
            print("    提示: 需要安装Vector扩展:")
            print(f"    CREATE EXTENSION vector;")
    except Exception as e:
        print(f"  ✗ 检查Vector扩展时出错: {e}")
    
    # 步骤5: 获取数据表信息
    print(f"\n步骤5: 获取Schema '{schema}'中的表")
    try:
        cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name;
        """, (schema,))
        
        tables = cursor.fetchall()
        if tables:
            print(f"  ✓ 已找到 {len(tables)} 个表:")
            for i, (table_name,) in enumerate(tables, 1):
                print(f"    {i}. {table_name}")
        else:
            print(f"  ! Schema '{schema}'中没有找到表")
    except Exception as e:
        print(f"  ✗ 获取表信息时出错: {e}")
    
    # 步骤6: 测试数据访问
    print("\n步骤6: 测试连接池")
    try:
        from psycopg2.pool import ThreadedConnectionPool
        
        print("  尝试创建连接池...")
        pool = ThreadedConnectionPool(1, 5, conn_string)
        print("  ✓ 连接池创建成功")
        
        # 测试从池中获取连接
        pooled_conn = pool.getconn()
        print("  ✓ 从池中获取连接成功")
        
        # 归还连接到池
        pool.putconn(pooled_conn)
        print("  ✓ 归还连接到池成功")
        
        # 关闭池
        pool.closeall()
        print("  ✓ 连接池关闭成功")
    except Exception as e:
        print(f"  ✗ 连接池测试失败: {e}")
    
    # 关闭连接
    try:
        conn.close()
        print("\n数据库连接已关闭")
    except:
        pass
    
    print("\n===== 测试完成 =====")

def main():
    """主函数，解析命令行参数并执行测试"""
    parser = argparse.ArgumentParser(description='PostgreSQL数据库连接测试')
    parser.add_argument('--host', default='192.168.2.2', help='数据库主机地址')
    parser.add_argument('--port', type=int, default=5432, help='数据库端口')
    parser.add_argument('--dbname', default='fda_device', help='数据库名称')
    parser.add_argument('--user', default='postgres', help='数据库用户名')
    parser.add_argument('--password', default='12345687', help='数据库密码')
    parser.add_argument('--schema', default='device', help='数据库模式名称')
    
    args = parser.parse_args()
    
    test_connection(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        schema=args.schema
    )

if __name__ == "__main__":
    main()