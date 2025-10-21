#!/usr/bin/env python3
# test_core_functionality.py

import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

try:
    print("=== 测试Ray核心功能 (禁用Dashboard) ===")
    
    import ray
    
    # 明确禁用dashboard，专注于核心功能
    print("正在初始化Ray (dashboard已禁用)...")
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    print("✓ Ray初始化成功")
    
    # 测试1: 基本远程函数
    print("\n--- 测试1: 基本远程函数 ---")
    @ray.remote
    def add(a, b):
        return a + b
    
    result = ray.get(add.remote(5, 3))
    print(f"✓ 远程加法测试: 5 + 3 = {result}")
    
    # 测试2: 对象引用
    print("\n--- 测试2: 对象引用 ---")
    @ray.remote
    def process_data(data):
        return f"处理后的数据: {data.upper()}"
    
    data_ref = process_data.remote("hello ray")
    result = ray.get(data_ref)
    print(f"✓ 对象引用测试: {result}")
    
    # 测试3: 多个任务并行
    print("\n--- 测试3: 并行任务 ---")
    @ray.remote
    def square(x):
        return x * x
    
    # 同时提交多个任务
    refs = [square.remote(i) for i in range(5)]
    results = ray.get(refs)
    print(f"✓ 并行任务测试: {results}")
    
    # 测试4: 类远程方法
    print("\n--- 测试4: 远程类方法 ---")
    @ray.remote
    class Counter:
        def __init__(self):
            self.value = 0
        
        def increment(self):
            self.value += 1
            return self.value
        
        def get_value(self):
            return self.value
    
    # 创建远程actor
    counter = Counter.remote()
    
    # 调用远程方法
    ray.get(counter.increment.remote())
    ray.get(counter.increment.remote())
    final_value = ray.get(counter.get_value.remote())
    print(f"✓ 远程Actor测试: 计数器值 = {final_value}")
    
    print("\n🎉 所有核心功能测试通过！")
    print("说明Ray的编译和基本分布式功能是正常的")
    
    # 关闭Ray
    ray.shutdown()
    print("✓ Ray关闭成功")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n这表明确实存在更基础的问题，需要先解决核心功能")