#!/usr/bin/env python3
# test_dashboard.py

import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

try:
    print("=== 修正后的Dashboard测试 ===")
    
    # 导入dashboard模块
    import ray.dashboard
    print("✓ ray.dashboard 导入成功")
    
    # 修正导入：从agent导入DashboardAgent
    try:
        from ray.dashboard.agent import DashboardAgent
        print("✓ DashboardAgent类导入成功")
    except ImportError as e:
        print(f"✗ DashboardAgent类导入失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 同样，检查dashboard的导入
    try:
        from ray.dashboard.dashboard import Dashboard
        print("✓ Dashboard类导入成功")
    except ImportError as e:
        print(f"✗ Dashboard类导入失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查Ray常量
    import ray._private.ray_constants as ray_constants
    print(f"✓ ray_constants导入成功")
    
    # 尝试初始化Ray
    print("\n=== 测试Ray初始化 ===")
    import ray
    
    ray.init(include_dashboard=True, ignore_reinit_error=True, logging_level=logging.INFO)
    print("✓ Ray初始化成功")
    
    # 测试基本功能
    @ray.remote
    def test_remote():
        return "Ray核心功能工作正常!"
    
    result = ray.get(test_remote.remote())
    print(f"✓ 远程函数测试: {result}")
    
    # 尝试获取dashboard URL
    try:
        dashboard_url = ray.get_dashboard_url()
        print(f"✓ Dashboard URL: {dashboard_url}")
    except Exception as e:
        print(f"⚠ 无法获取Dashboard URL: {e}")
    
    ray.shutdown()
    print("✓ Ray关闭成功")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()