#!/usr/bin/env python3
# debug_test_framework.py

import sys
import os
import subprocess

def run_test_like_framework():
    """使用与测试框架相同的方式运行"""
    
    print("=== 使用pytest运行单个测试函数 ===")
    
    # 使用pytest运行单个测试函数，模仿测试框架的环境
    test_command = [
        sys.executable, "-m", "pytest",
        "/home/luming/workspace/ray-project/ray/python/ray/tests/test_actor_pool.py::test_get_next",
        "-v",
        "-s"
    ]
    
    print("执行命令:", " ".join(test_command))
    result = subprocess.run(test_command, capture_output=True, text=True)
    
    print("返回值:", result.returncode)
    print("标准输出:")
    print(result.stdout)
    if result.stderr:
        print("标准错误:")
        print(result.stderr)
    
    return result.returncode == 0

def check_test_framework_differences():
    """检查测试框架可能设置的特殊环境"""
    
    print("\n=== 检查测试框架环境差异 ===")
    
    # 检查可能的环境变量
    env_vars = [
        'RAY_RAYLET_START_WAIT_TIME_S',
        'RAY_BACKEND_LOG_LEVEL', 
        'RAY_OVERRIDE_RESOURCES',
        'RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER',
        'RAY_REDIS_ADDRESS'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        print(f"{var}: {value}")
    
    # 检查测试框架可能设置的配置
    try:
        import ray
        from ray._private.test_utils import wait_for_condition
        
        print("测试工具模块可用")
    except ImportError as e:
        print(f"测试工具导入失败: {e}")

if __name__ == "__main__":
    check_test_framework_differences()
    success = run_test_like_framework()
    
    if success:
        print("\n✅ 确认：测试框架环境可以正常工作")
        print("问题在于手动初始化时缺少测试框架的特定设置")
    else:
        print("\n❌ 即使在测试框架中也有问题")