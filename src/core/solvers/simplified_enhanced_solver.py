# AutoDL官方学术资源加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


import sys
import os

# 添加项目根目录到Python路径
project_root = "/home/cuipeng/Gemma"
sys.path.append(project_root)

# 现在可以正常导入src下的模块
from src.core.model.model_initializer import initialize_model_and_tokenizer
from src.core.utils.model_utils import generate_response, apply_chat_template
from src.core.solvers.l2m import L2MSolver
from src.core.solvers.self_verification import SelfVerifier

import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
from peft import PeftModel  # type: ignore # 导入PeftModel用于加载微调模型

class SimplifiedEnhancedSolver:
    def __init__(self, l2m_solver, verifier):
        """初始化增强型求解器"""
        if l2m_solver is None or verifier is None:
            raise ValueError("必须提供已初始化的 L2MSolver 和 SelfVerifier 实例")
        
        self.l2m = l2m_solver
        self.verifier = verifier
        
        # 简化系统提示词
        self.system_prompt = """你是一个专业的问题解决专家。请确保答案准确完整、逻辑清晰、层次分明。"""

    def solve_complex_question(self, question):
        """使用简化的解决方案处理复杂问题"""
        print("开始处理问题:", question)
        
        # 1. 使用L2M分解并解决问题
        print("\n1. 分解并解决问题...")
        l2m_result = self.l2m.solve(question)
        
        # 2. 只对最终答案进行一次验证和改进
        print("\n2. 验证和改进最终答案...")
        final_verified = self.verifier.verify_and_improve(
            question,
            l2m_result["final_answer"]
        )
        
        return {
            "original_question": question,
            "sub_solutions": l2m_result["sub_solutions"],
            "final_answer": final_verified["improved_answer"],
            "verification": {
                "first_verification": final_verified["first_verification"],
                "final_verification": final_verified["final_verification"]
            }
        }

    def analyze_verification_results(self, result):
        """简化的验证结果分析"""
        return {
            "verification_summary": {
                "initial_issues": result["verification"]["first_verification"],
                "final_status": result["verification"]["final_verification"]
            }
        }