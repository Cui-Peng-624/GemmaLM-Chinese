# AutoDL官方学术资源加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
from peft import PeftModel  # type: ignore

from l2m import L2MSolver
from self_verification import SelfVerifier
from model_utils import generate_response, format_prompt, USER_TEMPLATE

class EnhancedSolver:
    def __init__(self, l2m_solver, verifier):
        """
        初始化增强型求解器
        Args:
            l2m_solver: 已初始化的 L2MSolver 实例
            verifier: 已初始化的 SelfVerifier 实例
        """
        if l2m_solver is None or verifier is None:
            raise ValueError("必须提供已初始化的 L2MSolver 和 SelfVerifier 实例")
        
        self.l2m = l2m_solver
        self.verifier = verifier

    def solve_complex_question(self, question):
        """
        使用增强的解决方案处理复杂问题
        1. 使用L2M分解并解决问题
        2. 对每个子问题的解答进行验证和改进
        3. 生成并验证最终答案
        """
        print("开始处理问题:", question)
        
        # 1. 使用L2M获取初始解答
        print("\n1. 分解并解决问题...")
        l2m_result = self.l2m.solve(question)
        # l2m_result 结构：{"original_question": complex_question, "sub_solutions": solutions, "final_answer": final_answer}
        # 其中 solutions 为: [{"question": sub_question, "solution": sub_solution}, {"question": sub_question, "solution": sub_solution}, ...]
        
        # 2. 对每个子问题的解答进行验证和改进
        print("\n2. 验证和改进子问题解答...")
        verified_solutions = []
        improved_context = ""
        
        for sub_l2m_result in l2m_result["sub_solutions"]: # sub_l2m_result 结构：{"question": sub_question, "solution": sub_solution}
            print(f"\n处理子问题: {sub_l2m_result['question']}")
            verified = self.verifier.verify_and_improve(
                sub_l2m_result["question"],
                sub_l2m_result["solution"]
            )
            # verified 结构：{"original_answer": ***, "first_verification", "improved_answer", "final_verification"}

            verified_solutions.append({
                "question": sub_l2m_result["question"],
                "original_solution": sub_l2m_result["solution"],
                "verified_solution": verified["improved_answer"]
            })
            improved_context += f"\n问题：{sub_l2m_result['question']}\n答案：{verified['improved_answer']}\n"
        
        # 3. 基于改进后的答案重新生成最终答案
        print("\n3. 生成最终答案...")
        final_prompt = format_prompt(f"""你需要根据以下改进后的解题过程和原始问题，生成一个完整且准确的答案。
                                    基于以下解题过程：{improved_context}
                                    请总结出原始问题的完整答案：{question}""")
        
        new_final_answer = generate_response(self.l2m.model, self.l2m.tokenizer, final_prompt)
        
        # 4. 对最终答案进行验证和改进
        print("\n4. 验证和改进最终答案...")
        final_verified = self.verifier.verify_and_improve(
            question,
            new_final_answer
        )
        # final_verified 结构：{"original_answer": ***, "first_verification", "improved_answer", "final_verification"}
        
        return {
            "original_question": question,
            "sub_solutions": verified_solutions,
            "final_answer": final_verified["improved_answer"],
            "solution_process": {
                "decomposition": l2m_result["sub_solutions"],
                "improved_context": improved_context
            }
        }
    
    # analyze_verification_results 函数用于分析验证结果，提供改进建议
    def analyze_verification_results(self, result):
        """分析验证结果，提供改进建议"""
        analysis = {
            "sub_problems": [],
            "final_answer": {
                "improvement_level": None,
                "key_improvements": [],
                "remaining_issues": []
            }
        }
        
        # 分析子问题的改进
        for sub_sol in result["sub_solutions"]:
            verified_sol = sub_sol["verified_solution"]
            analysis["sub_problems"].append({
                "question": sub_sol["question"],
                "improvements": self._extract_improvements(
                    verified_sol["first_verification"],
                    verified_sol["final_verification"]
                )
            })
        
        # 分析最终答案的改进
        final_answer = result["final_answer"]
        analysis["final_answer"].update(
            self._extract_improvements(
                final_answer["first_verification"],
                final_answer["final_verification"]
            )
        )
        
        return analysis

    def _extract_improvements(self, first_verification, final_verification):
        """从验证结果中提取改进信息"""
        return {
            "initial_issues": self._parse_verification(first_verification),
            "resolved_issues": self._parse_verification(final_verification),
        }

    def _parse_verification(self, verification):
        """解析验证结果中的具体问题"""
        # 这里可以添加更详细的解析逻辑
        return verification

# # 使用示例
# if __name__ == "__main__":
#     from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
#     from peft import PeftModel # type: ignore
#     from l2m import L2MSolver
#     from self_verification import SelfVerifier
    
#     # 初始化基础组件
#     model_path = "google/gemma-2-9b"
#     cache_dir = "/root/autodl-tmp/gemma"
#     lora_path = "/root/autodl-tmp/models/stage1/gemma-base-zh-final"
    
#     # 初始化tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    
#     # 初始化基础模型
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )
    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         cache_dir=cache_dir,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         quantization_config=quantization_config
#     )
    
#     model = PeftModel.from_pretrained(base_model, lora_path)
    
#     # 初始化L2M求解器和验证器
#     l2m_solver = L2MSolver(model, tokenizer)
#     verifier = SelfVerifier(model, tokenizer)
    
#     # 初始化增强求解器
#     solver = EnhancedSolver(
#         l2m_solver=l2m_solver,
#         verifier=verifier
#     )
    
#     # 测试问题
#     question = "请解释量子计算机的工作原理及其潜在应用。"
#     result = solver.solve_complex_question(question)
    
#     print("最终答案:", result["final_answer"]["improved_answer"])
#     print("\n解答过程分析:", solver.analyze_verification_results(result))


