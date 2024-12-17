import torch

# def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.95, repetition_penalty=1.1):
#     """
#     使用模型生成回答
#     Args:
#         model: 已加载的模型实例
#         tokenizer: 已加载的tokenizer实例
#         prompt: 输入提示
#         max_new_tokens: 最大生成token数，默认512
#         temperature: 采样温度，控制输出的随机性，默认0.2
#         top_p: 累积概率阈值，默认0.95
#         repetition_penalty: 重复惩罚系数，默认1.5
#     Returns:
#         str: 模型生成的回答
#     """
#     device = model.device
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     # print("inputs: ", inputs, "\n")
#     # print(type(inputs), type(inputs["input_ids"]), inputs["input_ids"])
#     inputs_list = inputs["input_ids"].tolist()[0]
#     # print("inputs_list: ", len(inputs_list), "\n")
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             repetition_penalty=repetition_penalty,
#         )
#     # 将tensor转换为list
#     outputs_list = outputs[0].tolist()
#     # print("outputs' token_list: ", outputs_list, "\n")
#     # print("outputs_list: ", len(outputs_list), "\n")
    
#     # 去掉输入的token，包含：model 以及后面的模型回答
#     response_list = outputs_list[len(inputs_list):]
#     print("response_list: ", response_list, "\n")

#     # 再找到 “model” 后面的内容，2516是model的token id。前三个token应该是：“<start_of_turn>model\n”
#     if 2516 in response_list[0:3]: # 如果"model"在response_list的前3个token中，则需要去掉 model，保留后面的内容
#         final_response_list = response_list[response_list.index(2516):]
#     else:
#         final_response_list = response_list
#     # 解码内容
#     response = tokenizer.decode(final_response_list, skip_special_tokens=True)
    
#     return response.strip()

# 优化后的 generate_response 函数
def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.95, repetition_penalty=1.1):
    """
    使用模型生成回答的优化版本
    """
    device = model.device
    
    # 1. 使用with torch.inference_mode()替代with torch.no_grad()
    # inference_mode()在推理时比no_grad()更快
    
    # 2. 预先分配好输入张量的设备
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.inference_mode():
        # 3. 使用更高效的生成参数
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            # 4. 添加early_stopping=True可以在生成完成时提前停止
            early_stopping=True,
            # 5. 使用num_beams=1进行贪婪解码
            num_beams=1
        )
    
    # 6. 直接切片获取新生成的token，避免不必要的列表转换
    response_tokens = outputs[0, input_length:]
    
    # 7. 优化model token的检测
    if response_tokens.size(0) >= 3 and 2516 in response_tokens[:3]:
        model_token_index = (response_tokens == 2516).nonzero(as_tuple=True)[0][0]
        response_tokens = response_tokens[model_token_index:]
    
    # 8. 直接解码tensor，避免额外的列表转换
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response.strip()

# 完整对话模板
SYSTEM_TEMPLATE = "<start_of_turn>system\n{context}\n<end_of_turn><eos>\n"
USER_TEMPLATE = "<start_of_turn>user\n{prompt}\n<end_of_turn><eos>\n"
MODEL_TEMPLATE = "<start_of_turn>model\n{response}\n<end_of_turn><eos>\n"

def format_system(context):
    """
    格式化系统输入的context
    Args:
        context: 原始context
    Returns:
        str: 格式化后的context
    """
    return SYSTEM_TEMPLATE.format(context=context)

def format_user(prompt):
    """
    格式化用户输入的prompt
    Args:
        prompt: 原始prompt
    Returns:
        str: 格式化后的prompt
    """
    return USER_TEMPLATE.format(prompt=prompt)

def format_model(response):
    """
    格式化模型的回答
    Args:
        response: 模型原始回答
    Returns:
        str: 格式化后的回答
    """
    return MODEL_TEMPLATE.format(response=response) 

# def format_conversation(system_context, user_prompt):
#     conversation = (
#         f"{SYSTEM_TEMPLATE.format(context=system_context)}"
#         f"{USER_TEMPLATE.format(prompt=user_prompt)}"
#         f"{MODEL_TEMPLATE}"  # 模型回复的模板
#     )
#     return conversation

def apply_chat_template(dialogue):
    """
    dialogue的结构类似：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "model", "content": "Hello! How can I assist you today?"}
    ]
    注意，dialogue 也可能没有 system 和 model 的内容，也就是：
    [
        {"role": "user", "content": "Hello!"},
    ] 
    此时需要额外处理
    """

    # 法一：使用for循环
    # dialogue_str = ""
    # for i in range(len(dialogue)):
    #     sub_dialogue = dialogue[i] # sub_dialogue 是一个 dict，结构类似 {"role": "user", "content": "Hello!"}
    #     if sub_dialogue["role"] == "system":
    #         dialogue_str += format_system(sub_dialogue["content"])
    #     elif sub_dialogue["role"] == "user":
    #         dialogue_str += format_user(sub_dialogue["content"])    
    #     elif sub_dialogue["role"] == "model":
    #         dialogue_str += format_model(sub_dialogue["content"])
    #     else:
    #         raise ValueError(f"Unknown role: {sub_dialogue['role']}")
        
    # return dialogue_str

    # 法二：使用字典映射来替代if-elif判断，较快效率
    # 使用字典映射来替代if-elif判断
    role_format_map = {
        "system": format_system,
        "user": format_user,
        "model": format_model
    }
    
    try:
        # 使用列表推导式和join来替代循环累加字符串
        return "".join(role_format_map[msg["role"]](msg["content"]) for msg in dialogue)
    except KeyError as e:
        raise ValueError(f"Unknown role: {e}")