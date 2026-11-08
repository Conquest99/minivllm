from datetime import datetime


def get_current_time() -> str:
    """获取当前时间的字符串表示，格式为 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def query_write_instruction(researcher_prompts: str, max_number: int, date: str = None) -> str:
    if date is None:
        date = get_current_time()

    # 大致就是这个意思，下面是一个示例的 prompt
    prompt = f"""
    instructions:
    你是一个研究员助手，当前时间是 {date}。
    你需要返回不超过 {max_number} 条与研究员提供的提示相关的指令。
    研究环境的提示是: {researcher_prompts}
    你的回答应该是一个JSON Schema格式的列表，每个元素是一个指令对象，包含以下字段：
    - "instruction": 指令的文本内容，类型为字符串。
    JSON Schema示例:
    '''JSON
    
    [
        {
        "instruction": "示例指令1"
            "query" : [示例查询1,示例查询2]
        },

    ]
    '''
    """
    return prompt


formatted_prompt = query_write_instruction(
    researcher_prompts=get_reasearcher_prompts(state["message"]),  # 获取研究员的提示
    max_number=get_max_number(state["message"]),  # 获取最大数量
    date=date,
)
