'''
# 在一次tool——call中，一些可能的变量
finish_reason = "tool_calls"
ChatCompletionMessage(
    content = '' //调用工具不需要返回什么别的
    role = "assistant" #消息来源是assistant生成的而非是大模型组织的语言
    rufuse = False #不拒绝调用工具
    function_call = "auto" #自动调用工具
    tool_calls = [ChatCompletionMessageToolCall(
        id = 'xxx' #工具调用的唯一标识符
        function = FunctionCall(
            name = "calculator", #工具的名称
            arguments = {
                "expression": "1+1" #工具的参数
            }
        type = "function" #这是一个被用来调用的工具
        index = 0 #工具调用的索引,这是第0个被调用的工具

)




# 系统拦截型号，调用工具
Tool_Map = {
    "calculator":calculator_tool
}
if finish_reason == "tool_calls":
    message.append(msg)
    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        fn_args = tool_call.function.arguments
        fn = Tool_Map.get(fn_name)
        result = fn(**fn_args)
        message.append({
            "role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)
                        })




# 一次简化的完整的工具调用流程
message = {
    content = '帮我计算1+1', #内容
    role = "user", #消息来源是用户
}
ChatCompletionMessage(
    content = '' //调用工具不需要返回什么别的
    role = "assistant" #消息来源是assistant生成的而非是大模型组织的语言
    annotations = None #没有注释
    audio = None #没有音频
    function_call = "auto" #自动调用工具
    tool_calls = [ChatCompletionMessageToolCall(
        id = 'xxx',
        function = FunctionCall(
            name = "calculator",
            arguments = {
                "expression": "1+1"
            }
        ),
        type = "function",
        index = 0
    )] #工具调用列表
),


{role = "tool",
  "tool_call_id": tool_call.id,
    "content": json.dumps(result)} #工具调用的结果



response = client.chat.comletions.create(
    model = MODEL_NAME,
    messages = message,
)
final_response = response.choices[0].message.content
print(final_response) #打印最终的结果

'''
# MCP示例

async with stdio_client(server_params) as (read,write): # 创建一个标准输入输出的客户端
    async with ClientSession(read,write) as session: # 创建一个客户端会话
        await session.initlized() # 初始化
        tool_result = await session.tool_lists() # 获取工具列表


# tool_result结果
ListToolResult(
    meta = None, #这个是工具的元信息
    nextCursor = None, #这个是工具的下一页游标
    tools = [
        tool = {
            name = "calculator", #工具的名称
            description = "计算器", #工具的描述
            inputSchema = {
                'propertys': {
                    'expression': { #工具的参数
                        'title' : 'expression', #参数的标题
                        'type': 'string', #参数的类型
                        'description': '计算表达式', #参数的描述
                        'required': True #参数是否必填
                    },
                    }
                },
            outputSchema = { #输出的schema，在这里是计算机这个工具的输出
                'type': '', #输出的类型
                'properties': {
                    'result': { #输出的属性
                        'title': 'result', #输出的标题
                        'type': 'string', #输出的类型
                        'description': '计算结果' #输出的描述
                    } #关闭result
                } #关闭properties
            }, #关闭outputSchema
            icons = None, #工具的图标
            annotations = None, #工具的注释
            meta = None, #工具的元信息
            excution = {
                'type': 'function', #工具的执行类型
                'function': 'calculator_tool' #工具的执行函数
            } #关闭excution
        } #关闭tool字典
    ] #关闭tools列表
) #关闭ListToolResult(


# MCP调用工具示例
message = {
    'role' : 'user', # 消息的角色是用户
    'content': user_message
}
response = session.chat.completions.create(
    model = MODEL_NAME, # 模型名称
    messages = message, # 消息内容
    tools = tool_result, # 工具列表
    tool_choice = 'auto' # 工具选择方式为自动
)



if response.finish_reason == 'tool_calls': # 如果完成原因是工具调用
    message.append(msg) # 将消息添加到消息列表中
    for tool_call in response.tool_calls: # 遍历工具调用列表
        fn_name = tool_call.function.name # 获取工具名称
        fn_args = tool_call.function.arguments # 获取工具参数
        fn = Tool_Map.get(fn_name) # 从工具映射中获取工具函数

        # 这一段是Tool_User的核心逻辑，调用工具函数并获取结果，然后将结果添加到消息列表中
        # result = fn(**fn_args) # 调用工具函数并获取结果
        # message.append({ # 将工具调用结果添加到消息列表中
        #     'role': 'tool', # 消息的角色是工具
        #     'tool_call_id': tool_call.id, # 工具调用的唯一标识符
        #     'content': json.dumps(result) # 工具调用结果的内容
        # }) # 关闭消息字典

        #MCP的逻辑
        call_result = await session.tool_call(fn_name,fn_args) # 调用工具函数并获取结果
        raw = call_result.content[0].text # 返回的结果
        message.append({ # 将工具调用结果添加到消息列表中
            'role': 'tool', # 消息的角色是工具
            'tool_call_id': tool_call.id, # 工具调用的唯一标识符
            'content': json.dumps(raw) # 工具调用结果的内容
        }) # 关闭消息字典