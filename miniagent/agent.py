"""Main module of MiniAgent, providing core Agent functionality"""

import os
import json
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .logger import get_logger
from .utils.json_utils import parse_json
from .utils.text_utils import smart_truncate
from .utils.reflector import Reflector
from .tools import get_registered_tools, get_tool, get_tool_description

from rich.console import Console
console = Console()

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dangerous command patterns for tool confirmation
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b",  # rm -rf
    r"\brm\s+-[a-zA-Z]*\s+/",      # rm anything under /
    r"\bmkfs\b",                     # format filesystem
    r"\bdd\s+",                      # disk dump
    r":>\s*/",                       # truncate root files
    r"\bchmod\s+-R\s+777\b",        # open permissions recursively
    r"\bchown\s+-R\b",              # recursive ownership change
    r">\s*/etc/",                    # overwrite system config
    r"\bsudo\b",                     # sudo commands
    r"\bshutdown\b|\breboot\b",     # system control
    r"\bkill\s+-9\b",              # force kill
    r"\bpkill\b|\bkillall\b",      # mass kill
    r"\bcurl\b.*\|\s*\bsh\b",      # pipe curl to shell
    r"\bwget\b.*\|\s*\bsh\b",      # pipe wget to shell
    r"[;&|]\s*\brm\s",             # chained rm after other commands
]

_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)

class MiniAgent:
    """
    Main MiniAgent class, providing core functionality for LLM interaction and tool calling
    """

    # Template for text-mode system prompt (tools list injected at runtime)
    
    _TEXT_MODE_PROMPT = """\
{base_prompt}

You are a helpful, empathetic assistant with access to tools. Follow these rules:

Available tools: {tools_prompt}

0.**Skill Selection** (On-Demand):
   - Check if the user's request matches any registered skill (e.g., literature review -> `literature_reviewer`, coding -> `coder`).
   - If matched: Call `use_skill` FIRST, then follow its instructions.
   - If unmatched: Skip. Answer directly or use general tools (search, read/write).

1. **Tool format** (required):
   TOOL: <tool_name>
   ARGS: {{"parameter_name": "parameter_value"}}
   - Use double quotes for strings, no quotes for numbers.
   - For multiline content, use \n for newlines.
   - For example, when the user asks "Create a file hello.py", you should respond:
   TOOL: write
   ARGS: {{"path": "hello.py", "content": "print('Hello World')"}}

2. **When to search**:
   - For factual, recent, or verifiable info (news, data, citations), ALWAYS use tavily_search or web_search first.
   - Use internal knowledge only for common sense, or if search returns nothing.
   - If unsure about a citation, verify via search before quoting.

3. **Batch search** (for efficiency):
   - Combine multiple related queries with ` | ` (OR) in one call, e.g., `"paper A" | "paper B"`.
   - Max 400 characters per call; split if exceeded.

4. **Response & Termination**:
   - After executing a tool, explain the result clearly.
   - **When you believe you have fully answered the user's question (whether or not you used a tool), you MUST start your final response with `FINAL_ANSWER:`**.
   - After using `FINAL_ANSWER:`, do NOT call any more tools. This is the only way to end the task.

5. **Final Output Format**:
   - Use Simplified Chinese as the primary language. Proper nouns, technical terms, acronyms, and file/API names may remain in English.
   - Use markdown for readability when helpful.

6. **Interpersonal skills**:
   - Greet briefly when appropriate.
   - If request is vague, ask clarifying questions BEFORE using tools.
   - Acknowledge user's emotions (e.g., "I understand you're looking for...").
   - Respond naturally, politely, and helpfully.


Remember: Be accurate, concise, and human-like. If unsure, ask rather than guess."""



    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant that can use tools to get information and perform tasks.",
        use_reflector: bool = False,
        confirm_dangerous: bool = False,
        confirm_callback: Optional[Callable[[str], bool]] = None,
        **kwargs
    ):
        """
        Initialize MiniAgent
        
        Args:
            model: Model name, e.g. "gpt-3.5-turbo", "deepseek-chat"
            api_key: API key for the model provider
            base_url: Base URL for the model provider
            temperature: Model temperature
            system_prompt: System prompt to use for the agent
            use_reflector: Whether to use the Reflector to improve reasoning
            confirm_dangerous: If True, dangerous bash commands require confirmation
            confirm_callback: Function(cmd) -> bool for confirmation. Defaults to stdin prompt.
            **kwargs: Additional parameters for the OpenAI client
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self._base_temperature = temperature  # 保存初始基准温度，用于重置
        self.system_prompt = system_prompt

        

        # ========== 插入开始：注入 Skill 目录到 system_prompt ==========
        from .skills import _SKILLS
        skills_catalog_parts = []
        for name, skill in _SKILLS.items():
            desc = skill.description or "无描述"
            skills_catalog_parts.append(f"- {name}: {desc}")
        if skills_catalog_parts:
            skills_catalog = "\n".join(skills_catalog_parts)
            self.system_prompt = self.system_prompt + "\n\n## 可用技能 (Available Skills)\n" + skills_catalog + "\n当用户问题匹配某技能时，可调用 use_skill 工具加载。"
        # ========== 插入结束 ==========

        

        self.tools = []
        self.client = None
        self.use_reflector = use_reflector
        self.confirm_dangerous = confirm_dangerous
        self.confirm_callback = confirm_callback
        self._skill_tool_whitelist = None   # 存储技能允许的工具名称列表，None表示不过滤
        
        # Cache config limits (read env vars once, not per-request)
        self._max_context_messages = int(os.environ.get("MAX_CONTEXT_MESSAGES", "20"))
        self._tool_result_limit = int(os.environ.get("TOOL_RESULT_LIMIT", "800000"))     #16000改到800000，同步改.env文件
        
        # Initialize the LLM client
        self._init_llm_client()
        
        # Initialize reflector if enabled
        if use_reflector:
            self.reflector = Reflector(self.client, self.model)
        else:
            self.reflector = None


        # ==================== 【改动 1 插入开始】 ====================
        # 注册内置的 use_skill 工具
        self.add_tool({
            "name": "use_skill",
            "description": "加载指定名称的技能。当用户的问题匹配某个特定技能（如 coder、researcher、literature_reviewer）时，调用此工具加载其完整指引。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名称，例如 'coder', 'researcher', 'literature_reviewer'"
                    }
                },
                "required": ["skill_name"]
            },
            "executor": self._use_skill_handler  # 指向下面的处理方法
        })
        # 初始化一个占位，用于存储当前加载的技能
        self._loaded_skill = None
        # ==================== 【改动 1 插入结束】 ====================

        
        logger.info(f"MiniAgent initialized, model: {model}, base URL: {base_url or 'default'}, temperature: {temperature}, reflector: {use_reflector}")
    
    def _init_llm_client(self):
        """Initialize the LLM client (OpenAI-compatible for all providers)"""
        try:
            import openai as _openai
            self.client = _openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"LLM client initialized: model={self.model}, base_url={self.base_url or 'default'}")
        except ImportError:
            logger.error("OpenAI package not installed. Please run 'uv sync' or 'pip install -r requirements.txt'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def add_tool(self, tool: Dict[str, Any]) -> None:
        """
        Add a tool to the agent
        
        Args:
            tool: Tool definition, containing name, description, and executor
        """
        if not isinstance(tool, dict):
            raise TypeError("Tool must be a dictionary type")
            
        required_keys = ["name", "description", "executor"]
        for key in required_keys:
            if key not in tool:
                raise ValueError(f"Tool is missing a required field: {key}")
                
        self.tools.append(tool)
        logger.debug(f"Added tool: {tool['name']}")
    
    def load_builtin_tool(self, tool_name: str) -> bool:
        """
        Load a built-in tool
        
        Args:
            tool_name: Tool name
            
        Returns:
            Whether the load was successful
        """
        tool_func = get_tool(tool_name)
        if tool_func:
            # Create tool definition
            tool_desc = get_tool_description(tool_func)
            tool = {
                "name": tool_desc["name"],
                "description": tool_desc["description"],
                "parameters": tool_desc.get("parameters", {}),
                "executor": tool_func
            }
            self.add_tool(tool)
            logger.info(f"Loaded built-in tool: {tool_name}")
            return True
        else:
            logger.warning(f"Built-in tool not found: {tool_name}")
            return False
    
    def get_available_tools(self) -> List[str]:
        """
        Get all available built-in tool names
        
        Returns:
            List of tool names
        """
        return list(get_registered_tools().keys())

    def load_all_tools(self) -> None:
        """Load all registered built-in tools into this agent."""
        for name in self.get_available_tools():
            self.load_builtin_tool(name)


    
    
    
    # ==================== 【改动 2 插入开始】 ====================
    
    def _reset_skill_state(self):
        """
        重置技能加载状态，确保每次新的 run() 调用不受上一次技能残留影响。
        """
        self._loaded_skill = None
        self._skill_tool_whitelist = None
        # 恢复用户最初设定的温度，避免被上次技能修改
        self.temperature = self._base_temperature
        logger.debug("Skill state has been reset.")

    
    
    
    def _use_skill_handler(self, skill_name: str) -> str:
        """
        处理 use_skill 工具调用的执行器。
        只负责加载 Skill 对象并暂存到 self._loaded_skill，不修改 messages。
        主循环会检测到这个变量并注入 prompt。
        """

        from .skills import get_skill, _SKILLS
        
        start = time.perf_counter()
    
        skill = get_skill(skill_name)
        if not skill:
            self._reset_skill_state()
            console.print(f"[dim]❌ Skill load FAILED: '{skill_name}' (not found)[/dim]")
            return f"错误：未找到技能 '{skill_name}'。可用技能：{list(_SKILLS.keys())}"
    
        self._loaded_skill = skill
        self._skill_tool_whitelist = skill.tools
        if skill.temperature is not None:
            self.temperature = skill.temperature

        elapsed = time.perf_counter() - start
        console.print(f"[dim]✅ SKILL LOADED: '{skill_name}' | temp={self.temperature} | tools_whitelist={self._skill_tool_whitelist} | max_iter={skill.max_iterations} | elapsed={elapsed:.3f}s[/dim]")
        
        return f"✅ 成功加载技能 '{skill_name}'。请根据该技能的指引执行任务。"
    # ==================== 【改动 2 插入结束】 ====================


    def _get_filtered_tools(self) -> List[Dict]:
        """
        根据当前加载的技能返回过滤后的工具列表。
        如果技能未指定白名单（_skill_tool_whitelist 为 None），则返回全部工具。
        """
        if self._loaded_skill and self._skill_tool_whitelist is not None:
            return [t for t in self.tools if t["name"] in self._skill_tool_whitelist]
        return self.tools

  
    def _build_tools_prompt(self) -> str:
        """
        构建工具描述字符串，供 system prompt 使用。
        如果当前加载了技能且技能指定了工具白名单，则只显示白名单中的工具。
        """
        # 获取当前应该展示的工具列表（根据技能白名单过滤）
        tools_to_show = self._get_filtered_tools()

        # 如果最终列表为空，直接返回提示信息（避免空描述）
        if not tools_to_show:
            return "(没有可用工具)"

        tools_desc = []
        for tool in tools_to_show:
            params = tool.get("parameters", {})
            param_desc = []
            # 构建参数描述
            for name, schema in params.get("properties", {}).items():
                required = name in params.get("required", [])
                param_desc.append(f"    - {name}: {schema.get('description', '')} {'(required)' if required else ''}")
            params_text = "\n".join(param_desc) if param_desc else "    (none)"
        
            # 组装单个工具的描述（格式与原来保持一致）
            desc = (
                f"\n            Tool: {tool['name']}\n"
                f"            Description: {tool['description']}\n"
                f"            Parameters:\n"
                f"            {params_text}\n"
                f"            "
            )
            tools_desc.append(desc)
    
        return "\n".join(tools_desc)
    



    def _parse_tool_call(self, content: str) -> Optional[Dict]:
        """
        Parse tool call from LLM response.

        Supports two text patterns:
          1. TOOL: <name>  ARGS: {json}
          2. Tool/工具: <name>  Args/参数: {json}

        Args:
            content: LLM response content

        Returns:
            Tool call information or None
        """
        logger.debug(f"Parsing tool call from content (length={len(content)})")

        # ========== 【新增】解析 MCP 格式的 XML 标签 ==========
        # 匹配 <｜｜DSML｜｜tool_calls> ... </｜｜DSML｜｜tool_calls>
        xml_pattern = re.compile(
            r'<｜｜DSML｜｜tool_calls>\s*(.*?)\s*</｜｜DSML｜｜tool_calls>',
            re.DOTALL
        )
        xml_match = xml_pattern.search(content)
        if xml_match:
            inner = xml_match.group(1)
            # 提取所有 <｜｜DSML｜｜invoke name="..."> ... </｜｜DSML｜｜invoke>
            invoke_pattern = re.compile(
                r'<｜｜DSML｜｜invoke\s+name="([^"]+)"\s*>(.*?)</｜｜DSML｜｜invoke>',
                re.DOTALL
            )
            invokes = invoke_pattern.findall(inner)
            if invokes:
                # 只取第一个工具调用（简化处理）
                name, args_xml = invokes[0]
                # 提取所有 <｜｜DSML｜｜parameter name="..." ...> ... </｜｜DSML｜｜parameter> 或自闭合
                param_pattern = re.compile(
                    r'<｜｜DSML｜｜parameter\s+name="([^"]+)"(?:\s+string="true")?\s*>(.*?)</｜｜DSML｜｜parameter>',
                    re.DOTALL
                )
                params = param_pattern.findall(args_xml)
                args = {}
                for pname, pvalue in params:
                    args[pname] = pvalue.strip()
                # 如果有参数则返回
                if args:
                    logger.info(f"Parsed MCP tool call: {name} with args {args}")
                    return {"name": name, "arguments": args}
                else:
                    # 如果没有参数，可能是无参调用
                    logger.info(f"Parsed MCP tool call: {name} with no args")
                    return {"name": name, "arguments": {}}
        # ======================================================


        # ===== 新增：获取当前已注册的工具名称列表 =====
        registered_tool_names = [tool["name"] for tool in self.tools]

        # Two clean patterns: strict and relaxed
        tool_name_patterns = [
            r"TOOL:\s*(\w+)\s*ARGS:\s*",
            r"(?:Tool|工具|USE TOOL|使用工具|工具名称|TOL):\s*(\w+)\s*(?:ARGS|Args|参数|WITH ARGS|工具参数|Arguments):\s*",
        ]

        for pattern in tool_name_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                name = match.group(1)
                remaining = content[match.end():]

                # Extract balanced JSON using brace counting
                args_str = self._extract_balanced_json(remaining)
                if not args_str:
                    continue

                logger.debug(f"Matched tool '{name}', args length={len(args_str)}")

                # Try strict parse first, then loose
                try:
                    return {"name": name, "arguments": json.loads(args_str)}
                except json.JSONDecodeError:
                    args = parse_json(args_str)
                    if args:
                        logger.info(f"Parsed tool call: {name} with {len(args)} args")
                        return {"name": name, "arguments": args}

                logger.warning(f"Failed to parse tool arguments for {name}: {args_str[:100]}...")

        # ---------- [NEW] 新增 fallback 解析（宽松匹配） ----------
        # 如果上面的严格模式都没匹配成功，尝试从内容中提取类似 "calculator({...})" 或 "calculator: {...}" 的格式
        fallback_patterns = [
            r'(\w+)\s*\(\s*(\{.*?\})\s*\)',          # 匹配 func({...})
            r'(\w+)[：:]\s*(\{.*\})',                # 匹配 func: {...}  或 func：{...}
            r'(\w+)\s*\(\s*([^)]*)\)',               # 匹配 func(key=value, ...)  —— 这种可能不是 JSON，先提取再尝试构造
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                name = match.group(1)
                args_candidate = match.group(2).strip()

                # ========== 【新增：防止误解析】 ==========
                # 只有提取的名称确实是已注册的工具时，才继续处理
                # 否则直接跳过，避免将普通文本（如 "我猜是 python(3.8)"）误判为工具调用
                if name not in registered_tool_names:
                    logger.debug(f"Fallback ignored '{name}' - not a registered tool.")
                    continue
                # ===========================================

                # 尝试补全为合法 JSON
                if not args_candidate.startswith('{'):
                    # 如果内容是 "key=value, key2=value2" 这种，尝试转为 JSON 对象
                    # 这里简单处理：如果它看起来像 JSON，直接解析；否则尝试用 parse_json 工具
                    pass
                # 优先尝试作为 JSON 解析
                try:
                    # 如果缺少花括号，补上
                    if not args_candidate.startswith('{'):
                        args_candidate = '{' + args_candidate + '}'
                    args = json.loads(args_candidate)
                    logger.info(f"Parsed tool call via fallback: {name} with {len(args)} args")
                    return {"name": name, "arguments": args}
                except json.JSONDecodeError:
                    # 如果还是失败，使用已有的 parse_json 宽松解析
                    args = parse_json(args_candidate)
                    if args:
                        logger.info(f"Parsed tool call via fallback (loose): {name} with {len(args)} args")
                        return {"name": name, "arguments": args}
                # 如果上述都失败，继续尝试下一个 fallback 模式
                continue


        # ---------- 原有结尾（保持不变） ----------
    
            logger.debug("No tool call pattern matched")

        # 新增：当所有解析尝试都失败时，记录前200字符供调试
        #logger.warning(f"无法解析的工具调用内容(前200字符): {content[:200]}")
        
        console.print(f"[dim]⚠️ 无法解析的工具调用内容(前200字符): {content[:200]}[/dim]")
        return None



    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """
        Extract a balanced JSON object from text by counting braces.

        Args:
            text: Text starting near a JSON object

        Returns:
            Extracted JSON string or None
        """
        # Find the first opening brace
        start = text.find('{')
        if start == -1:
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]

        logger.debug(f"Unbalanced braces (count={brace_count}), cannot extract JSON")
        return None
    


   
        
    def _execute_tool(
        self,
        tool_call: Dict,
        tool_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Any:

        """
        Execute a tool call
        
        Args:
            tool_call: Tool call information
            
        Returns:
            Tool execution result
        """


        start = time.perf_counter()
        tool_name = tool_call["name"]
        args = tool_call.get("arguments", {})


        # 原有白名单校验...
        if (tool_name != "use_skill" 
            and self._loaded_skill is not None 
            and self._skill_tool_whitelist is not None 
            and tool_name not in self._skill_tool_whitelist):
            elapsed = time.perf_counter() - start
            console.print(f"[dim]🚫 Tool BLOCKED: {tool_name} (not in skill whitelist) | elapsed={elapsed:.3f}s[/dim]")
            return f"错误：工具 '{tool_name}' 不在当前技能 '{self._loaded_skill.name}' 的允许列表中。"


        tool = next((t for t in self.tools if t["name"] == tool_name), None)
        if not tool:
            elapsed = time.perf_counter() - start
            console.print(f"[dim]❌ Tool NOT FOUND: {tool_name} | elapsed={elapsed:.3f}s[/dim]")
            return f"错误：找不到工具 '{tool_name}'"


        try:
            if tool_callback:
                tool_callback("start", tool_name, {"arguments": args})
            result = tool["executor"](**args)
            if tool_callback:
                tool_callback("end", tool_name, {"result": result})
            elapsed = time.perf_counter() - start
        

            # 关键：区分是否 use_skill
            if tool_name == "use_skill":
                console.print(f"[dim]🧠 SKILL TOOL CALL: {tool_name} args={str(args)[:80]} elapsed={elapsed:.3f}s[/dim]")
            else:
                console.print(f"[dim]🔧 TOOL CALL: {tool_name} args={str(args)[:80]} elapsed={elapsed:.3f}s[/dim]")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            console.print(f"[dim]💥 TOOL ERROR: {tool_name} | elapsed={elapsed:.3f}s | error={str(e)[:50]}[/dim]")
            return f"错误：执行工具 '{tool_name}' 时出错：{str(e)}"

    


    def _maybe_reflect(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply reflection if enabled and conversation has history."""
        if self.use_reflector and len(messages) > 1 and self.reflector:
            return self.reflector.apply_reflection(messages)
        return messages

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call LLM with messages
        
        Args:
            messages: Conversation messages
            
        Returns:
            LLM response content
        """
        start = time.perf_counter()  # <--- 计时开始

        try:
            logger.debug(f"Calling LLM with API key: {self.api_key[:6]}...")
            logger.debug(f"Base URL: {self.base_url or 'default OpenAI'}")
            logger.debug(f"Model: {self.model}")
            
            if not self.api_key:
                raise ValueError("API key is not set. Please check your environment variables.")
            
            # Apply reflection if enabled
            messages = self._maybe_reflect(messages)
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            content = response.choices[0].message.content
            elapsed = time.perf_counter() - start  # <--- 计算耗时

            # <--- 日志打印位置（成功时） --->
            console.print(f"[dim]🤖 LLM call ({self.model}) succeeded in {elapsed:.3f}s[/dim]")
            return content

        except Exception as e:
            elapsed = time.perf_counter() - start  # <--- 计算耗时（即使报错）
            logger.error(f"Error calling LLM: {str(e)}")

            # <--- 日志打印位置（失败时） --->
            console.print(f"[dim]💥 LLM call ({self.model}) failed in {elapsed:.3f}s: {str(e)[:60]}[/dim]")
            raise



    def _call_llm_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Call LLM with streaming, yielding tokens as they arrive.
        
        Args:
            messages: Conversation messages
            
        Yields:
            Token strings as they stream in
        """
        start = time.perf_counter()  # <--- 计时开始

        if not self.api_key:
            raise ValueError("API key is not set.")
        
        messages = self._maybe_reflect(messages)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        try:
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

            # <--- 正常流式结束，打印耗时（在循环外、try内部） --->
            elapsed = time.perf_counter() - start
            console.print(f"[dim]🤖 LLM stream ({self.model}) completed in {elapsed:.3f}s[/dim]")

        except Exception as e:
            # <--- 流式过程出错，打印失败耗时 --->
            elapsed = time.perf_counter() - start
            logger.error(f"Streaming error during iteration: {e}")
            console.print(f"[dim]💥 LLM stream ({self.model}) failed in {elapsed:.3f}s: {str(e)[:60]}[/dim]")
            raise




    @staticmethod
    def _summarize_messages(messages: List[Dict[str, str]], keep_last: int = 10) -> List[Dict[str, str]]:
        """
        Compress conversation history when it grows too long.
        
        Keeps the system prompt + a summary of old messages + the last N messages.
        This prevents token overflow in long-running sessions.
        
        Args:
            messages: Full message list
            keep_last: Number of recent messages to keep verbatim
            
        Returns:
            Compressed message list
        """
        if len(messages) <= keep_last + 2:  # system + enough messages
            return messages
        
        system = messages[0] if messages[0]["role"] == "system" else None
        start = 1 if system else 0
        old_messages = messages[start:-keep_last]
        recent = messages[-keep_last:]
        

        '''
        # Build a compact summary of old conversation
        summary_parts = []
        for m in old_messages:
            role = m.get("role", "")
            content = (m.get("content", "") or "")[:300]   # 从200增加到300
            if role == "user":
                summary_parts.append(f"User asked: {content}")
            elif role == "assistant":
                summary_parts.append(f"Assistant: {content}")
            elif role == "tool":
                summary_parts.append(f"Tool result: {content}")
        
        summary = "\n".join(summary_parts[-10:])  # keep last 10 entries in summary
        summary_msg = {
            "role": "user",
            "content": f"[Conversation summary - {len(old_messages)} earlier messages compressed]\n{summary}\n[End of summary. Continue from here.]"
        }
        '''

        # ========== 改进点 1：限制摘要条目数，防止摘要过长 ==========
        # 最多保留最近 15 条旧消息做摘要，避免压缩本身产生大量 token
        if len(old_messages) > 15:
            old_messages = old_messages[-15:]

        summary_parts = []
        for m in old_messages:
            role = m.get("role", "").upper()
            content = (m.get("content", "") or "")

            # ========== 改进点 2：用 smart_truncate 替代硬截断 ==========
            # smart_truncate 会尝试在完整句子边界处截断，避免切词
            truncated = smart_truncate(content, 300)

            # ========== 改进点 3：保留结构化角色标记 ==========
            if role == "USER":
                summary_parts.append(f"[用户] {truncated}")
            elif role == "ASSISTANT":
                summary_parts.append(f"[助手] {truncated}")
            elif role == "TOOL":
                summary_parts.append(f"[工具结果] {truncated}")
            else:
                summary_parts.append(f"[{role}] {truncated}")

        summary = "\n".join(summary_parts)

        # ========== 改进点 4：更清晰的压缩标记 ==========
        total_compressed = len(messages) - keep_last - (1 if system else 0)
        summary_msg = {
            "role": "user",
            "content": (
                f"[历史会话压缩] 已将较早的 {total_compressed} 条消息压缩为摘要：\n"
                f"{summary}\n"
                f"[压缩结束，请基于上述摘要和最近的对话继续]"
            )
        }

        #===改动结束
        
        result = []
        if system:
            result.append(system)
        result.append(summary_msg)
        result.extend(recent)
        return result



    def _check_dangerous(self, tool_call: Dict) -> bool:
        """
        Check if a tool call is potentially dangerous and needs confirmation.
        
        Returns True if the call is safe to proceed, False if user rejected.
        """
        if not self.confirm_dangerous:
            return True
        
        if tool_call["name"] != "bash":
            return True
        
        cmd = tool_call.get("arguments", {}).get("cmd", "")
        if not _DANGEROUS_RE.search(cmd):
            return True
        
        # Ask for confirmation
        if self.confirm_callback:
            return self.confirm_callback(cmd)
        
        # Default: stdin prompt
        try:
            answer = input(f"\n⚠️  Dangerous command detected: {cmd}\nAllow execution? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    # ------------------------------------------------------------------
    # Shared helpers for both run modes
    # ------------------------------------------------------------------

    def _compress_if_needed(self, messages, max_context_messages):
        """Compress conversation history when it exceeds the limit."""
        if len(messages) > max_context_messages:
            messages = self._summarize_messages(messages)
            logger.info(f"Compressed conversation to {len(messages)} messages")
        return messages

    def _safe_execute_tool(self, tool_call, tool_callback, status_callback, limit):
        """Execute a tool with safety check, status callbacks, and result truncation.
        
        Returns:
            (result_str, rejected): result_str is None if rejected.
        """
        if not self._check_dangerous(tool_call):
            return None, True
        
        if status_callback:
            status_callback(f"Executing tool: {tool_call['name']}...")
        
        logger.info(f"Executing tool: {tool_call['name']} with args: {tool_call['arguments']}")
        result = self._execute_tool(tool_call, tool_callback=tool_callback)
        return smart_truncate(str(result), limit), False
    


    

    def _build_dynamic_system_prompt(self) -> str:
        """根据当前加载的技能动态构建系统提示词"""

        base = self._loaded_skill.prompt if self._loaded_skill else self.system_prompt
        if self._loaded_skill:
            console.print(f"[dim]📄 Using SKILL PROMPT: {self._loaded_skill.name}[/dim]")
        else:
            console.print(f"[dim]📄 Using DEFAULT SYSTEM PROMPT (no skill loaded)[/dim]")

        return self._TEXT_MODE_PROMPT.format(
            base_prompt=base,
            tools_prompt=self._build_tools_prompt(),
        )



    #============新增两个方法，改进 run_with_tools 与 run_with_native_tools 大量重复 的问题

    def _init_run(self, query: str):
        """
        初始化每次运行的公共环境：重置技能状态、初始化消息列表、获取限制参数
        """
        self._reset_skill_state()
        logger.info(f"Starting query: {query[:50]}...")
        messages = [
            {"role": "system", "content": ""},  # 占位，每轮动态更新
            {"role": "user", "content": query}
        ]
        return messages, self._max_context_messages, self._tool_result_limit



    def _force_final_answer(self, messages: List[Dict], max_iterations: int) -> str:
        """
        当超出迭代次数时，强制LLM生成最终答案（两种模式完全一致的逻辑）
        """
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        messages.append({
            "role": "user",
            "content": "你已尝试多次工具调用，请根据所有已知信息，用中文给出最终答案，并以 FINAL_ANSWER: 开头。"
        })
        # 统一使用 _call_llm（自带重试），替代原生方法中直接调用 client.chat.completions.create
        final = self._call_llm(messages)
        if final.strip().startswith("FINAL_ANSWER:"):
            return final[len("FINAL_ANSWER:"):].strip()
        return final


    #========新增结束



    def run_with_tools(
        self,
        query: str,
        max_iterations: int = 10,
        tool_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:

         
       
        # ===== 替换初始化代码 =====
        messages, max_ctx, limit = self._init_run(query)


        # 使用 for 循环自动管理迭代次数
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            messages = self._compress_if_needed(messages, max_ctx)

            # ===== 动态更新 system prompt =====
            system_content = self._build_dynamic_system_prompt()
            messages[0] = {"role": "system", "content": system_content}

            if status_callback:
                status_callback(f"Thinking (Iteration {iteration + 1})...")

            # 获取模型响应
            if stream_callback:
                chunks = []
                for token in self._call_llm_stream(messages):
                    chunks.append(token)
                    stream_callback(token)
                response = "".join(chunks)
            else:
                response = self._call_llm(messages)

            # 检测 FINAL_ANSWER
            if response.strip().startswith("FINAL_ANSWER:"):
                return response[len("FINAL_ANSWER:"):].strip()

            messages.append({"role": "assistant", "content": response})

            # 解析工具调用
            tool_call = self._parse_tool_call(response)

            # 无工具调用
            if not tool_call:
                # 启发式判断：若响应足够长且无“未完成”暗示，视为完成
                if len(response) > 100 and not any(kw in response for kw in ["需要查询", "需要搜索", "请稍等", "我会查找"]):
                    logger.info("No tool call but response seems complete, returning.")
                    return response
                # 否则引导
                messages.append({
                    "role": "user",
                    "content": "当前回复没有工具调用且不完整。请调用工具获取信息，或如果已足够，请以 FINAL_ANSWER: 开头给出最终回答。"
                })
                continue  # 下一轮

            # 处理 use_skill
            if tool_call["name"] == "use_skill":
                result_str = self._use_skill_handler(**tool_call["arguments"])
                messages.append({
                    "role": "user",
                    "content": f"工具 '{tool_call['name']}' 执行结果：{result_str}"
                })
                continue  # 下一轮，此时 _loaded_skill 已更新

            # 执行其他工具
            result_str, rejected = self._safe_execute_tool(tool_call, tool_callback, status_callback, limit)

            if rejected:
                feedback = f"用户拒绝了工具 '{tool_call['name']}'，请建议安全的替代方案或用中文回答。"
            elif isinstance(result_str, str) and ("Error" in result_str or "Exception" in result_str):
                feedback = f"工具 '{tool_call['name']}' 出错：{result_str}\n请解释错误并给出解决方案。"
            else:
                feedback = f"工具 '{tool_call['name']}' 结果：{result_str}\n请继续用中文回答，完成时以 FINAL_ANSWER: 开头。"

            messages.append({"role": "user", "content": feedback})

        # 超出迭代次数，强制生成最终答案
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        messages.append({
            "role": "user",
            "content": "你已尝试多次工具调用，请根据所有已知信息，用中文给出最终答案，并以 FINAL_ANSWER: 开头。"
        })
        final = self._call_llm(messages)
        if final.strip().startswith("FINAL_ANSWER:"):
            return final[len("FINAL_ANSWER:"):].strip()
        return final




    def run_with_native_tools(
        self,
        query: str,
        max_iterations: int = 10,
        tool_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> str:

        
        # ===== 替换初始化代码 =====
        messages, max_ctx, limit = self._init_run(query)



        for iteration in range(max_iterations):
            messages = self._compress_if_needed(messages, max_ctx)

            # 动态更新 system
            system_content = self._build_dynamic_system_prompt()
            messages[0] = {"role": "system", "content": system_content}

            if status_callback:
                status_callback(f"Thinking (Iteration {iteration + 1})...")

            # [CHANGED] 在这里动态构建 tool_schemas，基于当前过滤后的工具列表
            filtered_tools = self._get_filtered_tools()
            tool_schemas = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            } for t in filtered_tools]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tool_schemas if tool_schemas else None,
                )
            except Exception as e:
                logger.error(f"Native FC LLM call failed: {e}")
                raise

            msg = response.choices[0].message

            # 检测 FINAL_ANSWER
            if msg.content and msg.content.strip().startswith("FINAL_ANSWER:"):
                return msg.content[len("FINAL_ANSWER:"):].strip()

            # 无工具调用，直接返回内容（可能已经回答）
            if not msg.tool_calls:
                return msg.content or ""

            messages.append(msg)

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = parse_json(tc.function.arguments) or {}

                if tool_name == "use_skill":
                    result_str = self._use_skill_handler(**arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                    # 注意：_loaded_skill 和 _skill_tool_whitelist 已更新，
                    # 下一轮循环会使用新的过滤列表
                    continue

                tool_call_info = {"name": tool_name, "arguments": arguments}
                result_str, rejected = self._safe_execute_tool(
                    tool_call_info, tool_callback, status_callback, limit
                )
                content = "Execution rejected by user. Suggest a safer alternative." if rejected else result_str
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })


        '''
        # 超出迭代次数，强制最终回答
        logger.warning(f"Native FC reached max iterations ({max_iterations})")
        messages.append({
            "role": "user",
            "content": "请根据已有信息，用中文给出最终答案，并以 FINAL_ANSWER: 开头。"
        })
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        final = final_response.choices[0].message.content
        if final and final.strip().startswith("FINAL_ANSWER:"):
            return final[len("FINAL_ANSWER:"):].strip()
        return final or ""
        '''

        # ===== 替换结尾的强制回答代码（现在统一使用 _call_llm） =====
        return self._force_final_answer(messages, max_iterations)




    def run(self, query: str, max_iterations: int = 15, mode: str = "native") -> str:     #mode默认模式从text改为native，max_iterations从10改到15，.env同步改
        """
        Execute the Agent with specified tool calling mode.
        
        Args:
            query: User query text
            max_iterations: Maximum number of iterations
            mode: Tool calling mode — "text" (default, transparent parsing) 
                  or "native" (OpenAI function calling)
            
        Returns:
            Agent response text
        """
        if mode == "native":
            return self.run_with_native_tools(query, max_iterations)
        return self.run_with_tools(query, max_iterations)
