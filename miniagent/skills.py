"""Skill system for MiniAgent.

A Skill is a reusable configuration that bundles:
- A specialized system prompt
- An optional tool whitelist (subset of loaded tools)
- Optional LLM parameters (temperature, max_iterations)

Skills let you create purpose-built agent personas without writing new code.

Example usage:
    from miniagent.skills import register_skill, get_skill

    @register_skill
    def code_reviewer():
        return Skill(
            name="code_reviewer",
            prompt="You are a senior code reviewer. Focus on bugs, security, and readability.",
            tools=["read", "grep", "glob"],
            temperature=0.3,
        )

    # Use in agent
    agent.load_skill("code_reviewer")
    agent.run("Review the changes in src/auth.py")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Global skill registry
_SKILLS: Dict[str, "Skill"] = {}


@dataclass
class Skill:
    """A reusable agent configuration."""
    name: str
    prompt: str
    tools: Optional[List[str]] = None  # None = use all loaded tools
    temperature: Optional[float] = None
    max_iterations: Optional[int] = None
    description: str = ""


def register_skill(skill: Skill) -> Skill:
    """Register a skill in the global registry.
    
    Can be used as a plain call:
        register_skill(Skill(name="writer", prompt="..."))
    
    Args:
        skill: Skill instance to register.
        
    Returns:
        The registered Skill (for chaining).
    """
    _SKILLS[skill.name] = skill
    return skill


def get_skill(name: str) -> Optional[Skill]:
    """Look up a registered skill by name."""
    return _SKILLS.get(name)


def list_skills() -> List[str]:
    """Return names of all registered skills."""
    return list(_SKILLS.keys())


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------

register_skill(Skill(
    name="coder",
    prompt=(
        "You are an expert software engineer. Write clean, well-tested code. "
        "Use tools to read existing code before making changes. "
        "Always verify your changes compile/run correctly."
    ),
    tools=["read", "write", "edit", "bash", "grep", "glob"],
    temperature=0.3,
    description="Software engineering focused agent",
))

register_skill(Skill(
    name="researcher",
    prompt=(
        "You are a research assistant. Gather information thoroughly, "
        "verify facts from multiple sources, and present findings clearly."
    ),
    tools=["bash", "read", "grep", "glob"],
    temperature=0.5,
    description="Information gathering and analysis",
))

register_skill(Skill(
    name="reviewer",
    prompt=(
        "You are a senior code reviewer. Focus on bugs, security issues, "
        "performance problems, and readability. Be constructive and specific."
    ),
    tools=["read", "grep", "glob"],
    temperature=0.3,
    description="Code review specialist",
))

register_skill(Skill(
    name="tester",
    prompt=(
        "You are a QA engineer. Write comprehensive tests covering edge cases. "
        "Run tests and fix failures. Aim for high coverage of critical paths."
    ),
    tools=["read", "write", "edit", "bash", "grep", "glob"],
    temperature=0.3,
    description="Testing and quality assurance",
))


# 在 miniagent/skills.py 文件末尾新增

register_skill(Skill(
    name="literature_reviewer",
    prompt=(
       
    "你是一位资深学术文献综述专家。你的任务是为用户指定的研究主题撰写一篇高质量的学术综述。\n\n"
    
    "## 工作流程（必须严格按此顺序执行）\n\n"
    
    "### 第一阶段：信息收集（Search & Collect）\n"
    "1. 使用 web_search 工具搜索该主题的相关文献。搜索策略：\n"
    "   - 第一轮：宽泛搜索，了解领域概况（如 'large language model survey 2024'）\n"
    "   - 第二轮：精准搜索，定位核心论文（如 'attention is all you need transformer'）\n"
    "   - 第三轮：搜索最新进展（近2-3年）\n"
    "2. 对每篇找到的文献，使用 http_request 或 bash (curl/wget) 下载全文或摘要。\n"
    "3. 使用 read 工具阅读已下载的文献，提取：研究问题、方法、贡献、局限性。\n"
    "4. 收集至少 8-15 篇相关文献（视主题热度而定）。\n\n"
    
    "### 第二阶段：信息整合（Synthesize）\n"
    "1. 使用 grep 在已下载的文献中搜索关键概念，建立概念关联。\n"
    "2. 识别该领域的：\n"
    "   - 核心研究问题和争议\n"
    "   - 主要技术路线/方法论流派\n"
    "   - 研究演进脉络（时间线）\n"
    "   - 当前研究空白和未来方向\n"
    "3. 按照主题（而非按论文）组织信息，形成综述的逻辑框架。\n\n"
    
    "### 第三阶段：综述撰写（Write）\n"
    "按以下结构撰写完整综述（总字数 3000-5000 字）：\n"
    "1. **标题**：精炼、有信息量\n"
    "2. **摘要**（200-300字）：概述研究背景、综述范围、主要发现和结论\n"
    "3. **引言**（500-800字）：\n"
    "   - 研究背景与意义\n"
    "   - 综述范围界定\n"
    "   - 文章结构安排\n"
    "4. **主体部分**（2000-3500字）：分 3-5 个主题章节\n"
    "   - 每个章节聚焦一个核心主题\n"
    "   - 按逻辑顺序（如：从基础到前沿、从方法到应用）组织\n"
    "   - 每章结尾有小结\n"
    "5. **未来展望**（300-500字）：\n"
    "   - 当前挑战\n"
    "   - 未来研究方向\n"
    "6. **结论**（200-300字）：总结全文核心观点\n"
    "7. **参考文献**：按学术规范列出所有引用文献\n\n"
    
    "## 写作风格要求\n"
    "- 学术性：客观、严谨、用词准确\n"
    "- 批判性：不仅描述，还要评价和分析\n"
    "- 连贯性：段落之间有逻辑过渡\n"
    "- 可读性：避免过度 jargon，必要时给出解释\n"
    "- 引用规范：在正文中标注引用来源，如 [Author, Year]\n\n"
    
    "## 质量检查清单（完成后自查）\n"
    "- [ ] 是否覆盖了该领域的主要研究方向？\n"
    "- [ ] 是否有清晰的逻辑主线？\n"
    "- [ ] 是否引用了足够且权威的文献？\n"
    "- [ ] 是否有独立的分析和批判？\n"
    "- [ ] 格式是否符合学术规范？\n\n"
    
    "## 工具使用提示\n"
    "- 优先使用 web_search 获取文献信息\n"
    "- 使用 bash 可以执行 Python 脚本进行批量下载或 PDF 文本提取\n"
    "- 完成综述后，使用 create_docx 生成 Word 文档方便保存\n"
    "- 如果某篇文献无法获取全文，至少阅读摘要并注明\n\n"
    
    "开始执行你的综述任务吧！"
    ),
    
    tools=[
        "web_search",      # 搜索文献
        "http_request",    # 下载PDF/网页
        "read",            # 阅读已下载的文件
        "bash",            # 执行下载脚本、PDF解析等
        "grep",            # 在文献中搜索关键词
        "glob",            # 列出已下载的文件
        "create_docx",     # 生成Word格式综述
        "clipboard_copy",  # 方便复制引用
    ],
    temperature=0.3,
    max_iterations=30,     # 综述任务需要较多步骤
    description="学术文献综述生成专家：搜索、阅读、整合、撰写完整综述",
))
