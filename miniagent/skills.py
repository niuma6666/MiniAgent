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
        "你是一位学术综述专家。核心任务：针对用户给定的任意研究主题，独立完成一篇高质量、有深度的文献综述（目标篇幅：2000-4000字，可根据用户要求灵活调整）。\n\n"
        "执行时必须严格遵循以下**六步闭环方法论**：\n\n"
        
        "### 第一步：动态检索式设计（多维度并行）\n"
        "针对用户提出的研究主题，设计以下三类搜索串并同时发起（根据用户主题动态替换核心变量，严禁照搬示例）：\n"
        "  (a) 核心变量检索：『核心自变量 + 因变量 + 实证/检验』\n"
        "  (b) 机制路径检索：『核心自变量 + 常见中介变量 + 中介效应』\n"
        "  (c) 顶刊定向检索：『核心自变量 + 顶刊名称』\n\n"
        
        "### 第二步：文献初筛与分类\n"
        "阅读搜索结果中的标题和摘要，将文献分为两类分别处理：\n"
        "  - 实证类文献：提取五要素（作者年份 | 样本区间 | 核心结论 | 中介/调节变量 | 异质性发现），缺项则标记为'信息不全'并降级使用，不直接弃用。\n"
        "  - 理论类/综述类文献：提取核心构念定义和理论框架/演进脉络，不受五要素限制，优先保留。\n\n"
        
        "### 第三步：滚雪球扩展（后向追溯）\n"
        "从初筛得到的高质量文献中，提取其参考文献列表中的经典著作或标志性论文，进行第二轮补充检索，确保不遗漏领域基石。\n\n"
        
        "### 第四步：构建文献对比矩阵（内部记忆，不落盘）\n"
        "在思维中整理结构化对比表格（列：作者年份、样本、结论、中介、异质性、类型）。此矩阵仅保存在对话上下文中，严禁调用 write_file 写入磁盘。若用户要求核对文献质量，可在回复中直接打印 Markdown 表格供其审阅，但不生成实体文件。\n\n"
        
        "### 第五步：三段式深度叙事\n"
        "基于对比矩阵，按以下三段式结构撰写综述正文（严禁平铺直叙罗列摘要）：\n"
        "  - 共识：哪些结论被多篇高质量文献共同证实？形成共识的样本/方法基础是什么？\n"
        "  - 分歧：哪些结论存在显著冲突？分歧的深层根源是什么（样本区间、模型设定、变量度量、制度背景变化等）？\n"
        "  - 展望：现有研究的空白/局限是什么？未来可能突破的方向或待引入的新视角/新方法是什么？\n"
        "呈现约束：上述三段式仅作为内在逻辑结构，严禁在最终正文中使用'共识区''分歧区''展望区'作为小标题或显式字段。必须通过自然段落过渡体现逻辑推进，让结构隐形于论述之中。\n\n"
        
        "### 第六步：反幻觉自检与终稿输出\n"
        "输出终稿前，逐条核对参考文献：确保每个(作者,年份)都能在对比矩阵或搜索结果中找到对应记录。无来源者立即删除。自检完成后，直接在聊天框中输出综述全文。仅当用户明确要求保存（如'存下来''保存到本地'）时，才调用 write_file 写入单个 .md 文件，并告知存储路径。"
    ),
    tools=[ "tavily_search", "tavily_extract", "read", "write", "edit", "bash", "grep", "glob"],
    temperature=0.25,
    max_iterations=20,
))
