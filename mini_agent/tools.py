"""工具层：定义 Agent 可以调用的所有能力。

一个工具 = 一个普通 Python 函数 + 它的 docstring（写给 LLM 看的说明书）。
LLM 本身不执行任何代码，它只输出「我想调用 calculate，参数是 {...}」这样的
结构化意图，真正执行的是本文件里的 Python。所以工具的安全边界完全由我们掌控。

本文件在安全上做了两处示范（面试常问）：
1. calculate 用 ast 白名单做安全求值，而不是 eval()——
   直接 eval 模型给的字符串等于开放任意代码执行；
2. 文件类工具被沙箱限制在项目目录内，路径先 resolve 成绝对路径再校验前缀，
   防止 "../" 或符号链接逃逸到系统目录。
"""

from __future__ import annotations

import ast
import datetime
import inspect
import operator
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

# 文件类工具的沙箱根目录：只允许访问本项目内的文件
SANDBOX_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 工具注册机制
# ---------------------------------------------------------------------------


class Tool:
    """把「函数 + docstring 描述」打包成一个可被 LLM 调用的工具。

    参数的 JSON Schema 从函数签名自动推导：类型注解映射到 JSON 类型，
    有默认值的参数视为可选。函数签名是唯一的真相来源，
    不会出现「schema 和函数对不上」的经典 bug。
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = func.__name__
        self.description = inspect.cleandoc(func.__doc__ or "").strip()

    @property
    def parameters(self) -> dict:
        schema: dict = {"type": "object", "properties": {}, "required": []}
        for name, param in inspect.signature(self.func).parameters.items():
            schema["properties"][name] = {"type": _annotation_to_json_type(param.annotation)}
            if param.default is inspect.Parameter.empty:
                schema["required"].append(name)
        return schema

    def to_openai_schema(self) -> dict:
        """转成 OpenAI function calling 的标准 tools 格式（全行业通用协议）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> str:
        """执行工具并把结果转成字符串——LLM 只能消费字符串。"""
        return str(self.func(**kwargs))


def _annotation_to_json_type(annotation: Any) -> str:
    """把 Python 类型注解映射到 JSON Schema 类型。

    开了 `from __future__ import annotations` 之后，inspect 拿到的注解是
    字符串（比如 "str"、"Optional[str]"）而不是类型对象，两种都要处理。
    """
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    name = name.replace("Optional[", "").rstrip("]").strip()
    return {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }.get(name, "string")


# ---------------------------------------------------------------------------
# 工具 1：安全计算器
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> Any:
    """递归求值，只放行白名单里的 AST 节点，其余一律拒绝。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        if isinstance(node.op, ast.Pow):
            exponent = _safe_eval(node.right)
            if abs(exponent) > 1000:
                raise ValueError("指数过大，拒绝计算（防资源耗尽攻击）")
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"表达式中包含不支持的元素: {type(node).__name__}")


def calculate(expression: str) -> str:
    """计算一个算术表达式。支持 + - * / // % ** 和括号，例如 "2+3*4" 或 "(1+2)**10"。
    任何需要精确数值计算的场景都必须调用本工具，不要用心算，心算很容易出错。"""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# 工具 2：当前时间
# ---------------------------------------------------------------------------

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def get_current_time(timezone: Optional[str] = None) -> str:
    """获取当前的日期和时间。参数 timezone 是 IANA 时区名，如 "Asia/Shanghai"、
    "America/New_York"；不传则用系统本地时区。LLM 的训练数据有截止日期，
    它自己不可能知道现在几点，这正是需要工具的典型场景。"""
    tz = None
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except Exception as e:
            # from e 保留原始异常链，方便排查（而不是吞掉换成新异常）
            raise ValueError(f"无法识别的时区: {timezone}，请用 IANA 名称，如 Asia/Shanghai") from e
    now = datetime.datetime.now(tz)
    text = now.strftime("%Y-%m-%d %H:%M:%S") + f" {_WEEKDAYS[now.weekday()]}"
    return text + (f" ({timezone})" if timezone else "")


# ---------------------------------------------------------------------------
# 工具 3/4：沙箱化的文件访问
# ---------------------------------------------------------------------------


def _resolve_in_sandbox(path: str) -> Path:
    """把模型给的路径限制在项目沙箱内：先解析成绝对路径，再校验是否越界。"""
    p = Path(path)
    if not p.is_absolute():
        p = SANDBOX_ROOT / p
    p = p.resolve()
    if not p.is_relative_to(SANDBOX_ROOT):
        raise PermissionError(f"拒绝访问沙箱外的路径: {path}")
    return p


def list_files(path: str = ".") -> str:
    """列出某个目录下的文件和子目录。path 只能是项目内的路径，默认为项目根目录。"""
    root = _resolve_in_sandbox(path)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "(空目录)"
    return "\n".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in entries)


def read_file(path: str) -> str:
    """读取项目内一个文本文件的内容。只能读项目目录内的文件，大小不超过 100KB。"""
    target = _resolve_in_sandbox(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if target.stat().st_size > 100 * 1024:
        raise ValueError(f"文件太大（>100KB），拒绝读取: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

ALL_TOOLS: list = [Tool(calculate), Tool(get_current_time), Tool(list_files), Tool(read_file)]


def get_tool(name: str) -> Optional[Tool]:
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def tools_schemas() -> list:
    """输出全部工具的 OpenAI schema，调用 LLM 时随请求带上。"""
    return [t.to_openai_schema() for t in ALL_TOOLS]
