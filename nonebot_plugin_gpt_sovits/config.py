"""插件配置。唯一配置源是 config/gpt_sovits.toml（分节 + 注释），键名 = pydantic 字段名：

    [api]
    base_url = "http://127.0.0.1:9880"

    [[emotions]]              # 情绪按数组顺序编号，群里 -e 序号选（改情绪即时生效）
    name = "日常"
    [[emotions.sentences]]
    text = "嗨，我是西格莉卡！"   # 参考音频的原话，必须与音频内容逐字一致
    path = "D:/.../xxx.wav"     # GPT-SoVITS 进程读的本机路径
    language = "all_zh"

分节只是视觉分组，[api] 装平成字段；规范见 docs/config-spec.md。
TOML 缺失时按代码内模板生成空骨架（_ensure_toml，xixi_repeat 同款）。
解析走 stdlib tomllib（嵌套结构拿纯 dict 好收窄），写模板走 tomlkit（保注释）。
"""

import tomllib
from pathlib import Path
from typing import TypedDict

import tomlkit
from nonebot import logger
from pydantic import BaseModel, Field
from tomlkit.items import Item

from .models import Emotion, Sentence, dict_language

# 项目根 config/gpt_sovits.toml。bot 进程 cwd 即项目根，同 xixi_repeat.toml 的取法。
GPT_SOVITS_TOML = Path.cwd() / "config" / "gpt_sovits.toml"

_API_KEYS = {"base_url", "api_v2", "command", "convert_to_silk"}

# api_v2 透传推理参数的合法键（generate_v2 只转发这些；写错键在这里当场报错更有用）
_ARG_KEYS = {
    "aux_ref_audio_paths", "top_k", "top_p", "temperature", "text_split_method",
    "batch_size", "batch_threshold", "split_bucket", "speed_factor",
    "fragment_interval", "streaming_mode", "seed", "parallel_infer",
    "repetition_penalty",
}

_KNOWN_SECTIONS = {"api", "args", "emotions"}


class _ApiSection(TypedDict, total=False):
    base_url: str
    api_v2: bool
    command: str
    convert_to_silk: bool


class Config(BaseModel):
    # ── api ───────────────────────────────────────────
    base_url: str = "http://127.0.0.1:9880"  # GPT-SoVITS api_v2 服务地址
    api_v2: bool = True                      # False 走旧版 api.py（不推荐）
    command: str = "tts"                     # 指令名（改动需重启：matcher 定义时消费）
    convert_to_silk: bool = False            # True 转 silk（需 ffmpeg + cli，Windows 未配）
    args: dict[str, object] = Field(default_factory=dict)  # api_v2 透传参数，键见 _ARG_KEYS

    # ── 情绪 ──────────────────────────────────────────
    emotions: list[Emotion] = Field(default_factory=list)


def _expect(section: dict[str, object], key: str, where: str, typ: type[str] | type[bool]) -> object:
    """取出键并校验标量类型；TOML 解析器已保证顶层值形态，这里只挡放错的值。"""
    value = section[key]
    if not isinstance(value, typ):
        raise ValueError(f"{where}: {key} 必须是 {typ.__name__}")
    return value


def _parse_api(table: object, path: Path) -> _ApiSection:
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [api] 必须是表")
    bad = set(table) - _API_KEYS
    if bad:
        raise ValueError(f"{path}: [api] 含未知键 {sorted(bad)}，合法键 {_API_KEYS}")
    api: _ApiSection = {}
    if "base_url" in table:
        api["base_url"] = str(_expect(table, "base_url", f"{path} [api]", str))
    if "api_v2" in table:
        api["api_v2"] = bool(_expect(table, "api_v2", f"{path} [api]", bool))
    if "command" in table:
        api["command"] = str(_expect(table, "command", f"{path} [api]", str))
    if "convert_to_silk" in table:
        api["convert_to_silk"] = bool(_expect(table, "convert_to_silk", f"{path} [api]", bool))
    return api


def _parse_args(table: object, path: Path) -> dict[str, object]:
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [args] 必须是表")
    bad = set(table) - _ARG_KEYS
    if bad:
        raise ValueError(f"{path}: [args] 含非法键 {sorted(bad)}，合法键 {_ARG_KEYS}")
    return dict(table)


def _parse_sentences(rows: object, where: str) -> list[Sentence]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{where}: 需要至少一个 [[{where.rsplit(' ', 1)[-1]}.sentences]]")
    sentences: list[Sentence] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{where}: sentences 项必须是 [[emotions.sentences]] 表")
        missing = {"text", "path", "language"} - set(row)
        if missing:
            raise ValueError(f"{where}: 句子缺少键 {sorted(missing)}")
        lang = str(row["language"])
        if lang not in dict_language:
            raise ValueError(f"{where}: language {lang!r} 不合法，取值见注释")
        sentences.append(Sentence(
            text=str(row["text"]), path=str(row["path"]), language=dict_language[lang],
        ))
    return sentences


def _parse_emotions(rows: object, path: Path) -> list[Emotion]:
    if not isinstance(rows, list):
        raise ValueError(f"{path}: [[emotions]] 必须是表数组")
    emotions: list[Emotion] = []
    for idx, row in enumerate(rows):
        where = f"{path} [[emotions]] #{idx}"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: 必须是表")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: 缺少 name")
        emotions.append(Emotion(name=name.strip(), sentences=_parse_sentences(row.get("sentences"), where)))
    return emotions


def _read_toml(path: Path) -> Config:
    """读 TOML 装平成 Config。未知键、放错分节当场报错——
    静默回落默认值会掩盖拼写错误。"""
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    bad = set(doc) - _KNOWN_SECTIONS
    if bad:
        raise ValueError(f"{path}: 未知的配置分节 {sorted(bad)}，见 docs/config-spec.md")
    args = _parse_args(doc["args"], path) if "args" in doc else {}
    emotions = _parse_emotions(doc["emotions"], path) if "emotions" in doc else []
    api = _parse_api(doc["api"], path) if "api" in doc else {}
    return Config(**api, args=args, emotions=emotions)


_TOML_HEADER = """# GPT-SoVITS 语音合成（nonebot-plugin-gpt-sovits）配置
# 键名 = 插件 config.py 的 pydantic 字段名，分节只是视觉分组（docs/config-spec.md）。
# [api]/[args] 改完重启生效；[[emotions]] 每次指令现读，改完即生效。
# 情绪按数组顺序编号（第一个 = -e 0），群里 tts 文本 -e 序号 触发。
# language 取值：all_zh / en / all_ja / zh / ja / auto（也可写 中文/英文/日文 等中文名）。

[api]

[args]

[[emotions]]
name = ""

[[emotions.sentences]]
text = ""
path = ""
language = "all_zh"
"""

# 生成模板的注释唯一事实源（键 → 行内注释）。
_TOML_COMMENTS: dict[str, str] = {
    "base_url": "GPT-SoVITS api_v2 服务地址（本机默认 9880）",
    "api_v2": "True=api_v2.py（v2/v2Pro/v3/v4/v5 权重），False=旧版 api.py",
    "command": "指令名，群里 `tts 文本 -e 序号` 触发；改动需重启",
    "convert_to_silk": "True=转 silk 发送（需 ffmpeg + cli 二进制，Windows 未配，保持 False）",
}


def _ensure_toml() -> None:
    """TOML 不存在时按代码内模板生成骨架；已存在则不动主人手改的内容。"""
    if GPT_SOVITS_TOML.exists():
        return
    GPT_SOVITS_TOML.parent.mkdir(parents=True, exist_ok=True)
    doc = tomlkit.parse(_TOML_HEADER)
    defaults = Config().model_dump()
    api = doc["api"]
    for key in ("base_url", "api_v2", "command", "convert_to_silk"):
        item: Item = tomlkit.item(defaults[key])
        item.comment(_TOML_COMMENTS[key])
        api[key] = item
    GPT_SOVITS_TOML.write_text(tomlkit.dumps(doc), encoding="utf-8")
    logger.info(f"[gpt_sovits] 已生成配置骨架 {GPT_SOVITS_TOML}（[[emotions]] 需手填）")


def load_config() -> Config:
    """从 gpt_sovits.toml 现读插件配置（AGENTS.md 红线 1.8：禁止模块级缓存）。"""
    _ensure_toml()
    return _read_toml(GPT_SOVITS_TOML)
