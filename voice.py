"""概率语音回复：模型回复的第一条气泡按概率换成西西的语音。

流程：_send_bubbles(allow_voice=True) → outbound.transform_reply → 本模块
maybe_voice。任何一步不满足（没抽中/冷却中/超长/离线/合成失败）都返回
None，正文照常发文字——语音挂了不能殃及说话本身。

会话状态（上次语音时间）存进程内存：重启清零可接受（概率闸门不是账本）。
"""
import random
import re
import time

from nonebot.adapters.onebot.v11 import MessageEvent, MessageSegment
from nonebot.log import logger

from . import tts_client
from .config import Config

# 语音不适合念的东西：emoji/颜文字/序号列表/代码样文本。命中即回落文字。
_SKIP_PATTERNS = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\U0001F900-\U0001F9FF]"  # emoji 区
    r"|```|https?://|www\."
    r"|^[0-9]{1,2}[.、)）]"    # 列表首行
)
# 念之前清掉：残留 markdown 强调/井号头/多余空白
_CLEAN_PATTERNS = re.compile(r"[*#`~]{1,3}|^\s*[>》]\s*")

_last_voice_at: dict[str, float] = {}


def _session_key(event: MessageEvent) -> str:
    gid = getattr(event, "group_id", None)
    return f"group_{gid}" if gid else f"private_{event.user_id}"


def pick_ref(cfg: Config) -> tuple[str, str, str] | None:
    """从情绪表随机挑一条参考（语音用 0 号「日常」最稳，短句优先）。"""
    if not cfg.emotions:
        return None
    emotion = cfg.emotions[0]
    sentence = random.choice(emotion.sentences)
    return sentence.path, sentence.text, sentence.language


def eligible(cfg: Config, event: MessageEvent, text: str) -> bool:
    """概率闸门：开关/白名单/长度/冷却/概率，全过才值得去合成。"""
    if not cfg.voice_enabled:
        return False
    if cfg.voice_groups and _session_key(event) not in cfg.voice_groups:
        return False
    cleaned = _CLEAN_PATTERNS.sub("", text).strip()
    if not cfg.voice_min_chars <= len(cleaned) <= cfg.voice_max_chars:
        return False
    if _SKIP_PATTERNS.search(text):
        return False
    now = time.monotonic()
    if now - _last_voice_at.get(_session_key(event), -1e9) < cfg.voice_cooldown:
        return False
    return random.random() < cfg.voice_probability


async def maybe_voice(event: MessageEvent, text: str) -> MessageSegment | None:
    """回复变换器入口：抽中且合成成功 → record 段；否则 None（发文字）。"""
    from .config import load_config
    cfg = load_config()
    if not eligible(cfg, event, text):
        return None
    ref = pick_ref(cfg)
    if ref is None:
        return None
    spoken = _CLEAN_PATTERNS.sub("", text).strip()
    wav = await tts_client.synthesize(cfg, spoken, *ref)
    if wav is None:
        return None
    _last_voice_at[_session_key(event)] = time.monotonic()
    logger.info(f"[gpt_sovits] 概率语音命中（{_session_key(event)}，{len(spoken)}字）")
    return MessageSegment.record(wav)
