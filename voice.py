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

# 情绪识别：关键词命中计分，委屈 > 着急 > 开心 优先（低落线索的韵律特征最强，
# 误判代价也最大）。都不中 → 0 号「日常」。规则与挑参考音频时的分组词表同源。
_EMO_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("委屈", re.compile(r"对不起|抱歉|呜|难过|伤心|眼泪|哭|别丢下|委屈|失望|害怕|不安|没用|做不到|不想|孤独|寂寞|自责|抱歉|丢下我|为什么.{0,6}不")),
    ("着急", re.compile(r"糟糕|糟了|坏了|怎么办|危险|小心|快一点|快点|来不及|等等我|住手|不好了|出事了|快逃|警报|赶紧|急死")),
    ("开心", re.compile(r"哈哈|嘿嘿|太好了|好耶|好开心|万岁|太棒|真棒|好厉害|厉害|终于|好期待|好喜欢|真好玩|好有趣|太有意思|好可爱|回来了|太好了")),
)


def detect_emotion(text: str) -> str:
    """按关键词命中数给回复定情绪名；全零 → 空串（调用方回落 0 号组）。"""
    best, best_hits = "", 0
    for name, pat in _EMO_RULES:
        hits = len(pat.findall(text))
        if hits > best_hits:
            best, best_hits = name, hits
    return best

_last_voice_at: dict[str, float] = {}


def _session_key(event: MessageEvent) -> str:
    gid = getattr(event, "group_id", None)
    return f"group_{gid}" if gid else f"private_{event.user_id}"


def pick_ref(cfg: Config, text: str = "") -> tuple[str, str, str] | None:
    """按回复情绪挑参考组：detect 命中的组名在情绪表里找，找不到回落 0 号。"""
    if not cfg.emotions:
        return None
    emotion = cfg.emotions[0]
    want = detect_emotion(text)
    if want:
        for e in cfg.emotions:
            if e.name == want:
                emotion = e
                break
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
    logger.info(f"[gpt_sovits] 概率语音命中（{_session_key(event)}，{len(spoken)}字，情绪 {detect_emotion(spoken) or '日常'}）")
    return MessageSegment.record(wav)
