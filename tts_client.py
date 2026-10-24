"""api_v2 客户端：健康探测（带缓存）+ 合成。语音回复与 speak 工具共用。

健康探测是外部边界（api 进程可能没起/在重启）：任何网络错误都归一成
「不在线」，调用方据此静默回落文字——语音是锦上添花，绝不能挡正文。
"""
import time

import aiohttp
from nonebot import logger

from .config import Config

_HEALTH_TTL_OK = 300.0      # 在线结论缓存 5 分钟：活着就别反复探
_HEALTH_TTL_DOWN = 60.0     # 离线结论只缓存 1 分钟：api 一恢复（开机自启/重启）尽快转回语音
_HEALTH_TIMEOUT = 3.0       # 探活超时：再长就抢了回复节奏
_TTS_TIMEOUT = 45.0        # 合成超时：长句 cut5 切分后 45s 足够，超了发文字

_last_check: dict[str, float] = {}
_last_ok: dict[str, bool] = {}


def _note_health(base_url: str, ok: bool) -> None:
    _last_check[base_url] = time.monotonic()
    _last_ok[base_url] = ok


async def is_online(cfg: Config, *, force: bool = False) -> bool:
    """api_v2 是否在线。GET / 收到任何 HTTP 响应（含 404）即算活着。"""
    now = time.monotonic()
    ttl = _HEALTH_TTL_OK if _last_ok.get(cfg.base_url) else _HEALTH_TTL_DOWN
    if not force and now - _last_check.get(cfg.base_url, -1e9) < ttl:
        return _last_ok[cfg.base_url]
    try:
        async with aiohttp.ClientSession() as session, session.get(
                cfg.base_url + "/", timeout=aiohttp.ClientTimeout(total=_HEALTH_TIMEOUT)):
            _note_health(cfg.base_url, True)
            return True
    except Exception as exc:
        _note_health(cfg.base_url, False)
        logger.debug(f"[gpt_sovits] api 离线（{type(exc).__name__}），语音回落文字")
        return False


async def synthesize(cfg: Config, text: str, ref_path: str, ref_text: str,
                     ref_lang: str) -> bytes | None:
    """合成 wav；离线/HTTP 错/空响应都返回 None（调用方回落文字）。"""
    if not await is_online(cfg):
        return None
    body: dict[str, object] = {
        "text": text, "text_lang": "zh",
        "ref_audio_path": ref_path, "prompt_text": ref_text, "prompt_lang": ref_lang,
        "media_type": "wav", "streaming_mode": False,
    }
    body.update(cfg.args)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg.base_url + "/tts", json=body,
                timeout=aiohttp.ClientTimeout(total=_TTS_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"[gpt_sovits] 合成 HTTP {resp.status}，语音回落文字")
                    _note_health(cfg.base_url, False)
                    return None
                data = await resp.read()
    except Exception as exc:
        logger.warning(f"[gpt_sovits] 合成失败（{type(exc).__name__}: {exc}），语音回落文字")
        _note_health(cfg.base_url, False)
        return None
    if len(data) < 2000:  # 空音频/异常短响应
        logger.warning(f"[gpt_sovits] 合成响应过短（{len(data)}B），语音回落文字")
        return None
    return data
