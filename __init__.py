import random
import re

import nonebot
from nonebot.log import logger
from nonebot import require, on_command
from nonebot.plugin import PluginMetadata, inherit_supported_adapters

require("nonebot_plugin_alconna")

from nonebot_plugin_alconna import (
    Args,
    Field,
    Voice,
    Option,
    Arparma,
    Alconna,
    on_alconna,
)

from .config import Config, load_config
from .utils import generate, generate_v2, get_wav_duration, encode_to_silk


# matcher 结构（指令名/情绪帮助文案）定义期消费一次；行为值在 handler 内现读。
plugin_config = load_config()


__plugin_meta__ = PluginMetadata(
    name="GPT-SoVITS 语音合成",
    description="调用 GPT-SoVITS 的 API 生成语音",
    usage=f"{plugin_config.command} [text] [-e emotion] [-l language]",
    type="application",
    homepage="https://github.com/zhaomaoniu/nonebot-plugin-gpt-sovits",
    config=Config,
    supported_adapters=inherit_supported_adapters("nonebot_plugin_alconna"),
)


emotion_map = {
    index: emotion.name
    for index, emotion in enumerate(plugin_config.emotions)
}
emotion_options = "\n".join(
    [f"{index}: {emotion}" for index, emotion in emotion_map.items()]
)


tts = on_alconna(
    Alconna(
        plugin_config.command,
        Args["text", str, Field(completion=lambda: "请输入要转换的文本")],
        Option(
            "-e",
            alias=["--emotion"],
            args=Args[
                "emotion",
                str,
                Field(
                    completion=lambda: f"请输入要转换的情绪，可选情绪：\n{emotion_options}"
                ),
            ],
            default="0",
            help_text=f"可选情绪：\n{emotion_options}",
        ),
        Option(
            "-l",
            alias=["--language"],
            args=Args[
                "language",
                str,
                Field(
                    completion=lambda: "请输入要转换的语言，可选语言：中文、英文、日文、中英混合、日英混合、多语种混合"
                ),
            ],
            default="auto",
            help_text="可选语言：中文、英文、日文、中英混合、日英混合、多语种混合",
        ),
    ),
    use_cmd_start=True,
    auto_send_output=True,
    comp_config={"lite": True},
    skip_for_unmatch=False,
    priority=10,
)
help_matcher = on_command(
    "gptsovits帮助", priority=10, block=True, aliases={"gsv帮助", "gsvhelp"}
)

_REPEAT_RUN = re.compile(r"(.)\1{3,}")


@tts.handle()
async def handle_tts(arp: Arparma):
    text: str = arp.main_args["text"]
    # 同一个字连打 4 次以上压到 3 次：t2s 训练集里没有长重复串，「啦啦啦啦…」会
    # 糊成杂音或整串被吞（实测 3 连字以内稳定），压完发音干净也不丢语气
    text = _REPEAT_RUN.sub(r"\1\1\1", text)
    # 新版 alconna 的 Option default 不落进 Arparma（缺省时 args 为空表），显式判空取默认
    e_args: dict = arp.options["e"].args if "e" in arp.options else {}
    emotion_index: str = str(e_args["emotion"]) if "emotion" in e_args else "0"
    l_args: dict = arp.options["l"].args if "l" in arp.options else {}
    text_language: str = str(l_args["language"]) if "language" in l_args else "auto"

    cfg = load_config()
    if not cfg.emotions:
        await tts.finish("情绪列表为空，请先在 config/gpt_sovits.toml 里填 [[emotions]]")
    index = int(emotion_index)
    if not 0 <= index < len(cfg.emotions):
        await tts.finish(f"情绪序号 {index} 不存在，可选 0~{len(cfg.emotions) - 1}")
    emotion = cfg.emotions[index]
    sentence = random.choice(emotion.sentences)
    base_url = cfg.base_url

    if cfg.api_v2:
        try:
            wav_file = await generate_v2(
                base_url,
                text,
                text_language,
                sentence.path,
                prompt_text=sentence.text,
                prompt_lang=sentence.language,
                **{
                    k: v
                    for k, v in cfg.args.items()
                    if k
                    in [
                        "aux_ref_audio_paths",
                        "top_k",
                        "top_p",
                        "temperature",
                        "text_split_method",
                        "batch_size",
                        "batch_threshold",
                        "split_bucket",
                        "speed_factor",
                        "fragment_interval",
                        "streaming_mode",
                        "seed",
                        "parallel_infer",
                        "repetition_penalty",
                    ]
                },
            )
        except ValueError as e:
            logger.error(f"生成语音时出现错误：{e}")
            await tts.finish("生成语音时出现错误，请联系 Bot 维护者查看日志")
    else:

        try:
            wav_file = await generate(
                base_url,
                refer_wav_path=sentence.path,
                prompt_text=sentence.text,
                prompt_language=sentence.language,
                text=text,
                text_language=text_language,
                **{
                    k: v
                    for k, v in cfg.args.items()
                    if k in ["cut_punc", "top_k", "top_p", "temperature", "speed"]
                },
            )
        except ValueError as e:
            logger.error(f"生成语音时出现错误：{e}")
            await tts.finish("生成语音时出现错误，请联系 Bot 维护者查看日志")

    mimetype = "audio/silk" if cfg.convert_to_silk else "audio/wav"
    duration = int(get_wav_duration(wav_file))

    if cfg.convert_to_silk:
        silk_file = encode_to_silk(wav_file)
        await tts.finish(Voice(raw=silk_file, mimetype=mimetype, duration=duration))
    else:
        await tts.finish(Voice(raw=wav_file, mimetype=mimetype, duration=duration))


@help_matcher.handle()
async def handle_help():
    cfg = load_config()
    options = "\n".join(f"{index}: {emotion.name}" for index, emotion in enumerate(cfg.emotions))
    await help_matcher.finish(
        "GPT-SoVITS 插件帮助\n"
        "使用方法：\n"
        f"   {cfg.command} [text] [-e emotion] [-l language] - 生成语音，可选情绪和语言\n"
        "   gptsovits帮助 - 显示本帮助\n"
        "使用示例：\n"
        f"   {cfg.command} 你好 - 生成一段语音\n"
        f"   {cfg.command} 你好 -e 1 - 以序号为 1 的情绪生成一段语音\n"
        f"   {cfg.command} hello -e 1 -l en - 以序号为 1 的情绪生成一段英文语音\n"
        f"可选情绪：\n{options}\n"
        f"可选语言：中文、英文、日文、中英混合、日英混合、多语种混合\n"
        f"(记得加上命令头，可用命令头: {nonebot.get_driver().config.command_start})"
    )
