#!/usr/bin/env python3
"""google/gemma-3-4b-it 本地推理接口 —— 强制 BF16。

模型是多模态的 Gemma3ForConditionalGeneration, 纯文本和图文都能跑。
config.json 里 torch_dtype 本来就是 bfloat16, 权重原生就是 BF16。
本模块把这一点钉死: 加载时指定 bfloat16, 加载后把所有参数扫一遍, 只要发现
一个非 BF16 的浮点权重就直接报错, 不允许模型被悄悄转成 fp16/fp32 跑。

命令行:
    python3 interface.py -p "用一句话解释什么是梯度下降"
    python3 interface.py -p "图里有什么?" -i cat.jpg --max-new-tokens 128

代码里:
    from interface import Gemma3
    m = Gemma3()
    print(m.chat("你好, 介绍一下你自己"))
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_DIR = Path(__file__).resolve().parent / "models"
DTYPE = torch.bfloat16


class Bf16Violation(RuntimeError):
    """模型里出现了非 BF16 的浮点权重。"""


def assert_bf16(model: torch.nn.Module) -> None:
    """所有浮点 *参数* 必须是 BF16。

    只查 parameters 不查 buffers: transformers 的 rotary embedding 会把
    inv_freq 这类小常量注册成 float32 buffer, 那是按设计来的, 不是权重。
    """
    bad = [
        f"{name}: {t.dtype}"
        for name, t in model.named_parameters()
        if t.is_floating_point() and t.dtype is not DTYPE
    ]
    if bad:
        head = "\n  ".join(bad[:20])
        more = f"\n  ... 共 {len(bad)} 个" if len(bad) > 20 else ""
        raise Bf16Violation(f"以下权重不是 BF16, 拒绝运行:\n  {head}{more}")


class Gemma3:
    """BF16 本地推理封装。"""

    def __init__(self, model_dir: str | Path = MODEL_DIR, device: str | None = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"找不到模型目录: {self.model_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))

        # device_map 交给 accelerate 时会按 dtype 参数搬运, 这里显式锁死 bfloat16
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            dtype=DTYPE,  # transformers>=4.56 用 dtype, 旧版叫 torch_dtype
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # 仓库里的 generation_config.json 是 transformers 4.50 时代存的, 带着
        # cache_implementation="hybrid"。新版已改成自动推断, 留着每次加载都会打一行
        # 废弃警告, 置空即可 (行为不变)。
        self.model.generation_config.cache_implementation = None

        assert_bf16(self.model)

    # ------------------------------------------------------------------ #
    def _build_inputs(self, messages: list[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # 纯文本时 token_type_ids 全是 0, 掩码函数退化成恒等变换, 传不传结果一样。
        # 全 0 时直接丢掉: 省掉一遍逐 token 的图像掩码计算, 也避开 torch 的 vmap
        # 掩码路径 (那条路在 torch<2.6 上会直接报错, 图片输入仍然需要 >=2.6)。
        tti = inputs.get("token_type_ids")
        if tti is not None and not bool(tti.any()):
            inputs.pop("token_type_ids")

        # 输入跟着模型走同一种精度, 视觉塔也不例外
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.device, dtype=DTYPE)
            elif torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        return inputs

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """messages 用 OpenAI 风格: [{"role": "user", "content": [...]}]"""
        assert_bf16(self.model)  # 每次生成前再确认一遍, 防止外部改动

        inputs = self._build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs |= {"temperature": temperature, "top_p": top_p}
        else:
            # 模型自带的 generation_config 里带着 top_p=0.95 / top_k=64, 贪心解码下
            # 它们不生效, 但 transformers 每次都会打一行 "flags are not valid" 的
            # 警告, 这里清掉, 免得每次运行都冒出来
            gc = copy.deepcopy(self.model.generation_config)
            gc.top_p = None
            gc.top_k = None
            gen_kwargs["generation_config"] = gc

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][prompt_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    def chat(self, prompt: str, image: str | Path | None = None, **kw) -> str:
        """纯文本, 或带一张图。"""
        content: list[dict] = []
        if image is not None:
            from PIL import Image

            content.append({"type": "image", "image": Image.open(image).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        return self.generate([{"role": "user", "content": content}], **kw)

    def multi_turn(self, turns: list[tuple[str, str]], **kw) -> str:
        """turns = [("user", "..."), ("assistant", "..."), ("user", "...")]"""
        messages = [{"role": r, "content": [{"type": "text", "text": t}]} for r, t in turns]
        return self.generate(messages, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description="gemma-3-4b-it 本地推理 (BF16)")
    ap.add_argument("-p", "--prompt", required=True, help="提示词")
    ap.add_argument("-i", "--image", default=None, help="可选图片路径")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default=None, help="默认 cuda, 没有 GPU 时回落 cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心解码")
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    if args.device is None and not torch.cuda.is_available():
        print("[warn] 没有可用 GPU, 回落到 CPU —— BF16 在 CPU 上会很慢", file=sys.stderr)

    model = Gemma3(args.model_dir, args.device)
    print(f"[info] 加载完成: {model.model.dtype} @ {model.device}", file=sys.stderr)

    text = model.chat(
        args.prompt,
        image=args.image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
