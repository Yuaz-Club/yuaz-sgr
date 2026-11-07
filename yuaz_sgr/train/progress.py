"""YUAZ-SGR 训练进度面板 (原地刷新, 非 TTY 自动降级为追加输出)"""
import os
import sys

def _enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
            k.SetConsoleMode(k.GetStdHandle(-12), 7)
        except Exception:
            pass

FG = {
    "title": "\x1b[38;5;103m",
    "label": "\x1b[38;5;242m",
    "value": "\x1b[38;5;250m",
    "good": "\x1b[38;5;71m",
    "bad": "\x1b[38;5;174m",
    "flat": "\x1b[38;5;240m",
    "warn": "\x1b[38;5;180m",
    "best": "\x1b[38;5;109m",
    "bar_lo": 60,
    "bar_hi": 68,
    "bar_empty": 237,
}
STAGE_FG = {"ae": "\x1b[38;5;66m", "gan": "\x1b[38;5;132m", "joint": "\x1b[38;5;136m"}
STAGE_BG = {"ae": "\x1b[48;5;23m", "gan": "\x1b[48;5;53m", "joint": "\x1b[48;5;58m"}
RESET = "\x1b[0m"
BLOCKS = "▁▂▃▄▅▆▇█"
RAMP = ".:-=+*#%@"
LOSS_ORDER = ["stft", "rvq", "ap", "gan", "fm", "cfm", "feature", "lag", "dur", "f0_pred"]
LOSS_LABEL = {
    "stft": "stft", "rvq": "rvq", "ap": "ap", "gan": "gan", "fm": "fm", "cfm": "cfm",
    "feature": "feat", "lag": "lag", "dur": "dur", "f0_pred": "f0p",
}

def _human(t):
    t = max(0, int(t))
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

def _num(v):
    a = abs(v)
    if a != a or a == float("inf"):
        return "nan"
    if a >= 10000:
        return f"{v / 1000:.1f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"

class ProgressDashboard:
    def __init__(self, total_steps, stage_ends=None, width=40, sparks=32,
                 color=None, live=None, unicode=None, spike_ratio=2.0, log_interval=100,
                 start_step=0):
        self.total = int(total_steps)
        self.start_step = int(start_step)
        self.width = width
        self.sparks = sparks
        self.spike_ratio = spike_ratio
        self.log_interval = int(log_interval)
        self.stage_ends = stage_ends or [("ae", self.total // 10),
                                         ("gan", self.total // 4), ("joint", self.total)]
        tty = sys.stdout.isatty()
        self.color = tty if color is None else color
        if os.environ.get("NO_COLOR"):
            self.color = False
        self.live = tty if live is None else live
        if unicode is None:
            enc = (getattr(sys.stdout, "encoding", "") or "").lower()
            self.unicode = "utf" in enc
        else:
            self.unicode = unicode
        if self.color:
            _enable_ansi()
        self.prev = {}
        self.hist = []
        self.drawn = 0
        self.best_val = None
        self.val_improved = False

    def _c(self, key, s):
        return f"{FG[key]}{s}{RESET}" if self.color else s

    def _bar(self, frac):
        n = int(round(frac * self.width))
        if not self.unicode:
            return "[" + "#" * n + "-" * (self.width - n) + "]"
        out = []
        for i in range(self.width):
            if i < n:
                code = FG["bar_lo"] + int(i / max(1, self.width - 1) * (FG["bar_hi"] - FG["bar_lo"]))
                out.append(f"\x1b[38;5;{code}m█" if self.color else "█")
            else:
                out.append(f"\x1b[38;5;{FG['bar_empty']}m░" if self.color else "░")
        return "".join(out) + (RESET if self.color else "")

    def _spark(self, vals):
        v = [x for x in vals[-self.sparks:]
             if x == x and x not in (float("inf"), float("-inf"))]
        if not v:
            return ""
        lo, hi = min(v), max(v)
        if hi - lo < 1e-9:
            return (BLOCKS[0] if self.unicode else RAMP[0]) * len(v)
        ramp = BLOCKS if self.unicode else RAMP
        return "".join(ramp[min(7, int((x - lo) / (hi - lo) * 7))] for x in v)

    def _loss_cell(self, k, v):
        p = self.prev.get(k)
        self.prev[k] = v
        name = self._c("label", LOSS_LABEL[k])
        if v != v or v in (float("inf"), float("-inf")):
            return f"{name} {self._c('bad', 'nan')}{self._c('warn', ' ⚠')}"
        if p is not None and p > 0 and v > p * self.spike_ratio:
            return f"{name} {self._c('bad', _num(v))}{self._c('warn', ' ⚠')}"
        val = self._c("value", _num(v))
        if p is None:
            return f"{name} {val} {self._c('flat', '·')}"
        if v < p:
            return f"{name} {val}{self._c('good', '↓')}"
        if v > p:
            return f"{name} {val}{self._c('bad', '↑')}"
        return f"{name} {val}{self._c('flat', '=')}"

    def _timeline(self, step, stage):
        order = [n for n, _ in self.stage_ends]
        cur = order.index(stage) if stage in order else 0
        if not self.unicode:
            segs, prev = [], 0
            for name, end in self.stage_ends:
                w = max(1, int((end - prev) / self.total * 20))
                segs.append(f"[{'=' * w}]" if name == stage else f" {'=' * w} ")
                prev = end
            return "   " + "".join(segs) + f"  ^{step}"
        cells, prev = [], 0
        for idx, (name, end) in enumerate(self.stage_ends):
            w = max(1, int(round((end - prev) / self.total * 26)))
            if idx < cur:
                cells.append(f"\x1b[38;5;238m{'▬' * w}" if self.color else "▬" * w)
            elif idx == cur:
                cells.append(f"{STAGE_FG[name]}{'▬' * w}" if self.color else "▬" * w)
            else:
                cells.append(f"\x1b[38;5;236m{'▭' * w}" if self.color else "▭" * w)
            prev = end
        bar = "".join(cells) + (RESET if self.color else "")
        names = "  ".join(
            (f"{STAGE_FG[n]}{n.upper()}{RESET}" if self.color else n.upper()) if n == stage
            else self._c("flat", n.upper()) for n in order)
        return f"   {names}\n   {bar}  {self._c('value', f'{step:,}')}"

    def _chip(self, stage):
        label = stage.upper()
        if not self.color:
            return f"[{label}]"
        return f"{STAGE_BG.get(stage, '')}{STAGE_FG.get(stage, '')} {label} {RESET}"

    def render(self, step, stage, losses, rate, elapsed, lr):
        frac = min(1.0, step / self.total)
        total = losses.get("total", 0.0)
        if total == total and total not in (float("inf"), float("-inf")):
            self.hist.append(total)
        done = step - self.start_step
        eta = (elapsed / done) * (self.total - step) if done > 0 else 0.0
        lines = []
        lines.append(
            f" {self._c('title', 'YUAZ-SGR')}  "
            f"{self._c('value', f'{step:,}/{self.total:,}')}  {self._bar(frac)} "
            f"{self._c('value', f'{frac * 100:5.1f}%')}  {self._chip(stage)}")
        val_txt = ""
        if self.best_val is not None:
            star = self._c("best", "★") if self.val_improved else ""
            val_txt = f" · best {self._c('value', f'{self.best_val:.4f}')}{star}"
        lines.append(
            f"   {self._c('value', _human(elapsed))} · eta {self._c('value', _human(eta))} · "
            f"{self._c('value', f'{rate:.2f} it/s')} · lr {self._c('value', f'{lr:.1e}')}{val_txt}")
        cells = [self._loss_cell(k, losses[k]) for k in LOSS_ORDER if k in losses]
        lines.append("   " + "  ".join(cells))
        _hist_min = _num(min(self.hist)) if self.hist else "-"
        lines.append(f"   {self._c('label', 'total')} {self._c('value', _num(total))}  "
                     f"{self._c('title', self._spark(self.hist))}  "
                     f"{self._c('label', f'min {_hist_min}')}")
        lines.append(self._timeline(step, stage))
        return "\n".join(lines)

    def draw(self, step, stage, losses, rate, elapsed, lr):
        text = self.render(step, stage, losses, rate, elapsed, lr)
        n = text.count("\n") + 1
        if not self.live:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            self.drawn = 0
            return
        if self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}A")
        sys.stdout.write("\x1b[J" + text + "\n")
        sys.stdout.flush()
        self.drawn = n

    def val(self, value):
        if value is None:
            return
        if self.best_val is None or value < self.best_val:
            self.best_val = float(value)
            self.val_improved = True
        else:
            self.val_improved = False

    def log(self, msg):
        if self.live and self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}A\x1b[J")
            self.drawn = 0
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def close(self):
        if self.live and self.drawn:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.drawn = 0

def _demo():
    import math
    import random
    import time
    d = ProgressDashboard(200000, stage_ends=[("ae", 20000), ("gan", 50000), ("joint", 200000)])
    base = {"stft": 24000.0, "rvq": 2.1, "ap": 1.44, "gan": 2.4, "fm": 1.8,
            "cfm": 0.9, "feature": 0.62, "lag": 0.4, "dur": 0.5, "f0_pred": 0.35}
    t0 = time.time()
    for i in range(140):
        step = int((i + 1) / 140 * 200000)
        stage = "ae" if step <= 20000 else ("gan" if step <= 50000 else "joint")
        decay = math.exp(-step / 200000 * 2.2)
        losses = {}
        for k, b in base.items():
            if k in ("gan", "fm") and stage == "ae":
                continue
            if k in ("cfm", "feature", "lag", "dur", "f0_pred") and stage != "joint":
                continue
            losses[k] = b * (0.35 + 0.65 * decay) * (1 + random.uniform(-0.05, 0.05))
        if i == 96:
            losses["ap"] *= 3.1
        losses["total"] = sum(losses.values())
        if step % 20000 < 1500:
            d.val(3.42 - step / 200000 * 0.5 + random.uniform(-0.02, 0.02))
        d.draw(step, stage, losses, 7.4 + random.uniform(-0.3, 0.3),
               time.time() - t0, 1e-5)
        time.sleep(0.05)
    d.close()

if __name__ == "__main__":
    _demo()
