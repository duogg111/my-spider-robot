"""servo_driver.py - điều khiển 18 servo MG996R qua 2x PCA9685 (I2C) trên Raspberry Pi 4.

Cấu trúc:
  Backend (PCA9685Backend / MockBackend)  : ghi xung thô xuống phần cứng
  ServoBank                               : đổi góc khớp -> góc servo -> xung,
                                            kẹp giới hạn, giới hạn tốc độ, khởi động/dừng an toàn

Các module khác (kinematics, gait...) CHỈ gọi ServoBank.set_joints() với 18 góc khớp (độ),
thứ tự: [chân0_coxa, chân0_femur, chân0_tibia, chân1_coxa, ...].

Chạy thử:
  python servo_driver.py --mock                    # kiểm tra logic, không cần phần cứng
  python servo_driver.py --real --stand            # đưa tất cả khớp về 0° (robot phải treo lơ lửng!)
  python servo_driver.py --real --servo 0 --amp 15 # quay chậm servo số 0 ±15°
  python servo_driver.py --real --relax            # thả lỏng toàn bộ
"""
from __future__ import annotations

import argparse
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import yaml

NUM_LEGS = 6
JOINTS_PER_LEG = 3
NUM_SERVOS = NUM_LEGS * JOINTS_PER_LEG
JOINT_NAMES = ("coxa", "femur", "tibia")


# --------------------------------------------------------------------------- config
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ServoMap:
    index: int          # 0..17
    leg: int
    joint: int          # 0 coxa, 1 femur, 2 tibia
    board: int
    channel: int
    direction: int      # +1 / -1
    offset_deg: float


def build_servo_map(cfg: dict) -> list[ServoMap]:
    n_boards = len(cfg["i2c"]["boards"])
    out: list[ServoMap] = []
    used: set[tuple[int, int]] = set()
    legs = cfg["legs"]
    if len(legs) != NUM_LEGS:
        raise ValueError(f"Cần đúng {NUM_LEGS} chân trong config, đang có {len(legs)}")
    for li, leg in enumerate(legs):
        for j in range(JOINTS_PER_LEG):
            board, ch = leg["board"], leg["channels"][j]
            if not (0 <= board < n_boards):
                raise ValueError(f"Chân {li}: board {board} không tồn tại")
            if not (0 <= ch <= 15):
                raise ValueError(f"Chân {li}: kênh {ch} ngoài 0..15")
            if (board, ch) in used:
                raise ValueError(f"Kênh trùng: board {board} kênh {ch}")
            used.add((board, ch))
            out.append(ServoMap(
                index=li * JOINTS_PER_LEG + j, leg=li, joint=j, board=board, channel=ch,
                direction=-1 if leg["invert"][j] else 1,
                offset_deg=float(leg["offsets_deg"][j]),
            ))
    return out


# --------------------------------------------------------------------------- backends
class Backend(ABC):
    n_boards: int

    @abstractmethod
    def write_board(self, board: int, pulses_us: Sequence[Optional[float]]) -> None:
        """Ghi xung (µs) cho tối đa 16 kênh của 1 board. None = không phát xung (servo thả lỏng)."""

    def set_output_enabled(self, enabled: bool) -> None:
        """Bật/tắt toàn bộ ngõ ra qua chân OE (nếu có)."""

    def close(self) -> None:
        pass


class MockBackend(Backend):
    """Giả lập: lưu lại xung cuối cùng, không đụng phần cứng. Dùng để test trên laptop."""

    def __init__(self, n_boards: int = 2, verbose: bool = False):
        self.n_boards = n_boards
        self.verbose = verbose
        self.output_enabled = False
        self.last: list[list[Optional[float]]] = [[None] * 16 for _ in range(n_boards)]
        self.writes = 0

    def write_board(self, board, pulses_us):
        row = list(pulses_us) + [None] * (16 - len(pulses_us))
        self.last[board] = row
        self.writes += 1
        if self.verbose:
            txt = " ".join("  --" if p is None else f"{p:4.0f}" for p in row)
            print(f"[mock] board{board} {'ON ' if self.output_enabled else 'OFF'} {txt}")

    def set_output_enabled(self, enabled):
        self.output_enabled = enabled
        if self.verbose:
            print(f"[mock] OE -> {'enabled' if enabled else 'disabled'}")


class PCA9685Backend(Backend):
    MODE1, MODE2, PRESCALE, LED0_ON_L = 0x00, 0x01, 0xFE, 0x06

    def __init__(self, bus: int, addresses: Sequence[int], freq_hz: float = 50,
                 osc_hz: float = 25_000_000, oe_gpio: Optional[int] = None):
        from smbus2 import SMBus, i2c_msg  # import trễ để mock chạy được khi chưa cài smbus2
        self._i2c_msg = i2c_msg
        self._bus = SMBus(bus)
        self.addresses = list(addresses)
        self.n_boards = len(self.addresses)

        prescale = int(round(osc_hz / (4096 * freq_hz))) - 1
        if not 3 <= prescale <= 255:
            raise ValueError(f"prescale {prescale} ngoài phạm vi")
        self._tick_us = (prescale + 1) / osc_hz * 1e6   # thời gian 1 bước đếm (µs)

        # OE: giữ mức cao (tắt xung) NGAY từ đầu để servo không giật khi khởi tạo
        self._oe = None
        if oe_gpio is not None:
            from gpiozero import DigitalOutputDevice
            self._oe = DigitalOutputDevice(oe_gpio, initial_value=True)

        for addr in self.addresses:
            self._bus.write_byte_data(addr, self.MODE1, 0x10)        # sleep để đổi prescale
            self._bus.write_byte_data(addr, self.PRESCALE, prescale)
            self._bus.write_byte_data(addr, self.MODE2, 0x04)        # totem-pole
            self._bus.write_byte_data(addr, self.MODE1, 0x20)        # wake, auto-increment
            time.sleep(0.005)
            self._bus.write_byte_data(addr, self.MODE1, 0xA0)        # restart + auto-increment
        for b in range(self.n_boards):
            self.write_board(b, [None] * 16)

    def write_board(self, board, pulses_us):
        data = [self.LED0_ON_L]
        for ch in range(16):
            p = pulses_us[ch] if ch < len(pulses_us) else None
            if p is None:
                data += [0, 0, 0, 0x10]                              # full OFF
                continue
            c = max(0, min(4095, int(round(p / self._tick_us))))
            if c == 0:
                data += [0, 0, 0, 0x10]
                continue
            on = (ch * 256) % 4096          # so-le cạnh lên giữa các kênh -> giảm dòng đỉnh
            off = (on + c) % 4096
            data += [on & 0xFF, on >> 8, off & 0xFF, off >> 8]
        # 1 giao dịch I2C duy nhất cho cả 16 kênh (65 byte); ~1 ms ở 400 kHz
        self._bus.i2c_rdwr(self._i2c_msg.write(self.addresses[board], data))

    def set_output_enabled(self, enabled):
        if self._oe is not None:
            self._oe.off() if enabled else self._oe.on()   # OE active-low

    def close(self):
        try:
            self.set_output_enabled(False)
        finally:
            self._bus.close()
            if self._oe is not None:
                self._oe.close()


# --------------------------------------------------------------------------- ServoBank
class ServoBank:
    """Lớp an toàn duy nhất giữa thuật toán (IK/gait) và phần cứng."""

    def __init__(self, cfg: dict, backend: Backend):
        s = cfg["servo"]
        self.backend = backend
        self.map = build_servo_map(cfg)
        self.pulse_min = float(s["pulse_min_us"])
        self.pulse_max = float(s["pulse_max_us"])
        self.range_deg = float(s["range_deg"])
        self.center = float(s["center_deg"])
        self.min_deg = float(s["min_deg"])
        self.max_deg = float(s["max_deg"])
        self.max_speed = float(s["max_speed_dps"])
        self._servo_deg: list[float] = [self.center] * NUM_SERVOS
        self._last_t = 0.0
        self.started = False

    # ---- chuyển đổi
    def joint_to_servo_deg(self, i: int, joint_deg: float) -> float:
        m = self.map[i]
        deg = self.center + m.offset_deg + m.direction * joint_deg
        return max(self.min_deg, min(self.max_deg, deg))

    def servo_deg_to_us(self, deg: float) -> float:
        return self.pulse_min + (deg / self.range_deg) * (self.pulse_max - self.pulse_min)

    def _flush(self) -> None:
        boards = [[None] * 16 for _ in range(self.backend.n_boards)]
        for m in self.map:
            boards[m.board][m.channel] = self.servo_deg_to_us(self._servo_deg[m.index])
        for b, pulses in enumerate(boards):
            self.backend.write_board(b, pulses)

    # ---- vòng đời
    def start(self, initial_joints_deg: Sequence[float]) -> None:
        """Ghi tư thế ban đầu khi ngõ ra còn bị tắt, rồi mới bật OE -> không giật."""
        self._check(initial_joints_deg)
        self.backend.set_output_enabled(False)
        self._servo_deg = [self.joint_to_servo_deg(i, a) for i, a in enumerate(initial_joints_deg)]
        self._flush()
        time.sleep(0.2)
        self.backend.set_output_enabled(True)
        self._last_t = time.monotonic()
        self.started = True

    def set_joints(self, joints_deg: Sequence[float]) -> list[float]:
        """Đặt 18 góc khớp (độ). Có kẹp giới hạn + giới hạn tốc độ. Trả về góc servo thực tế đã ghi."""
        if not self.started:
            raise RuntimeError("Gọi start() trước")
        self._check(joints_deg)
        now = time.monotonic()
        dt = min(now - self._last_t, 0.1)
        self._last_t = now
        max_step = self.max_speed * dt
        targets = [self.joint_to_servo_deg(i, a) for i, a in enumerate(joints_deg)]
        for i, tgt in enumerate(targets):
            delta = max(-max_step, min(max_step, tgt - self._servo_deg[i]))
            self._servo_deg[i] += delta
        self._flush()
        return list(self._servo_deg)

    def estop(self) -> None:
        """Dừng khẩn: cắt xung ngay lập tức qua OE."""
        self.backend.set_output_enabled(False)
        self.started = False

    def relax(self) -> None:
        """Thả lỏng toàn bộ servo."""
        self.backend.set_output_enabled(False)
        for b in range(self.backend.n_boards):
            self.backend.write_board(b, [None] * 16)
        self.started = False

    def close(self) -> None:
        self.relax()
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _check(angles: Sequence[float]) -> None:
        if len(angles) != NUM_SERVOS:
            raise ValueError(f"Cần {NUM_SERVOS} góc, nhận {len(angles)}")
        for i, a in enumerate(angles):
            if not math.isfinite(a):
                raise ValueError(f"Servo {i}: góc không hợp lệ ({a}) - thường do IK ngoài tầm với")


def create_bank(config_path: str = "config.yaml", mock: bool = False, verbose: bool = False) -> ServoBank:
    cfg = load_config(config_path)
    i2c = cfg["i2c"]
    if mock:
        backend: Backend = MockBackend(len(i2c["boards"]), verbose=verbose)
    else:
        backend = PCA9685Backend(i2c["bus"], i2c["boards"], i2c["pwm_freq_hz"],
                                 i2c["osc_hz"], i2c.get("oe_gpio"))
    return ServoBank(cfg, backend)


# --------------------------------------------------------------------------- CLI
def _selftest(bank: ServoBank) -> None:
    print("== Self-test (mock) ==")
    bank.start([0.0] * NUM_SERVOS)
    print("Nhảy khớp 0 từ 0° lên 90° - phải bị giới hạn tốc độ và kẹp ở max_deg:")
    for step in range(6):
        time.sleep(0.02)
        goal = [90.0] + [0.0] * (NUM_SERVOS - 1)
        deg = bank.set_joints(goal)
        print(f"  step {step}: servo0 = {deg[0]:6.2f}°")
    try:
        bank.set_joints([float("nan")] + [0.0] * 17)
    except ValueError as e:
        print("NaN bị chặn đúng:", e)
    bank.relax()
    print("OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stand", action="store_true", help="đưa mọi khớp về 0° và giữ")
    ap.add_argument("--relax", action="store_true", help="thả lỏng servo rồi thoát")
    ap.add_argument("--servo", type=int, help="quay chậm 1 servo (0..17)")
    ap.add_argument("--amp", type=float, default=15.0, help="biên độ ± (độ)")
    ap.add_argument("--period", type=float, default=4.0, help="chu kỳ (giây)")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    bank = create_bank(args.config, mock=args.mock, verbose=args.mock and args.servo is not None)
    try:
        if args.mock and args.servo is None and not args.stand and not args.relax:
            _selftest(bank)
            return
        if args.relax:
            bank.relax()
            return
        bank.start([0.0] * NUM_SERVOS)
        if args.stand:
            print("Đang giữ tư thế trung tính. Ctrl+C để thả lỏng.")
            while True:
                bank.set_joints([0.0] * NUM_SERVOS)
                time.sleep(0.02)
        elif args.servo is not None:
            if not 0 <= args.servo < NUM_SERVOS:
                raise SystemExit("--servo phải trong 0..17")
            m = bank.map[args.servo]
            print(f"Servo {args.servo}: chân {m.leg} ({JOINT_NAMES[m.joint]}), "
                  f"board {m.board} kênh {m.channel}. Ctrl+C để dừng.")
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.duration:
                t = time.monotonic() - t0
                pose = [0.0] * NUM_SERVOS
                pose[args.servo] = args.amp * math.sin(2 * math.pi * t / args.period)
                bank.set_joints(pose)
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDừng.")
    finally:
        bank.close()   # luôn thả lỏng servo khi thoát


if __name__ == "__main__":
    main()
