"""
Moduł bezpieczeństwa GPU dla treningu ResNet-18 (Pokemony).

Cel: trenować na karcie graficznej NVIDIA (CUDA), ale w kontrolowanych warunkach:
  1. Limit pamięci VRAM        - proces nie zajmie całej pamięci karty.
  2. Limit obciążenia (duty)   - po każdym kroku GPU odpoczywa, więc nie pracuje na 100%.
  3. Nadzór temperatury        - powyżej progu trening jest wstrzymywany do ostygnięcia,
                                 po przekroczeniu progu krytycznego jest przerywany.
  4. Opcjonalny limit mocy (W) - wymaga uprawnień administratora/root.

Temperaturę odczytujemy przez NVML (pakiet nvidia-ml-py) lub, gdy go brak, przez `nvidia-smi`.
Jeśli temperatury nie da się odczytać, trening jest odmawiany (chyba że użyto
--allow-no-thermal-guard), bo wtedy nie ma nadzoru nad kartą.
"""

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch


class GPUSafetyError(RuntimeError):
    """Błąd uniemożliwiający bezpieczne uruchomienie lub kontynuację treningu na GPU."""


NO_CUDA_MESSAGE = (
    "CUDA jest niedostępne - PyTorch nie widzi karty NVIDIA.\n"
    "  * Zainstaluj PyTorch w wersji CUDA (patrz Instrukcja_Testow_i_Treningu.md, sekcja GPU),\n"
    "  * sprawdź sterownik poleceniem `nvidia-smi`,\n"
    "  * albo uruchom z --allow-cpu, aby świadomie trenować na procesorze."
)


@dataclass
class GPUSafetyConfig:
    gpu_index: int = 0
    vram_fraction: float = 0.7
    duty_cycle: float = 0.8
    pause_temp_c: int = 80
    resume_temp_c: int = 72
    abort_temp_c: int = 88
    power_limit_w: Optional[int] = None
    check_interval_s: float = 1.0
    allow_no_thermal_guard: bool = False

    def validate(self):
        if not 0.1 <= self.vram_fraction <= 1.0:
            raise ValueError("--vram-fraction musi być w przedziale 0.1-1.0")
        if not 0.1 <= self.duty_cycle <= 1.0:
            raise ValueError("--gpu-duty musi być w przedziale 0.1-1.0")
        if not self.resume_temp_c < self.pause_temp_c < self.abort_temp_c:
            raise ValueError("Wymagane: --resume-temp < --pause-temp < --abort-temp")


class _NvmlBackend:
    def __init__(self, index):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def temperature(self):
        return self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU)

    def power_w(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def power_limit_w(self):
        return self.nvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0

    def power_limit_range_w(self):
        lo, hi = self.nvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
        return lo / 1000.0, hi / 1000.0

    def set_power_limit_w(self, watts):
        self.nvml.nvmlDeviceSetPowerManagementLimit(self.handle, int(watts * 1000))

    def shutdown(self):
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass


class _SmiBackend:
    def __init__(self, index):
        self.index = index
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("brak nvidia-smi")
        self.temperature()

    def _query(self, field):
        out = subprocess.run(
            ["nvidia-smi", f"--id={self.index}", f"--query-gpu={field}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return float(out)

    def temperature(self):
        return self._query("temperature.gpu")

    def power_w(self):
        return self._query("power.draw")

    def power_limit_w(self):
        return self._query("power.limit")

    def power_limit_range_w(self):
        return self._query("power.min_limit"), self._query("power.max_limit")

    def set_power_limit_w(self, watts):
        subprocess.run(["nvidia-smi", f"--id={self.index}", "-pl", str(int(watts))],
                       capture_output=True, text=True, timeout=10, check=True)

    def shutdown(self):
        pass


def _open_backend(index):
    errors = []
    for factory in (_NvmlBackend, _SmiBackend):
        try:
            return factory(index)
        except Exception as e:
            errors.append(f"{factory.__name__}: {e}")
    return errors


class GPUGuard:
    """Nadzoruje kartę podczas treningu. Użycie: `with GPUGuard(cfg) as guard:`."""

    def __init__(self, config: GPUSafetyConfig):
        config.validate()
        self.cfg = config
        self.backend = None
        self.device = None
        self._original_power_limit = None
        self._last_check = 0.0
        self._step_started = None
        self.pause_count = 0
        self.paused_seconds = 0.0
        self.peak_temp = 0.0

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def setup(self):
        cfg = self.cfg
        if not torch.cuda.is_available():
            raise GPUSafetyError(NO_CUDA_MESSAGE)
        if cfg.gpu_index >= torch.cuda.device_count():
            raise GPUSafetyError(
                f"Brak GPU o indeksie {cfg.gpu_index} (dostępne: {torch.cuda.device_count()})."
            )

        self.device = torch.device(f"cuda:{cfg.gpu_index}")
        torch.cuda.set_device(self.device)
        props = torch.cuda.get_device_properties(self.device)
        total_gb = props.total_memory / 1024 ** 3

        backend = _open_backend(cfg.gpu_index)
        if isinstance(backend, list):
            if not cfg.allow_no_thermal_guard:
                raise GPUSafetyError(
                    "Nie można odczytać temperatury karty (NVML ani nvidia-smi), więc nie ma nadzoru nad GPU.\n"
                    "  * Zainstaluj: pip install nvidia-ml-py\n"
                    "  * lub użyj --allow-no-thermal-guard, jeśli pilnujesz temperatury innym narzędziem.\n"
                    f"  Szczegóły: {'; '.join(backend)}"
                )
            print("[GPU-GUARD] OSTRZEŻENIE: brak odczytu temperatury - działają tylko limity VRAM i obciążenia.")
        else:
            self.backend = backend

        torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction, self.device)

        print("=" * 80)
        print("TRYB BEZPIECZNEGO TRENINGU GPU")
        print("=" * 80)
        print(f"Karta:                {props.name} (indeks {cfg.gpu_index})")
        print(f"Limit VRAM:           {cfg.vram_fraction * 100:.0f}% z {total_gb:.1f} GB "
              f"(~{cfg.vram_fraction * total_gb:.1f} GB)")
        print(f"Limit obciążenia:     {cfg.duty_cycle * 100:.0f}% czasu pracy (reszta to przerwy dla GPU)")
        if self.backend is not None:
            temp = self.backend.temperature()
            print(f"Temperatura teraz:    {temp:.0f} C")
            print(f"Progi temperatury:    pauza >= {cfg.pause_temp_c} C, wznowienie <= {cfg.resume_temp_c} C, "
                  f"przerwanie >= {cfg.abort_temp_c} C")
            if temp >= cfg.pause_temp_c:
                print("[GPU-GUARD] Karta jest już gorąca - trening zacznie się po jej ostygnięciu.")
            self._apply_power_limit()
        print("=" * 80)

    def _apply_power_limit(self):
        cfg = self.cfg
        if cfg.power_limit_w is None:
            return
        try:
            self._original_power_limit = self.backend.power_limit_w()
            lo, hi = self.backend.power_limit_range_w()
            watts = min(max(cfg.power_limit_w, lo), hi)
            if watts != cfg.power_limit_w:
                print(f"[GPU-GUARD] Limit mocy dostosowany do zakresu karty: {watts:.0f} W ({lo:.0f}-{hi:.0f} W)")
            self.backend.set_power_limit_w(watts)
            print(f"Limit mocy:           {watts:.0f} W (było {self._original_power_limit:.0f} W)")
        except Exception as e:
            self._original_power_limit = None
            print(f"[GPU-GUARD] Nie udało się ustawić limitu mocy ({e}).\n"
                  "            Zwykle wymaga to uruchomienia terminala jako administrator/root. "
                  "Trening toczy się dalej bez limitu mocy.")

    def close(self):
        if self.backend is not None:
            if self._original_power_limit is not None:
                try:
                    self.backend.set_power_limit_w(self._original_power_limit)
                    print(f"[GPU-GUARD] Przywrócono pierwotny limit mocy: {self._original_power_limit:.0f} W")
                except Exception as e:
                    print(f"[GPU-GUARD] Nie udało się przywrócić limitu mocy ({e}). "
                          "Zostanie zresetowany po restarcie komputera.")
            self.backend.shutdown()
            self.backend = None
        if self.pause_count:
            print(f"[GPU-GUARD] Podsumowanie: {self.pause_count} pauz termicznych "
                  f"({self.paused_seconds:.0f} s), szczytowa temperatura {self.peak_temp:.0f} C")
        elif self.peak_temp:
            print(f"[GPU-GUARD] Podsumowanie: brak pauz termicznych, szczytowa temperatura {self.peak_temp:.0f} C")

    def step_begin(self):
        self._step_started = time.perf_counter()

    def step_end(self):
        """Wywołaj po każdym kroku: pilnuje temperatury i dokłada przerwę ograniczającą obciążenie."""
        if self._step_started is None:
            return
        torch.cuda.synchronize(self.device)
        busy = time.perf_counter() - self._step_started
        self._step_started = None

        self._check_temperature()

        if self.cfg.duty_cycle < 1.0:
            time.sleep(busy * (1.0 / self.cfg.duty_cycle - 1.0))

    def _check_temperature(self, force=False):
        if self.backend is None:
            return
        now = time.monotonic()
        if not force and now - self._last_check < self.cfg.check_interval_s:
            return
        self._last_check = now

        temp = self.backend.temperature()
        self.peak_temp = max(self.peak_temp, temp)

        if temp >= self.cfg.abort_temp_c:
            raise GPUSafetyError(
                f"Temperatura GPU {temp:.0f} C przekroczyła próg krytyczny {self.cfg.abort_temp_c} C - "
                "trening przerwany. Sprawdź chłodzenie/przepływ powietrza i zmniejsz --gpu-duty lub --batch-size."
            )
        if temp >= self.cfg.pause_temp_c:
            self._cool_down(temp)

    def _cool_down(self, temp):
        cfg = self.cfg
        self.pause_count += 1
        started = time.monotonic()
        print(f"\n[GPU-GUARD] {temp:.0f} C >= {cfg.pause_temp_c} C - pauza, czekam na ostygnięcie do "
              f"{cfg.resume_temp_c} C...")
        while True:
            time.sleep(5)
            temp = self.backend.temperature()
            if temp >= cfg.abort_temp_c:
                raise GPUSafetyError(
                    f"Temperatura GPU rośnie mimo pauzy ({temp:.0f} C >= {cfg.abort_temp_c} C) - "
                    "trening przerwany. Sprawdź wentylatory karty i obudowy."
                )
            if temp <= cfg.resume_temp_c:
                break
        self.paused_seconds += time.monotonic() - started
        print(f"[GPU-GUARD] {temp:.0f} C - wznawiam trening (pauza trwała {time.monotonic() - started:.0f} s).")
        self._last_check = time.monotonic()

    def status_line(self):
        """Krótki opis stanu karty do wypisania po epoce."""
        if self.backend is None:
            return ""
        try:
            temp = self.backend.temperature()
            self.peak_temp = max(self.peak_temp, temp)
            power = self.backend.power_w()
            vram = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            return f"GPU: {temp:.0f} C, {power:.0f} W, VRAM szczyt {vram:.2f} GB"
        except Exception:
            return ""


def add_gpu_safety_args(parser):
    g = parser.add_argument_group("Bezpieczeństwo GPU")
    g.add_argument('--allow-cpu', action='store_true',
                   help='Pozwól trenować na CPU (domyślnie skrypt wymaga GPU CUDA i nie przełączy się cicho na CPU)')
    g.add_argument('--gpu-index', type=int, default=0, help='Indeks karty CUDA (domyślnie 0)')
    g.add_argument('--vram-fraction', type=float, default=0.7,
                   help='Maksymalny ułamek VRAM dostępny dla procesu, 0.1-1.0 (domyślnie 0.7)')
    g.add_argument('--gpu-duty', type=float, default=0.8,
                   help='Docelowy ułamek czasu pracy GPU, 0.1-1.0; 0.8 = 20%% czasu karta odpoczywa (domyślnie 0.8)')
    g.add_argument('--pause-temp', type=int, default=80,
                   help='Temperatura (C), od której trening jest wstrzymywany (domyślnie 80)')
    g.add_argument('--resume-temp', type=int, default=72,
                   help='Temperatura (C), poniżej której trening jest wznawiany (domyślnie 72)')
    g.add_argument('--abort-temp', type=int, default=88,
                   help='Temperatura (C), od której trening jest przerywany (domyślnie 88)')
    g.add_argument('--power-limit-w', type=int, default=None,
                   help='Opcjonalny limit mocy karty w watach (wymaga administratora/root)')
    g.add_argument('--no-amp', action='store_true',
                   help='Wyłącz mieszaną precyzję (AMP). Domyślnie AMP jest włączone na GPU: szybciej i chłodniej')
    g.add_argument('--num-workers', type=int, default=None,
                   help='Liczba procesów wczytujących dane (domyślnie 2 na Linuxie, 0 na Windows)')
    g.add_argument('--allow-no-thermal-guard', action='store_true',
                   help='Pozwól trenować, gdy nie da się odczytać temperatury karty (niezalecane)')
    return parser


def config_from_args(args):
    return GPUSafetyConfig(
        gpu_index=args.gpu_index,
        vram_fraction=args.vram_fraction,
        duty_cycle=args.gpu_duty,
        pause_temp_c=args.pause_temp,
        resume_temp_c=args.resume_temp,
        abort_temp_c=args.abort_temp,
        power_limit_w=args.power_limit_w,
        allow_no_thermal_guard=args.allow_no_thermal_guard,
    )
