"""
gpu_power_log_nvidia.py

Logs NVIDIA GPU power draw over time using NVML (the driver's own API).
Useful as a high-resolution power trace (e.g. prefill spike vs. decode plateau
during an Ollama call) and as a cross-check on CodeCarbon's aggregate GPU figure.

Requires:
    pip install nvidia-ml-py

Run:
    python gpu_power_log_nvidia.py
"""

import csv
import sys
import time
from datetime import datetime

import pynvml

SAMPLE_INTERVAL = 0.5  # seconds between readings


def pick_gpu():
    count = pynvml.nvmlDeviceGetCount()
    if count == 0:
        sys.exit("No NVIDIA GPU found.")

    if count == 1:
        return pynvml.nvmlDeviceGetHandleByIndex(0)

    print("GPUs found:")
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print(f"  [{i}] {pynvml.nvmlDeviceGetName(handle)}")
    raw = input(f"Which GPU? (0-{count - 1}, blank = 0): ").strip()
    index = int(raw) if raw else 0
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def read_power(handle):
    """Watts. NVML reports milliwatts."""
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def prompt_duration():
    raw = input("Duration in seconds (blank = run until Ctrl+C): ").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        sys.exit("Not a number.")
    if seconds <= 0:
        sys.exit("Duration must be positive.")
    return seconds


def main():
    pynvml.nvmlInit()

    try:
        handle = pick_gpu()
        name = pynvml.nvmlDeviceGetName(handle)

        try:
            read_power(handle)
        except pynvml.NVMLError as e:
            sys.exit(f"This GPU does not report power draw via NVML: {e}")

        duration = prompt_duration()

        print(f"\nGPU: {name}")
        print(f"Interval: {SAMPLE_INTERVAL}s")
        print("Logging... (Ctrl+C to stop)\n")

        outfile = f"gpu_power_{datetime.now():%Y%m%d_%H%M%S}.csv"
        samples = []  # (elapsed_seconds, watts)
        start = time.time()

        try:
            with open(outfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["elapsed_s", "watts"])

                while True:
                    elapsed = time.time() - start
                    if duration is not None and elapsed >= duration:
                        break

                    watts = read_power(handle)
                    samples.append((elapsed, watts))
                    writer.writerow([f"{elapsed:.3f}", f"{watts:.2f}"])
                    f.flush()
                    print(f"\r{elapsed:7.1f}s   {watts:6.2f} W", end="")

                    time.sleep(SAMPLE_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")

    finally:
        pynvml.nvmlShutdown()

    if len(samples) < 2:
        print("\nNot enough samples to compute energy.")
        return

    # Trapezoidal integration: watts * seconds = joules
    joules = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        joules += (w0 + w1) / 2 * (t1 - t0)

    total_time = samples[-1][0] - samples[0][0]
    watts_only = [w for _, w in samples]

    print("\n")
    print(f"Samples:     {len(samples)}")
    print(f"Duration:    {total_time:.1f} s")
    print(f"Mean power:  {sum(watts_only) / len(watts_only):.2f} W")
    print(f"Min / Max:   {min(watts_only):.2f} / {max(watts_only):.2f} W")
    print(f"Energy:      {joules:.1f} J  ({joules / 3600:.4f} Wh)")
    print(f"Saved to:    {outfile}")


if __name__ == "__main__":
    main()
