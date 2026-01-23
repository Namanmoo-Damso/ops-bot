"""
Spawns bot instances concurrently with intervals between each start.
All bots run until Ctrl+C is pressed, then they all exit cleanly.

Simplified version without video - audio only for Windows compatibility.

Usage:
    python stress_test_bots.py              # Spawns 30 bots (default)
    python stress_test_bots.py -n 5         # Spawns 5 bots
    python stress_test_bots.py --num-bots 10  # Spawns 10 bots
"""

import argparse
import subprocess
import time
import sys
import os


def get_user_id_for_bot(bot_number: int) -> str:
    """Generate dummy user ID for a bot.

    Format: 10000000-0000-0000-0000-000000000001 to 000000000100
    """
    return f"10000000-0000-0000-0000-{bot_number:012d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn multiple bot instances for stress testing (audio only)"
    )
    parser.add_argument(
        "-n",
        "--num-bots",
        type=int,
        default=30,
        help="Number of bots to spawn (default: 30)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1,
        help="Seconds between spawning each bot (default: 1)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable resource monitoring (prints CPU/RAM usage periodically)",
    )
    parser.add_argument(
        "-s", "--start", type=int, default=1, help="Starting bot number (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_bots = args.num_bots
    start_bot = args.start
    spawn_interval = args.interval
    enable_monitor = args.monitor

    # Ensure we are in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(
        f"Will spawn {total_bots} bots (starting from {start_bot}) with {spawn_interval}s intervals.",
        flush=True,
    )
    print("Audio only mode (no video).", flush=True)
    print("Press Ctrl+C to stop all bots.\n", flush=True)

    processes = []

    try:
        for i in range(start_bot, start_bot + total_bots):
            user_id = get_user_id_for_bot(i)
            print(
                f"[{time.strftime('%H:%M:%S')}] Starting bot {i} ({i - start_bot + 1}/{total_bots}) (userId={user_id})...",
                flush=True,
            )
            # Pass bot number and user ID as environment variables
            bot_env = os.environ.copy()
            bot_env["BOT_NUMBER"] = str(i)
            bot_env["USER_ID"] = user_id
            proc = subprocess.Popen(
                [sys.executable, "bot_client.py"],
                # Let bot output go to console
                stdout=None,
                stderr=None,
                env=bot_env,
            )
            processes.append(proc)
            print(
                f"[{time.strftime('%H:%M:%S')}] Bot {i} started (PID: {proc.pid})",
                flush=True,
            )

            # Wait before spawning the next bot (except after the last one)
            if i < start_bot + total_bots - 1:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Waiting {spawn_interval}s before next bot...",
                    flush=True,
                )
                time.sleep(spawn_interval)

        print(
            f"\n[{time.strftime('%H:%M:%S')}] All {total_bots} bots are running.",
            flush=True,
        )
        print("Press Ctrl+C to stop all bots.\n", flush=True)

        # Wait indefinitely until interrupted, with optional monitoring
        if enable_monitor:
            try:
                import psutil

                while True:
                    # Collect stats for all bot processes
                    total_cpu = 0.0
                    total_mem_mb = 0.0
                    alive_count = 0
                    for proc in processes:
                        if proc.poll() is None:
                            try:
                                p = psutil.Process(proc.pid)
                                total_cpu += p.cpu_percent(interval=0.1)
                                total_mem_mb += p.memory_info().rss / (1024 * 1024)
                                alive_count += 1
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass

                    # System-wide stats
                    sys_cpu = psutil.cpu_percent()
                    sys_mem = psutil.virtual_memory()

                    print(
                        f"[{time.strftime('%H:%M:%S')}] MONITOR: {alive_count} bots | "
                        f"Bot CPU: {total_cpu:.1f}% | Bot RAM: {total_mem_mb:.0f}MB | "
                        f"System CPU: {sys_cpu:.1f}% | System RAM: {sys_mem.percent:.1f}% "
                        f"({sys_mem.available // (1024*1024)}MB free)",
                        flush=True,
                    )
                    time.sleep(10)
            except ImportError:
                print(
                    "psutil not installed, monitoring disabled. Install with: pip install psutil",
                    flush=True,
                )
                while True:
                    time.sleep(1)
        else:
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print(
            f"\n[{time.strftime('%H:%M:%S')}] Caught Ctrl+C. Shutting down all bots...",
            flush=True,
        )

    finally:
        # Terminate all bot processes
        for i, proc in enumerate(processes, start_bot):
            if proc.poll() is None:  # Still running
                print(
                    f"[{time.strftime('%H:%M:%S')}] Terminating bot {i} (PID: {proc.pid})...",
                    flush=True,
                )
                proc.terminate()

        # Give them a moment to exit gracefully
        time.sleep(2)

        # Force kill any that didn't exit
        for i, proc in enumerate(processes, start_bot):
            if proc.poll() is None:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Force killing bot {i} (PID: {proc.pid})...",
                    flush=True,
                )
                proc.kill()

        # Wait for all to finish
        for proc in processes:
            proc.wait()

        print(f"[{time.strftime('%H:%M:%S')}] All bots stopped.", flush=True)


if __name__ == "__main__":
    main()
