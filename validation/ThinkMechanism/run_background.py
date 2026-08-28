#!/usr/bin/env python3
"""
Background validation runner with CPU/resource limits
Launches expanded ThInK 1.0 validation with proper resource constraints
"""

import subprocess
import sys
import os
import signal
import time
import psutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'results' / 'ch3ocho_expanded'
LOG_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_LOG = LOG_DIR / 'background_runner.log'
PIDFILE = LOG_DIR / '.validation_pid'

def log_msg(msg):
    """Log to both console and file."""
    print(msg)
    with open(BACKGROUND_LOG, 'a') as f:
        f.write(msg + '\n')

def get_cpu_count():
    """Get number of CPUs to use (50% of available)."""
    total = psutil.cpu_count(logical=True)
    safe = max(1, total // 2)
    return safe

def launch_validation():
    """Launch the expanded validation as background process."""
    
    script_path = BASE_DIR / 'run_expanded_validation.py'
    
    if not script_path.exists():
        print(f"ERROR: {script_path} not found!")
        return None
    
    log_msg("\n" + "=" * 70)
    log_msg(f"[{time.strftime('%H:%M:%S')}] Launching expanded ThInK validation")
    log_msg("=" * 70)
    log_msg(f"Script: {script_path}")
    log_msg(f"Log: {BACKGROUND_LOG}")
    log_msg(f"CPU cores available: {psutil.cpu_count(logical=True)}")
    log_msg(f"CPU cores to use: {get_cpu_count()}")
    log_msg("=" * 70)
    
    try:
        # Launch Python script with nohup for true background execution
        # Output goes to background log
        with open(BACKGROUND_LOG, 'a') as logf:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from terminal
                cwd=str(BASE_DIR)
            )
        
        # Save PID for monitoring
        with open(PIDFILE, 'w') as f:
            f.write(str(proc.pid))
        
        log_msg(f"✓ Process started (PID: {proc.pid})")
        log_msg(f"✓ Running in background (detached from terminal)")
        log_msg(f"✓ Monitor progress with:")
        log_msg(f"    tail -f {BACKGROUND_LOG}")
        log_msg(f"    ps aux | grep {proc.pid}")
        log_msg("")
        
        return proc
        
    except Exception as e:
        log_msg(f"✗ Failed to launch: {e}")
        return None

def monitor_process(proc):
    """Monitor process and show status."""
    if proc is None:
        return
    
    try:
        p = psutil.Process(proc.pid)
        
        # Set process priority (nice level for Unix)
        try:
            p.nice(5)  # Lower priority than normal
            log_msg(f"  Set process nice level to 5 (lower priority)")
        except:
            pass
        
        # Show initial status
        with open(BACKGROUND_LOG, 'a') as f:
            f.write(f"\n[Process Status]\n")
            f.write(f"  PID: {p.pid}\n")
            f.write(f"  Priority: {p.nice()}\n")
            f.write(f"  Memory: {p.memory_info().rss / 1024 / 1024:.1f} MB\n")
            f.write(f"  Status: {p.status()}\n")
            f.write("\n")
        
    except psutil.NoSuchProcess:
        pass

def show_monitor_command():
    """Show how to monitor the background process."""
    log_msg("\n" + "=" * 70)
    log_msg("MONITORING COMMANDS")
    log_msg("=" * 70)
    log_msg(f"  Live log:        tail -f {BACKGROUND_LOG}")
    log_msg(f"  CPU/Memory:      ps aux | grep python | grep expanded")
    log_msg(f"  Kill process:    kill $(cat {PIDFILE})")
    log_msg(f"  Check results:   ls -lh {LOG_DIR / 'ch3ocho_expanded'}")
    log_msg("=" * 70 + "\n")

if __name__ == '__main__':
    
    # Clear old log
    with open(BACKGROUND_LOG, 'w') as f:
        f.write(f"Background Validation Runner\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
    
    proc = launch_validation()
    
    if proc:
        monitor_process(proc)
        show_monitor_command()
        
        log_msg("Process is running in background. You can close this terminal.\n")
    else:
        log_msg("Failed to start background process\n")
        sys.exit(1)
