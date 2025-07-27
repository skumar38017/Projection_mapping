#!/usr/bin/env python3
"""
🎯 Real-Time 3D Object Detection & Matching System
Quick start script - runs with conda environment
"""

import subprocess
import sys
import os

def main():
    """Run the system with conda environment"""
    print("🎯 Starting Real-Time Object Detection & Matching System...")
    
    try:
        # Use conda run to ensure proper environment
        cmd = [
            'conda', 'run', '-n', '.projection-mapping',
            'python', 'start_system.py'
        ]
        
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try: python start_system.py")

if __name__ == "__main__":
    main()
