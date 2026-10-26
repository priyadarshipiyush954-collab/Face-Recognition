"""
Attendance Manager for Face Recognition System
- Real-time punch logging into standard CSV files
- Duplicate punch debounce (prevents multiple registrations within cooldown window)
- Today's attendance tracking and historical queries
- CSV export and analytics
"""

import os
import csv
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

if os.getenv("VERCEL"):
    # Serverless disk is temporary and per-instance; use /tmp to avoid writes
    # to the read-only repository bundle.
    ATTENDANCE_DIR_DEFAULT = os.path.join(tempfile.gettempdir(), "aura-attendance")
else:
    ATTENDANCE_DIR_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Attendance"))


class AttendanceManager:
    def __init__(self, attendance_dir: str = ATTENDANCE_DIR_DEFAULT, cooldown_seconds: int = 45):
        self.attendance_dir = attendance_dir
        self.cooldown_seconds = cooldown_seconds
        os.makedirs(self.attendance_dir, exist_ok=True)
        if os.getenv("VERCEL"):
            self.alt_dir = self.attendance_dir
        else:
            # Also ensure Attandance folder exists for backwards compatibility
            alt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Attandance"))
            os.makedirs(alt_dir, exist_ok=True)
            self.alt_dir = alt_dir
        self.last_punch_times: Dict[str, float] = {}

    def get_today_filename(self) -> str:
        date_str = datetime.now().strftime("%d-%m-%Y")
        return os.path.join(self.attendance_dir, f"Attendance_{date_str}.csv")

    def punch_attendance(self, name: str, confidence: float = 95.0) -> Dict[str, Any]:
        """
        Logs attendance for a student. Returns status: 'logged', 'cooldown', or 'rejected'.
        """
        clean_name = name.strip()
        if not clean_name or clean_name.lower() in ["unknown", "uncalibrated", "unverified"]:
            return {
                "success": False,
                "status": "rejected",
                "message": "Cannot log attendance for unknown or unverified face.",
            }

        now = time.time()
        last_time = self.last_punch_times.get(clean_name, 0)
        elapsed = now - last_time

        if elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            return {
                "success": False,
                "status": "cooldown",
                "message": f"Attendance already logged recently for {clean_name}. Please wait {remaining}s.",
                "remaining_seconds": remaining,
            }

        timestamp_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        date_str = datetime.fromtimestamp(now).strftime("%d-%m-%Y")
        filepath = self.get_today_filename()
        alt_filepath = os.path.join(self.alt_dir, f"Attendance_{date_str}.csv")

        file_exists = os.path.exists(filepath)
        row = [clean_name, timestamp_str, f"{confidence:.1f}%", "Verified"]

        # Write to primary Attendance file
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["NAME", "TIME", "CONFIDENCE", "STATUS"])
            writer.writerow(row)

        # Write to alternate Attandance folder if distinct
        if filepath != alt_filepath:
            alt_exists = os.path.exists(alt_filepath)
            with open(alt_filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not alt_exists:
                    writer.writerow(["NAME", "TIME", "CONFIDENCE", "STATUS"])
                writer.writerow(row)

        self.last_punch_times[clean_name] = now

        return {
            "success": True,
            "status": "logged",
            "message": f"Attendance successfully marked for {clean_name}!",
            "record": {
                "name": clean_name,
                "time": timestamp_str,
                "date": date_str,
                "confidence": f"{confidence:.1f}%",
                "status": "Verified",
            },
        }

    def get_today_records(self) -> List[Dict[str, str]]:
        filepath = self.get_today_filename()
        if not os.path.exists(filepath):
            return []

        records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    records.append({
                        "name": row[0] if len(row) > 0 else "Unknown",
                        "time": row[1] if len(row) > 1 else "",
                        "confidence": row[2] if len(row) > 2 else "N/A",
                        "status": row[3] if len(row) > 3 else "Present",
                    })
        except Exception:
            pass

        # Return latest entries first
        return list(reversed(records))

    def get_summary_stats(self) -> Dict[str, Any]:
        records = self.get_today_records()
        unique_students = len(set(r["name"] for r in records))
        return {
            "total_punches": len(records),
            "unique_attendees": unique_students,
            "date": datetime.now().strftime("%d-%m-%Y"),
            "latest_record": records[0] if records else None,
        }
