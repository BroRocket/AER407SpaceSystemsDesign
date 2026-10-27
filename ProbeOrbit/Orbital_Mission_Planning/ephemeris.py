
import requests
import os 
import csv
from datetime import datetime

from calendar import monthrange
import time


def parse_horizons_result(result, vec_type: int, metric : bool = False):

    data = result["result"].split("$$SOE")[1].split("$$EOE")[0]

    lines = data.strip().splitlines()

    states = []

    i = 0

    while i < len(lines):

        if "=" in lines[i] and "A.D." in lines[i]:

            epoch = lines[i].split("=")[0].strip()

            position = lines[i + 1]

            if metric is False:
                r = [
                    float(position.split("X =")[1].split()[0]),
                    float(position.split("Y =")[1].split()[0]),
                    float(position.split("Z =")[1].split()[0])
                    ]
            else:
                r = [
                    1000*float(position.split("X =")[1].split()[0]),
                    1000*float(position.split("Y =")[1].split()[0]),
                    1000*float(position.split("Z =")[1].split()[0])
                    ]

            if vec_type == 2:

                velocity = lines[i + 2]

                if metric is False:
                    v = [
                        float(velocity.split("VX=")[1].split()[0]),
                        float(velocity.split("VY=")[1].split()[0]),
                        float(velocity.split("VZ=")[1].split()[0])
                        ]
                else:
                    v = [
                        1000*float(velocity.split("VX=")[1].split()[0]),
                        1000*float(velocity.split("VY=")[1].split()[0]),
                        1000*float(velocity.split("VZ=")[1].split()[0])
                        ]
                
                states.append({"epoch": epoch, "r": r, "v": v})
                i += 3

            else:
                states.append({"epoch": epoch, "r": r})
                i += 2

        else:
            i += 1

    return states

class Ephemeris:
    def __init__(self):
        self.url = "https://ssd.jpl.nasa.gov/api/horizons.api"

        self.params = {"format": "json",
                    "COMMAND": "'399'",
                    "EPHEM_TYPE": "'VECTORS'",
                    "CENTER": "'@sun'",
                    "START_TIME": "'2026-01-01'",
                    "STOP_TIME": "'2026-01-02'",
                    "STEP_SIZE": "'1 d'",
                    "OUT_UNITS": "'KM-S'",
                    "VEC_TABLE": "'2'",
                    "VEC_CORR": "'NONE'",
                    "OBJ_DATA": "'NO'"}

    def get_state(self, body: str, start_date: str, end_date: str, kwargs : dict = {}, metric : bool = False):
        self.params["COMMAND"] = body
        self.params["START_TIME"] = start_date
        self.params["STOP_TIME"] = end_date
        for key, value in kwargs.items():
            self.params[key] = value
        response = requests.get(self.url, params=self.params)

        data = response.json()
        states = parse_horizons_result(data, int(self.params["VEC_TABLE"][1:-1]), metric)
    
        return states

    @staticmethod
    def _parse_epoch(epoch) -> float:
        """Convert a Horizons Julian Day epoch to a float."""
        return float(epoch)

    
    def save_to_csv(self, states: list, filepath: str, expected_step_seconds: float = None) -> int:
        """Append new ephemeris states without rereading the entire CSV."""

        if not states:
            return 0

        seconds_per_day = 86400.0
        tolerance_seconds = 0.01

        has_velocity = "v" in states[0]
        columns = ["epoch", "x", "y", "z"]

        if has_velocity:
            columns += ["vx", "vy", "vz"]

        # Convert the incoming batch into CSV rows.
        rows = []

        for state in states:
            if ("v" in state) != has_velocity:
                raise ValueError("All states must use the same format.")

            row = {
                "epoch": state["epoch"],
                "x": state["r"][0],
                "y": state["r"][1],
                "z": state["r"][2],
            }

            if has_velocity:
                row.update({
                    "vx": state["v"][0],
                    "vy": state["v"][1],
                    "vz": state["v"][2],
                })

            rows.append(row)

        times = [float(row["epoch"]) for row in rows]

        # Validate chronological order within this batch.
        for t1, t2 in zip(times, times[1:]):
            if t2 <= t1:
                raise ValueError("New states must have strictly increasing epochs.")
            
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        last_time = None

        if file_exists:
            # Read only the header and final row, not the entire database.
            with open(filepath, "rb") as f:
                header = f.readline().decode("utf-8-sig").strip()
                expected_header = ",".join(columns)

                if header != expected_header:
                    raise ValueError(f"CSV header mismatch: {header!r} != {expected_header!r}")

                f.seek(0, os.SEEK_END)
                pos = f.tell()

                # Skip trailing newline characters, if any.
                while pos > 0:
                    f.seek(pos - 1)
                    if f.read(1) not in (b"\n", b"\r"):
                        break
                    pos -= 1

                if pos == 0:
                    raise ValueError("CSV contains a header but no data.")

                # Read the last data line.
                end = pos
                while pos > 0:
                    f.seek(pos - 1)
                    if f.read(1) == b"\n":
                        break
                    pos -= 1

                f.seek(pos)
                last_line = f.read(end - pos).decode("utf-8").strip()

            last_row = next(csv.reader([last_line]))

            if len(last_row) != len(columns):
                raise ValueError("The final CSV row has an invalid number of columns.")

            last_time = float(last_row[0])

        # Infer cadence from the batch if not explicitly supplied.
        step_days = None

        if expected_step_seconds is not None:
            step_days = expected_step_seconds / seconds_per_day
        elif len(times) >= 2:
            step_days = times[1] - times[0]

        # Filter overlap with the existing database.
        if last_time is not None:
            start_index = 0

            while start_index < len(times) and times[start_index] <= last_time:
                start_index += 1

            rows = rows[start_index:]
            times = times[start_index:]

        if not rows:
            print("No new data to append.")
            return 0

        # Check the boundary between the existing file and new data.
        if last_time is not None:
            if step_days is None:
                raise ValueError("Cannot verify boundary cadence. Supply expected_step_seconds.")

            gap_seconds = (times[0] - last_time) * seconds_per_day

            if abs(gap_seconds - step_days * seconds_per_day) > tolerance_seconds:
                raise ValueError(f"Time-step mismatch at CSV boundary: expected {step_days * seconds_per_day} s, got {gap_seconds} s.")

        # Validate cadence within the incoming batch.
        if step_days is not None:
            for t1, t2 in zip(times, times[1:]):
                dt_seconds = (t2 - t1) * seconds_per_day

                if abs(dt_seconds - step_days * seconds_per_day) > tolerance_seconds:
                    raise ValueError(f"Inconsistent new-data time step: expected {step_days * seconds_per_day} s, got {dt_seconds} s.")

        # Append only after all checks pass.
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)

            if not file_exists:
                writer.writeheader()

            writer.writerows(rows)

        print(f"Appended {len(rows)} states to {filepath}")
        return len(rows)


if __name__ == "__main__":
    # test = Ephemeris()
    # test.get_state("'DES=1004083'", "'2026-01-01'", "'2026-01-02'")
    eph_orbits = Ephemeris()

    body = "'899'"
    filepath = "Ephemeris/NEPTUNE.csv"

    current = datetime(2025, 6, 1) # 2025
    final_date = datetime(2035, 6, 1)  # Exclusive endpoint

    while current < final_date:
        # Advance by two calendar months.
        month_index = current.month - 1 + 2
        next_year = current.year + month_index // 12
        next_month = month_index % 12 + 1

        chunk_end = datetime(next_year, next_month, 1)

        if chunk_end > final_date:
            chunk_end = final_date

        start_date = f"'{current.strftime('%Y-%m-%d %H:%M')}'"
        end_date = f"'{chunk_end.strftime('%Y-%m-%d %H:%M')}'"

        print(f"Fetching {start_date} to {end_date}...")

        try:
            states = eph_orbits.get_state(body=body, start_date=start_date,end_date=end_date, kwargs={"STEP_SIZE": "'1 m'"})

            print("First epoch:", states[0]["epoch"])
            print("Last epoch:", states[-1]["epoch"])

            written = eph_orbits.save_to_csv(states, filepath=filepath,expected_step_seconds=60)

            print(f"Saved {written} new states.")

        except Exception as e:
            print(f"Failed for {start_date} to {end_date}: {e}")
            raise

        # Next chunk starts exactly where the previous chunk ended.
        current = chunk_end

        time.sleep(1)

    print("Ephemeris download complete!")
