
import requests


def parse_horizons_result(result):

    data = result["result"].split("$$SOE")[1].split("$$EOE")[0]

    lines = data.strip().splitlines()

    states = []

    i = 0

    while i < len(lines):

        if "=" in lines[i] and "A.D." in lines[i]:

            epoch = lines[i].split("=")[0].strip()

            position = lines[i + 1]
            velocity = lines[i + 2]

            r = [
                float(position.split("X =")[1].split()[0]),
                float(position.split("Y =")[1].split()[0]),
                float(position.split("Z =")[1].split()[0])
            ]

            v = [
                float(velocity.split("VX=")[1].split()[0]),
                float(velocity.split("VY=")[1].split()[0]),
                float(velocity.split("VZ=")[1].split()[0])
            ]

            states.append({
                "epoch": epoch,
                "r": r,
                "v": v
            })

            i += 3

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

    def get_state(self, body: str, star_date: str, end_date: str):
        self.params["COMMAND"] = body
        self.params["START_TIME"] = star_date
        self.params["STOP_TIME"] = end_date
        response = requests.get(self.url, params=self.params)

        data = response.json()
        states = parse_horizons_result(data)
    
        return states


if __name__ == "__main__":
    test = Ephemeris()
    test.get_state("'DES=1004083'", "'2026-01-01'", "'2026-01-02'")