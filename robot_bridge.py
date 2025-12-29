import time
import random

class RobotCommander:
    def __init__(self, robot_id="FARM_BOT_01"):
        self.robot_id = robot_id
        self.is_connected = False

    def connect(self):
        """Simulates establishing connection via MQTT/WiFi"""
        print(f"[{self.robot_id}] Connecting to IoT Cloud...")
        time.sleep(1) # Fake delay
        self.is_connected = True
        print(f"[{self.robot_id}] Connected! Status: ONLINE")
        return True

    def dispatch(self, target_x, target_y, action="SPRAY"):
        """Sends a command to the robot to go to (x,y) and perform action"""
        if not self.is_connected:
            if not self.connect():
                return {"status": "error", "message": "Connection failed"}

        payload = {
            "device_id": self.robot_id,
            "cmd": "GOTO",
            "params": {
                "x_coord": target_x,
                "y_coord": target_y,
                "action": action,
                "speed": 100
            }
        }
        
        # Simulate network transmission
        time.sleep(0.5)
        print(f"[{self.robot_id}] >>> SENDING PAYLOAD: {payload}")
        return {"status": "success", "payload": payload, "message": f"Robot dispatched to ({target_x}, {target_y})"}
