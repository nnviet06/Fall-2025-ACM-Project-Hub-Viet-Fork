""" Import necessary libraries """
import tkinter as tk
import serial
import time
import serial.tools.list_ports
from tkinter import messagebox


""" Find arduino port, stored in ser """
def find_arduino():
    # auto-detect Arduino COM port
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if ("Arduino" in p.description) or ("CH340" in p.description) or ("USB Serial" in p.description):
            return p.device
    return None
# try to find Arduino
arduino_port = find_arduino()
if not arduino_port:
    messagebox.showerror("Error", "No Arduino detected. Please connect your board and restart.")
    exit()
try:
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)  # wait for Arduino to reset
except Exception as e:
    messagebox.showerror("Connection Error", f"Could not open {arduino_port}:\n{e}")
    exit()


""" Sliders GUI & Logic """
# GUI setup
root = tk.Tk()
root.title("Robot Arm Controller")
servo_names = [
    "Gripper",
    "Wrist 2",
    "Wrist 1",
    "Elbow",
    "Shoulders"
]

# slightly delay motors to prevent jagged movement
last_sent = 0
# create list to store sliders
sliders = []

def send_all_servos(_=None):
    global last_sent
    now = time.time()
    # slightly delay motors to prevent jagged movement
    if now - last_sent < 0.1:
        return
    
    # send 5 servo values separated by spaces
    try:
        servo_vals = [str(s.get()) for s in sliders[:5]]
        data = " ".join(servo_vals) + " 0\n"
        ser.write(data.encode())
        last_sent = now
    except serial.SerialException:
        messagebox.showerror("Connection Lost", "Lost connection to Arduino.")
        root.destroy()

# create 5 sliders for controlling 5 servos (gripper, wrist 2, wrist 1, elbow, and shoulder)
for name in servo_names:
    slider = tk.Scale(
        root,
        from_=0,
        to=180,
        orient='horizontal',
        length=400,
        label=name,
        command=send_all_servos
    )
    slider.set(90)
    slider.pack(padx=20, pady=10)
    sliders.append(slider)

_ignore_stepper_callback = False
def on_stepper_move(value):
    global last_sent, _ignore_stepper_callback
    if _ignore_stepper_callback:
        return

    try:
        step_delta = int(float(value))
    except ValueError:
        return
    if step_delta == 0:
        return

    now = time.time()
    # slightly delay motors to prevent jagged movement
    if now - last_sent < 0.1:
        return

    # send all motor values separated by spaces
    try:
        servo_vals = [str(s.get()) for s in sliders[:5]]
        data = " ".join(servo_vals) + " " + str(step_delta) + "\n"
        ser.write(data.encode())
        last_sent = now
        # reset slider back to 0 to prevent infinite spinning (IMPORTANT)
        _ignore_stepper_callback = True
        stepper_slider.set(0)
        _ignore_stepper_callback = False
    except serial.SerialException:
        messagebox.showerror("Connection Lost", "Lost connection to Arduino.")
        root.destroy()

# adds the final stepper slider to control the stepper motor for base rotation
stepper_slider = tk.Scale(
    root,
    from_=0,
    to=270,
    orient='horizontal',
    length=400,
    label="Base Rotation (Stepper: steps)",
    command=on_stepper_move
)
stepper_slider.set(0)
stepper_slider.pack(padx=20, pady=10)
sliders.append(stepper_slider)

root.mainloop()