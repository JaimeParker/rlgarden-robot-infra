# rlgarden-robot-infra

`robot_infra` is the real-robot layer of [rl-garden](https://github.com/JaimeParker/rl-garden).
It provides:

- a **device layer**: typed `Arm` / `Gripper` / `Camera` / `Teleop` contracts with backends for
  Franka (franky, legacy ROS1 bridge), AgileX Piper (pyAgxArm, piper_sdk), Intel RealSense and
  SpaceMouse, plus device-side **safety limits** (TCP box, joint limits, per-step caps);
- **discovery** of attached cameras, teleop devices, CAN interfaces, robot hosts, and HID devices;
- a **doctor** that checks a robot config layer by layer and prints a fix hint for each failure;
- a **robot daemon** that serves one robot over gRPC from the robot PC;
- `RemoteRobot`, a client-side mirror of that robot for use on a GPU machine.

The gym environment for real-world RL lives in
[rlgarden-real-world](https://github.com/JaimeParker/rlgarden-real-world), which imports
this package. `robot_infra` itself does not depend on `rl_garden`.

## Install

Requires Python >= 3.10.

```bash
pip install -e .                    # core: devices, discovery, doctor, fake backends
pip install -e '.[franky,grpc]'     # e.g. a Franka robot PC running the daemon
```

| Extra | Installs | You need it when | Machine |
|---|---|---|---|
| `franky` | `franky-control` | Franka arm/hand via libfranka (`arm_backend: franky`). The default wheel bundles one libfranka version; for other robot firmware, install the matching wheel (see [libfranka version mismatch](#troubleshooting)). | robot PC |
| `pyagxarm` | `pyAgxArm` (from GitHub) | Piper via `arm_backend: pyagxarm` (the default for Piper) | robot PC |
| `piper-sdk` | `piper_sdk` | Piper via the vendor SDK (`arm_backend: piper_sdk`) | robot PC |
| `realsense` | `pyrealsense2` | RealSense cameras | robot PC |
| `spacemouse` | `pyspacemouse>=2.0,<3` | `teleop: {backend: spacemouse}` (optional `hidapi` improves discovery) | robot PC |
| `grpc` | `grpcio>=1.60` | the daemon (`python -m robot_infra.server`) and `RemoteRobot` | robot PC **and** GPU machine |
| `jpeg` | `pillow` | `python -m robot_infra.server --jpeg-quality N` (OpenCV also works as the codec) | robot PC **and** GPU machine |
| `real` | `flask`, `pyzmq`, `scipy`, `pyspacemouse` | legacy Flask bridge/gripper server and the legacy `teleop/spacemouse/` (old `read_all()` API; do not combine with the `spacemouse` extra's 2.x pin; ROS1 must be installed separately) | robot PC |
| `sim` | `mani_skill` | the ManiSkill impedance controllers in `controller/simulator/` | GPU machine |
| `dev` | `pytest` | running the tests | any |

Vendor SDKs are imported lazily. The core package, the CLIs and `--mock` runs work
without any SDK installed.

## How to Start

These steps run on the robot PC (the computer wired to the arm, CAN adapter and cameras),
except step 5, which runs on the GPU machine.

### 1. Write `robot.yaml`

A robot config is one YAML mapping with a `type` key. It can also be nested under a
top-level `robot:` key. Unknown keys are rejected with the list of valid ones. Omitted
keys take the defaults shown below. Full annotated copies are in
[`docs/robot.franka.example.yaml`](docs/robot.franka.example.yaml) and
[`docs/robot.piper.example.yaml`](docs/robot.piper.example.yaml).

**Franka Panda + Franka Hand + one RealSense**

```yaml
type: franka
name: franka                  # default "franka"
arm_backend: franky           # franky (default) | serl_http (legacy ROS1 bridge) | fake
robot_ip: 172.16.0.2          # FCI IP set in Desk (default 172.16.0.2)
# bridge_url: http://127.0.0.1:5000   # serl_http only: URL of the legacy Flask bridge
realtime: ignore              # ignore (default) | enforce (requires a PREEMPT_RT kernel)
relative_dynamics_factor: 0.2 # franky velocity/accel/jerk scale for move_to_* (default 0.2)
home_joints: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # 7 values (default shown)
safety_box: [0.3, -0.3, 0.05, 0.7, 0.3, 0.6]  # [xmin, ymin, zmin, xmax, ymax, zmax] m, base frame of tcp_pose (O_T_EE)
joint_limits:                 # 7 [lo, hi] pairs in rad; home_joints must lie inside
  - [-2.8, 2.8]
  - [-1.7, 1.7]
  - [-2.8, 2.8]
  - [-3.0, -0.1]
  - [-2.8, 2.8]
  - [0.0, 3.7]
  - [-2.8, 2.8]
max_delta_per_step: {xyz_m: 0.05, joint_rad: 0.1}  # servo step cap from the measured state
claim: true                   # daemon takes a per-host exclusive lock while connected
gripper:                      # null = no gripper
  backend: franky             # franky (default) | serl_http | fake
  max_width_m: 0.08
  mode: binary                # binary (default) | continuous
  binary_threshold: 0.5       # binary mode: an action > threshold opens
  shares_arm_session: false   # the Franka Hand opens its own connection
cameras:                      # default: []
  - name: wrist
    backend: realsense        # realsense (default) | fake
    serial: ""                # "" = use the only attached camera; required when >1 is attached
    width: 640
    height: 480
    fps: 30
    enable_depth: false
```

**AgileX Piper (CAN)**

```yaml
type: piper
name: piper
arm_backend: pyagxarm         # pyagxarm (default) | piper_sdk | fake
can_interface: can0           # default can0
bitrate: 1000000              # must match the interface (default 1000000)
firmware_profile: null        # pyagxarm only: default | v183 | v188 | v189; null probes the arm
home_joints: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 values, rad
safety_box: [0.1, -0.3, 0.1, 0.5, 0.3, 0.5]   # m, base frame; tcp_pose is the FLANGE pose (no tool offset)
joint_limits: [[-2.6, 2.6], [0.0, 3.1], [-2.9, 0.0], [-1.7, 1.7], [-1.2, 1.2], [-2.0, 2.0]]  # rad
max_delta_per_step: {xyz_m: 0.02, joint_rad: 0.05}
claim: true
gripper:
  backend: pyagxarm           # must match the arm backend: pyagxarm | piper_sdk
  max_width_m: 0.07
  mode: continuous            # Piper default
  binary_threshold: 0.5
  shares_arm_session: true    # required: the Piper gripper borrows the arm's CAN session
cameras: []
```

Safety limits and teleop (all optional, all three keys are read from the robot PC's yaml):

- `safety_box`: `[xmin, ymin, zmin, xmax, ymax, zmax]` in metres, in the frame of
  `read_state()["tcp_pose"]` (Franka: `O_T_EE`, Piper: the **flange**, not the gripper tip).
  Only the position is limited; the orientation is left untouched.
- `joint_limits`: one `[lo, hi]` pair per joint, in rad. `home_joints` must lie inside it.
- `max_delta_per_step`: `xyz_m` and/or `joint_rad`, a cap on how far a servo target may be from
  the measured state. Servo only; `move_to_*` is not capped.
- The yaml on the robot PC is the source of truth. `servo_*` targets outside the limits are
  **clipped** and counted (`safety_clips`); `move_to_*` targets outside them are **refused** with
  `SafetyViolation` and nothing is sent (`safety_rejects`). The same applies through the daemon,
  `send_action`, and the local API, see [Safety rules](#safety-rules).
- `teleop:` adds an operator input device as a part of the robot (see [step 7](#7-add-a-teleop-device)).

Notes:

- `shares_arm_session: true` is mandatory for the `pyagxarm` and `piper_sdk` grippers and
  forbidden for `franky` and `serl_http`. A mismatch raises `ValueError` at build time.
- With `arm_backend: piper_sdk`, also set `gripper.backend: piper_sdk`. That backend ignores
  `firmware_profile` and always opens the bus at 1 Mbit/s, so `bitrate` is only checked
  against the interface.
- `serl_http` has no joint-space control and needs the legacy bridge running (see [Legacy](#legacy)).

Build a robot without touching hardware to check the file:

```python
from robot_infra.devices import Robot

robot = Robot.from_yaml("robot.yaml")   # builds parts; does not connect
print(robot.describe_text())
```

```text
franka (franka)
  arm: FrankyArm [sdk=franky, connected=False]
  gripper: FrankyGripper [sdk=franky, connected=False]
  cameras: PartGroup [sdk=None, connected=False]
    wrist: RealSenseCamera [sdk=pyrealsense2, connected=False]
```

With a `teleop:` section a last line is added, for example `  teleop: SpaceMouseTeleop [sdk=pyspacemouse, connected=False]`.

### 2. Plug in and discover

```bash
python -m robot_infra.discovery --config robot.yaml
```

Discovery only probes. It never connects to a robot. `--config` is optional: when given, the
config's `robot_ip` is pinged and TCP-probed on ports 22, 80, 443, 1337 and 50051.

On a machine with nothing attached (real output):

```text
$ python -m robot_infra.discovery --config robot.yaml
cameras:
  none
teleops:
  none
can interfaces:
  none
hosts:
  172.16.0.2: ping=no reply open ports=none
hid devices:
  none
```

With hardware attached, the lines take the form `realsense: <serial>, <serial>`,
`spacemouse: <vid>:<pid>:<path>` (or `/dev/hidrawN` when `hidapi` is not installed),
`can0: operstate=up bitrate=1000000`, `172.16.0.2: ping=ok open ports=<list>` and
`/dev/hidraw0`. The `fake` backends are not listed.

| Section | Source | What to look for |
|---|---|---|
| `cameras` | each camera backend's `discover()` (RealSense serials) | copy the serial into `cameras[].serial` |
| `teleops` | each teleop backend's `discover()` (SpaceMouse: `vid:pid:path` through `hidapi` when installed, else every `/dev/hidraw*`) | the device you put in `teleop:` shows up |
| `can interfaces` | `/sys/class/net/*` with type CAN, bitrate from `ip -details link show` | `operstate=up` and the bitrate in your yaml |
| `hosts` | `ping` and TCP connect to the config's `robot_ip` | `ping=ok`. For Franka, 1337 open usually means FCI is reachable. |
| `hid devices` | `/dev/hidraw*` | SpaceMouse and other HID teleop devices |

Without `--config`, `hosts:` prints `none`.

### 3. Check with the doctor

```bash
python -m robot_infra.doctor --config robot.yaml [--mock] [--fix] [--json]
```

The doctor runs five layers in order (a `teleop:` section adds checks to layers 1, 2 and 4):

| Layer | Checks | Notes |
|---|---|---|
| 1 `sdk` | each backend name resolves and its SDK imports (`sdk:arm`, `sdk:gripper`, `sdk:camera:<name>`, `sdk:teleop:<backend>`) | an unknown backend skips layers 3-5 |
| 2 `os` | ping `robot_ip` and port 1337 (informational); CAN interface exists / is up / bitrate; camera serial attached; `os:teleop:<name>` (a device of that backend is discovered); `os:hid` (a readable and writable `/dev/hidraw*` exists, SpaceMouse only) | driven by config fields; skipped for backends that need no SDK (fakes) |
| 3 `handshake` | takes the robot's device claim (unless `claim: false`), then `robot.connect()` | a failure (including a held claim) skips layers 4-5 |
| 4 `feedback` | the arm returns state; the gripper is readable; each camera delivers a frame within 2 s; `feedback:teleop:<name>`: the teleop poll thread is alive and healthy | |
| 5 `mode` | backend preflight: Franka RT kernel, robot mode and libfranka version; Piper status, follower feedback and firmware profile (pyagxarm) or gripper stroke (piper_sdk) | |

Flags:

- `--mock` swaps every backend for its fake and skips layer 2. Use it to check the YAML and
  the install with no hardware.
- `--fix` runs the autofix attached to a failed check, then re-runs that check. The useful
  autofixes are `arm.recover()` for a Franka in Reflex/UserStopped/Guiding mode
  (`mode:robot_mode`) and `camera.reopen()` for a camera that delivers no frames.
- `--json` prints a list of `{layer, name, ok, message, hint, skipped}` objects.
- Exit code: `1` if any check failed, else `0`. Skipped checks do not fail the run.

The doctor connects to the robot and disconnects it again. It never powers anything off.

A `--mock` run of the Franka example:

```text
$ python -m robot_infra.doctor --config robot.yaml --mock
[1/5] sdk:arm ... OK  no SDK needed
[1/5] sdk:gripper ... OK  no SDK needed
[1/5] sdk:camera:wrist ... OK  no SDK needed
[2/5] os:ping ... SKIP  skipped (--mock)
[2/5] os:tcp ... SKIP  skipped (--mock)
[2/5] os:camera:wrist ... SKIP  skipped (--mock)
[3/5] handshake ... OK  robot connected
[4/5] feedback:arm ... OK  arm feedback received
[4/5] feedback:gripper ... OK  gripper readable
[4/5] feedback:camera:wrist ... OK  frames arriving
[5/5] mode:none ... OK  backend declares no mode checks
summary: 8 ok, 0 failed, 3 skipped
```

The same config on a machine with no franky, no robot and no camera:

```text
$ python -m robot_infra.doctor --config robot.yaml
[1/5] sdk:arm ... FAIL  FrankyArm needs the 'franky' package
    hint: pip install franky-control (or: pip install 'rlgarden-robot-infra[franky]')
[1/5] sdk:gripper ... FAIL  FrankyGripper needs the 'franky' package
    hint: pip install franky-control (or: pip install 'rlgarden-robot-infra[franky]')
[1/5] sdk:camera:wrist ... OK  'pyrealsense2' importable
[2/5] os:ping ... FAIL  172.16.0.2 does not reply
    hint: check the Ethernet cable to the robot's Control port and the robot IP in Desk
[2/5] os:tcp ... OK  port 1337 closed (informational)
[2/5] os:camera:wrist ... FAIL  no RealSense device found (serial unset)
    hint: plug in the camera and run `python -m robot_infra.discovery`
[3/5] handshake ... FAIL  FrankyArm needs the 'franky' package
    hint: pip install franky-control (or: pip install 'rlgarden-robot-infra[franky]')
[4/5] feedback ... SKIP  skipped: handshake failed
[5/5] mode ... SKIP  skipped: handshake failed
summary: 2 ok, 5 failed, 2 skipped
```

With a `teleop: {backend: spacemouse}` section, `--mock` swaps the teleop for the fake one
(its options are dropped) and the extra lines are:

```text
$ python -m robot_infra.doctor --config robot.yaml --mock
...
[1/5] sdk:teleop:fake ... OK  no SDK needed
[2/5] os:teleop:teleop ... SKIP  skipped (--mock)
...
[4/5] feedback:teleop:teleop ... OK  teleop poll thread alive
```

Without `--mock`, on a machine without `pyspacemouse` and without a SpaceMouse, the teleop
lines read (the `os:hid` line is only checked for the SpaceMouse backend):

```text
[1/5] sdk:teleop:spacemouse ... FAIL  SpaceMouseTeleop needs the 'pyspacemouse' package
    hint: pip install 'pyspacemouse>=2.0,<3' (or: pip install 'rlgarden-robot-infra[spacemouse]')
[2/5] os:teleop:teleop ... FAIL  no 'spacemouse' teleop device found
    hint: plug in the teleop device and run `python -m robot_infra.discovery`
[2/5] os:hid ... FAIL  no HID device found
    hint: no /dev/hidraw* node: check the USB cable and port; inside Docker pass it with `docker run --device=/dev/hidrawN`
```

When a `/dev/hidraw*` node exists but this user cannot open it, `os:hid` fails with
`no read/write access to ['/dev/hidraw3']` and the hint
`add a udev rule SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666" to /etc/udev/rules.d/99-spacemouse.rules, then sudo udevadm control --reload && sudo udevadm trigger and replug; inside Docker pass the node with docker run --device=/dev/hidrawN`
(the vendor id in the rule is the 3Dconnexion one; check `lsusb` for your unit).

The Piper example with no CAN adapter plugged in (trimmed):

```text
[2/5] os:can_exists ... FAIL  CAN interface can0 not found
    hint: run `bash can_activate.sh can0 1000000` from piper_sdk, or `sudo ip link set can0 up type can bitrate 1000000`
[2/5] os:can_up ... FAIL  can0 operstate=missing
[2/5] os:can_bitrate ... FAIL  can0 bitrate=None, expected 1000000
[3/5] handshake ... FAIL  CAN interface 'can0' not found
    hint: plug in the USB-CAN adapter and run: bash can_activate.sh can0 1000000  (piper_sdk) or: sudo ip link set can0 up type can bitrate 1000000
```

Without pyAgxArm installed, the SDK hint reads `pip install 'rlgarden-robot-infra[pyagxarm]'  (pyAgxArm is installed from GitHub)`.
pyAgxArm is not on PyPI, so the extra installs it from GitHub.

#### Troubleshooting

The table quotes the messages and hints these checks print. `<ip>`, `can0` and similar
values are filled in from your config.

| Symptom | Where / what you see | What to do |
|---|---|---|
| Franka unreachable: FCI not activated, cable on the Arm port, firewall | `os:ping` FAIL; `handshake` FAIL `cannot reach Franka at <ip>: ...`, hint `check: ping <ip>; FCI activated in Desk; cable on the Control (not Arm) port; 'UDP receive: Timeout' means a firewall` | Activate FCI in Desk. Plug the PC into the Control box (not the arm base). Allow UDP from the robot through the firewall. |
| libfranka version mismatch | `handshake` FAIL `libfranka/robot firmware mismatch: ...`, hint `install the franky wheel matching your robot server version: pip install franky-control --extra-index-url https://timschneider42.github.io/franky/whl/by-robot-server-version/<N>/ ...` | Look up your robot server version in libfranka's compatibility matrix and install that wheel. `mode:libfranka_version` shows the installed libfranka. |
| No PREEMPT_RT kernel | with `realtime: enforce`: `mode:realtime_kernel` FAIL `realtime: enforce but the kernel is not PREEMPT_RT`, or `handshake` FAIL `libfranka realtime setup failed ...`; with `realtime: ignore` the check passes with `control jitter possible` | Install a PREEMPT_RT kernel and realtime group limits, or use `realtime: ignore`. |
| Franka in Reflex / UserStopped / Guiding | `mode:robot_mode` FAIL `robot mode is Reflex`, hint `release the external activation device / clear errors in Desk, then run with --fix (recover)` | Release the activation device, then run the doctor with `--fix`. |
| CAN interface missing / down / wrong bitrate | `os:can_exists`, `os:can_up`, `os:can_bitrate` FAIL (see output above); `handshake` FAIL `CAN interface 'can0' is down, not up` or `runs at N bit/s, expected 1000000` | `sudo ip link set can0 up type can bitrate 1000000` (or piper_sdk's `can_activate.sh can0 1000000`). |
| Piper does not enable (power, e-stop) | `handshake` FAIL `Piper on 'can0' did not enable within 5s`, hint `check that the arm is powered, that 'can0' is up at 1000000 bit/s, and that its emergency stop is released` | Power the arm, release the e-stop, rerun. `mode:piper_status` names the arm status once connected. |
| No CAN feedback (arm in master/teach mode) | `handshake` FAIL `... enabled but sent no joint feedback within 2s`, or `mode:follower` FAIL `no joint feedback`; hint `no CAN feedback: the arm may be in master/teach mode; switch it to follower (slave) mode and power-cycle` | Switch the arm to follower (slave) mode and power-cycle it. |
| Wrong Piper firmware profile (pyagxarm) | `handshake` FAIL `firmware probe ... failed` / `cannot map Piper firmware ...`, hint `set firmware_profile explicitly (default\|v183\|v188\|v189) in the robot config` | Set `firmware_profile` in the yaml. |
| Camera serial not found | `os:camera:<name>` FAIL `camera '<name>' serial '<s>' not found`, hint `set serial; discovered: [...]` (with `serial` unset: `no RealSense device found (serial unset)` or `serial unset and N devices attached; set serial to one of [...]`); `handshake` FAIL `RealSense serial '<s>' for camera '<name>' not attached` | Copy a serial from `python -m robot_infra.discovery`. With more than one camera attached, `serial` is required. |
| Camera gives no frames | `feedback:camera:<name>` FAIL `no frames within 2s`, hint `check USB 3 bandwidth; try another port; run with --fix to reopen`; or `RealSense '<s>' failed to start`, hint `check the USB 3 cable/port and udev rules (librealsense scripts/setup_udev_rules.sh)` | Use a USB 3 port and cable (not a hub shared with other cameras), install the librealsense udev rules, then try `--fix`. |
| Daemon unreachable | client `TransportError: robot daemon unreachable at <addr>` | Start the daemon, check `--bind`, the address and the firewall. |
| Token mismatch | client `TransportError: token mismatch`, hint `pass the same --token to the daemon and the remote config` | Use the same token on both sides. |
| Watchdog stopped the arm | client `DeviceFault: the daemon watchdog stopped the arm after 0.62s without actions (limit 0.5s)`, hint `... call set_control_mode(...) again to resume servoing, and keep the action rate above 1/watchdog_s` | Send actions faster than `1/watchdog_s`, or start the daemon with a larger `--watchdog-s`. Then call `set_control_mode` again. |
| Robot already in use | client `DeviceBusy: device 'franka:172.16.0.2' is claimed by another process`, hint `stop the other process using 'franka:172.16.0.2', or set `claim: false` in the robot config` | Stop the other process on this host that serves this robot through `RobotService` (a daemon or a `LocalClient` setup). Plain `Robot.from_yaml` scripts do not take the claim. To opt out deliberately, set `claim: false` in the yaml. |

### 4. Serve the robot

```bash
pip install -e '.[grpc]'   # plus the robot's backend extras
python -m robot_infra.server --config robot.yaml --bind 0.0.0.0:50051 [--token T] [--watchdog-s 0.5] [--no-preflight]
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | required | robot yaml |
| `--bind` | `0.0.0.0:50051` | `host:port` to listen on |
| `--token` | none | shared secret; clients must send the same one (gRPC metadata `x-robot-token`) |
| `--watchdog-s` | `0.5` | idle a servoing arm after this many seconds without actions |
| `--no-preflight` | off | skip the doctor run before serving |
| `--transport` | `grpc` | the only transport |
| `--jpeg-quality` | none | JPEG quality 1-100 for every camera's `color` image; see below |

`--jpeg-quality N` (needs `pip install -e '.[jpeg]'`, Pillow or OpenCV, on the daemon **and** on
every client) sends each camera's `color` image as a JPEG instead of raw bytes, which cuts the
bandwidth over WiFi. It is **lossy** and daemon-global: only `color` is compressed, `depth` and
all other fields stay raw, and the client decodes back to a `uint8 (H, W, 3)` array. The policy
therefore sees compressed pixels, so **train and deploy with the same value** (or none). The
value is reported by `robot.describe` as the top-level `image_encoding` (`{"jpeg": 80}`, or `null`
when raw) and as `robot.cameras["wrist"].image_encoding` on the client. A client without
Pillow/OpenCV fails at `connect()` with `SdkMissing: JPEG frames need Pillow or OpenCV`
(hint `pip install "rlgarden-robot-infra[jpeg]"`).

Daemon and client ship from the same repo; upgrade the **client first, then the daemon**. A new
daemon's `arm.health` carries two extra fields (`safety_clips`, `safety_rejects`) that a
pre-upgrade client cannot parse, and enabling `--jpeg-quality` needs a client that can decode
JPEG. A client that predates JPEG support cannot decode the frames (its error was not checked).

What happens:

1. **Config.** The yaml is loaded. Then every config field that is empty (`""` or null) is
   filled from the environment variable of the same name in upper case, for example
   `FIRMWARE_PROFILE=v188`. `ROBOT_IP` and `CAN_INTERFACE` apply only if the yaml sets
   `robot_ip: ""` / `can_interface: ""`, because both have non-empty defaults. A camera
   with an empty `serial` takes the serial when exactly one camera of its backend is
   attached. The daemon, the doctor and `python -m robot_infra.discovery --config` all apply
   this autofill; `Robot.from_yaml` reads the yaml as written.
2. **Preflight.** Unless you pass `--no-preflight`, the full doctor (not `--mock`) runs
   and prints its report. If any check fails, the daemon prints `preflight failed; fix the
   items above or start with --no-preflight` and exits with code 1. The doctor connects
   and disconnects the robot while holding its device claim, so a running daemon on the same
   robot is reported as a failed `handshake`. Fake backends skip the OS checks
   (`backend 'fake' needs no OS device`), so demo or fake configs start without
   `--no-preflight`.
3. **Startup output** (from a fake-backend config):

   ```text
   franka (franka)
     arm: FakeArm [sdk=None, connected=False]
     gripper: FakeGripper [sdk=None, connected=False]
     cameras: PartGroup [sdk=None, connected=False]
       wrist: FakeCamera [sdk=None, connected=False]
     teleop: FakeTeleop [sdk=None, connected=False]
   serving franka 'franka' on 127.0.0.1:50061 (token required)
   ```

   The robot is connected only when a client calls `connect()`. At that point the daemon
   takes the device claim, and it releases the claim on `disconnect()`.
4. **Watchdog.** While the arm is in `CARTESIAN_SERVO` or `JOINT_SERVO`, the daemon checks
   every `watchdog_s / 4` how long it has been since the last action. Actions are
   `send_action`, `step`, `servo_cartesian` and `servo_joints`; entering servo mode also
   starts the clock. After more than `watchdog_s` with no action, the daemon calls
   `arm.stop()`, switches the arm to `IDLE`, and logs `watchdog: no action for 0.624s in
   joint_servo mode; stopping arm 'arm' and idling it`. Later servo commands raise the
   watchdog `DeviceFault` until the client calls `set_control_mode(...)` again.
   `move_to_*` calls (which hardware backends allow only in `IDLE`) are not watched.
5. **Teleop.** If the yaml has a `teleop:` section the daemon serves `teleop.<name>.read` and
   `teleop.<name>.reset`. Reads do not refresh the watchdog, and a client's `timeout_s` is
   clamped to 2 s so a client cannot park a gRPC worker thread. A `teleop` section is
   mandatory once present: `connect()` fails when the device is absent and takes the whole
   robot down with it, so a deployment that needs no teleop should use a yaml without the section.
6. **Ctrl-C / SIGTERM.** The daemon stops the transport, stops the watchdog, disconnects the
   robot and releases the claim. It never powers the robot off and never disables the
   motors (see [Safety rules](#safety-rules)).

`robot.describe` is what a client builds its mirror from. For the fake robot above (trimmed):

```text
arm: {"num_joints": 7, "supported_state_fields": ["tcp_pose", "tcp_vel", "joint_pos", "joint_vel", "tcp_force", "tcp_torque"], "control_mode": "idle", "safety": {"safety_box": [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], "joint_limits": [[-1.0, 1.0], ...], "max_delta_per_step": {"xyz_m": 0.05, "joint_rad": 0.1}}, "home_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
image_encoding: {'jpeg': 80}
camera cfg: {'name': 'wrist', 'serial': '', 'width': 64, 'height': 48, 'fps': 30, 'enable_depth': False, 'backend': 'fake'}
teleop: {'name': 'teleop', 'backend': 'fake', 'action_space': 'joint_abs', 'has_gripper': True, 'hold_s': 0.0}
```

`arm.safety` is `null` when no limit is configured; `arm.home_joints` is the yaml's value (or `null`).

### 5. Use the robot from the GPU machine

```bash
pip install -e '.[grpc]'
```

```python
import robot_infra.server  # registers robot type "remote"
from robot_infra.devices import ControlMode, Robot

robot = Robot.from_dict({"type": "remote", "address": "robot-pc:50051", "token": "T"})
robot.connect()                       # builds arm/gripper/cameras from the daemon's describe
print(robot.describe_text())
robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
q = robot.arm.read_state()["joint_pos"]
obs = robot.step({"arm": {"joint_pos": q}, "gripper": {"gripper": [1.0]}})  # action, then observation
print(robot.health())                 # Health(ok=True, detail='rtt=1.2 ms, watchdog_trips=0', ...)
robot.arm.set_control_mode(ControlMode.IDLE)
robot.disconnect()
```

Without `import robot_infra.server`, `Robot.from_dict` raises
`unknown robot type 'remote'; registered: ['franka', 'piper']`.

- Actions are nested per part, and the arm key depends on the control mode: `tcp_pose`
  (7 values: xyz plus quaternion xyzw) in `CARTESIAN_SERVO`, `joint_pos` in
  `JOINT_SERVO`. The gripper takes a normalized width in [0, 1].
- `step()` returns `{"arm": {...}, "gripper": {"gripper_pos": ...}, "cameras": {<name>:
  {"color": ...}}}`. The arm fields are the backend's `SUPPORTED_STATE_FIELDS`.
- Other `RemoteRobotConfig` keys: `timeout_s` (5.0), `reconnect_attempts` (5),
  `reconnect_backoff_s` (0.5). After a dropped connection, the client reconnects and
  retries only read-only calls. Commands are never retried: reconnect and set the control
  mode again.
- For single-machine use without gRPC, wrap a local robot in-process:

  ```python
  from robot_infra.devices import Robot
  from robot_infra.server import RemoteRobot, RobotService
  from robot_infra.server.transport import LocalClient

  service = RobotService(Robot.from_yaml("robot.yaml"), watchdog_s=0.5)
  robot = RemoteRobot.from_client(LocalClient(service))   # same API as above, incl. watchdog
  ```

- The daemon's device-side facts are mirrored on the client:
  `robot.arm.safety_limits` (the dict under `arm.safety` above, or `None`),
  `robot.arm.home_joints` (list or `None`), `robot.cameras["wrist"].image_encoding`, and
  `robot.teleop` (`None` without a `teleop:` section). The client does not enforce limits
  itself (`RemoteRobotConfig` ignores `safety_box`, `joint_limits`, `max_delta_per_step` and
  `teleop`); the daemon does.
- A command outside the limits behaves as follows on the client (run it before `robot.disconnect()`):

  ```python
  import numpy as np
  from robot_infra.devices import SafetyViolation

  robot.arm.set_control_mode(ControlMode.JOINT_SERVO)
  robot.arm.servo_joints(np.full(7, 5.0))      # no error: clipped on the daemon, safety_clips += 1
  robot.arm.set_control_mode(ControlMode.IDLE)
  try:
      robot.arm.move_to_joints(np.full(7, 5.0))   # outside joint_limits
  except SafetyViolation as error:
      print(error.message, error.hint)            # nothing was sent; the arm and its mode are unchanged
  print(robot.arm.health().safety_clips, robot.arm.health().safety_rejects)   # cumulative counters since the daemon started, e.g. 2 1
  print(robot.remote_health()["arm"]["safety_clips"])                         # the same counters from the daemon's robot.health
  ```

  `SafetyViolation` is a `DeviceError` (not a `DeviceFault`): it does not latch a fault, does
  not need `recover()`, and does not refresh the daemon watchdog. `robot.health()` is a
  `Health`; the per-arm counters live in `robot.arm.health()` and `robot.remote_health()`.
- `rlgarden-real-world` wraps this robot in a gym environment.

### 6. Use the robot locally (no daemon)

On the robot PC, you can drive the device layer directly:

```python
import numpy as np
from robot_infra.devices import ControlMode, Robot

robot = Robot.from_yaml("robot.yaml")
robot.connect()
robot.arm.set_control_mode(ControlMode.CARTESIAN_SERVO)
pose = robot.arm.read_state()["tcp_pose"]
robot.send_action({"arm": {"tcp_pose": pose}, "gripper": {"gripper": np.array([0.0])}})
obs = robot.get_observation()
robot.arm.set_control_mode(ControlMode.IDLE)
robot.disconnect()
```

This path has no watchdog and takes no device claim. Those are daemon features. The safety
limits, however, are enforced here too: `Robot.from_yaml` builds the arm with the yaml's
`safety_box`/`joint_limits`/`max_delta_per_step`, so `robot.arm.safety_limits`,
`robot.arm.health().safety_clips` and `SafetyViolation` behave exactly as through the daemon.

### 7. Add a teleop device

A `teleop:` section makes an operator input device a part of the robot. Hardware-free, with
the fake backend (a script advanced one step per `read()`):

```yaml
teleop:
  name: teleop                # the daemon serves teleop.<name>.read / .reset
  backend: fake               # fake | spacemouse
  hold_s: 0.0                 # keep intervening this long after the last activity (default 0)
  options:                    # backend-private
    action_space: joint_abs   # fake: cartesian_delta (default) | joint_abs | joint_delta
    num_joints: 7             # joint spaces only
    has_gripper: true         # default true
    on_end: hold_last         # hold_last (default) | idle | loop, what happens after the script
    script:
      - {action: [0, 0, 0, 0, 0, 0, 0], active: false, hold: 2}
      - {action: [0.5, 0, 0, 0, 0, 0, 0], gripper: 0.0, active: true, hold: 2}
      - {events: {episode_end: true}}
```

A step is `{action, gripper, buttons, events, active, hold}`; `hold` is the number of reads the
step lasts and its events fire on the first of them. `fail_after: n` makes every poll after the
n-th sample fail, to test the error path.

```python
robot = Robot.from_yaml("robot.yaml")          # local; RemoteRobot works the same way
robot.connect()
sample = robot.teleop.read()                   # blocks for a new sample, consumes the events
print(sample.seq, sample.action_space, sample.action, sample.gripper, sample.intervening, sample.events["episode_end"])
latest = robot.teleop.latest()                 # no blocking, does not consume events
robot.teleop.reset()                           # drops latched events and the hold_s hysteresis
```

`read()` returns a `TeleopSample`; over the daemon (`teleop.<name>.read`) it travels as a dict
with the same keys:

| Field | Type | Meaning |
|---|---|---|
| `action` | `float32` ndarray | `cartesian_delta`: `(6,)` normalized twist in [-1, 1]; `joint_abs` / `joint_delta`: `(N,)` joint values in rad |
| `action_space` | `str` | `cartesian_delta`, `joint_abs` or `joint_delta` |
| `gripper` | `float` or `None` | normalized width target in [0, 1]; `None` = the device has no gripper channel |
| `buttons` | `dict[str, bool]` | current levels (SpaceMouse: `button_0`, `button_1`) |
| `events` | `dict[str, bool]` | latched edges `episode_end`, `success`, `failure`, `rerecord` (always all four keys); `read()` clears them |
| `intervening` | `bool` | a human is operating (backend's `active`, extended by `hold_s`) |
| `timestamp` | `float` | `time.monotonic()` on the machine that polled, not comparable across hosts |
| `seq` | `int` | increases with every polled sample, also across reconnects; a gap means dropped samples |

`read(timeout_s)` raises `DeviceFault` on a timeout and a recoverable `DeviceFault` once 5 polls
in a row failed (reconnect the robot); it raises `DeviceNotConnected` if the robot is
disconnected while it waits. There is a single consumer: two clients reading the same teleop
split the events between them.

`backend: spacemouse` (extra `spacemouse`; options `device_index` 0, `deadzone` 1e-3,
`invert_axes`, `poll_interval_s` 0.005) is a `cartesian_delta` device with a gripper: the puck
gives the twist `[-y, x, z, -roll, -pitch, -yaw]`, button 0 toggles the sticky gripper target
(1.0 = open at connect), button 1 raises `episode_end`. It needs read/write access to
`/dev/hidraw*`; `python -m robot_infra.doctor` checks this as `os:hid` and prints the udev
rule. See [Known limitations](#known-limitations-and-needs-hardware-verification).

## Architecture

```text
Connection (ABC)                 connect()/disconnect() lifecycle, owner borrowing, SDK, health()
└── Part                         observation_features, get_observation()
    ├── Camera   [registry]      background capture thread, read()/latest()/reopen()
    │   ├── realsense, fake
    │   └── (RemoteCamera is a Part, not a Camera)
    ├── Teleop   [registry]      background poll thread, read()/latest()/reset(), latched events
    │   ├── spacemouse, fake
    │   └── (RemoteTeleop is a Part, not a Teleop)
    └── ControllablePart         action_features, send_action()
        ├── Arm      [registry]  read_state, servo_cartesian/joints, move_to_pose/joints
        │   │                    (template methods, see below), stop, recover, set_control_mode, preflight
        │   ├── franky, serl_http, pyagxarm, piper_sdk, fake
        │   └── RemoteArm
        ├── Gripper  [registry]  open/close/move(width in [0, 1]), width; binary | continuous
        │   ├── franky, serl_http, pyagxarm, piper_sdk, fake
        │   └── RemoteGripper
        └── PartGroup            named children; connect in order, disconnect in reverse
            └── Robot  [type registry]  config -> arm (+ gripper) (+ cameras group)
                ├── FrankaRobot "franka"   (FrankaConfig)
                ├── PiperRobot  "piper"    (PiperConfig)
                └── RemoteRobot "remote"   (RemoteRobotConfig; registered by importing robot_infra.server)

Robot PC:   Robot ── RobotService (method table, watchdog, DeviceClaim) ── GrpcTransport
GPU side:   RemoteRobot ── GrpcClient (msgpack over gRPC, token metadata, read-only retry)
```

Backends register by name (`Arm.register("franky")`). `Arm.backends()`, `Gripper.backends()`,
`Camera.backends()` and `Teleop.backends()` list them, and `Robot.types()` lists robot types. A gripper with
`shares_arm_session: true` borrows the arm's session as its `owner` instead of opening its
own.

| Backend | SDK (extra) | Joint control | Wrench (`tcp_force`/`tcp_torque`) | Notes |
|---|---|---|---|---|
| arm `franky` | `franky` (`franky`) | yes | yes | torque-level impedance trackers for servo modes, Ruckig `move_to_*` (`IDLE` only); `realtime` and `relative_dynamics_factor` apply here |
| arm `serl_http` | none (HTTP) | no | yes | legacy client of the ROS1 bridge at `bridge_url`; Cartesian servo only; state has no `joint_pos`/`joint_vel` |
| arm `pyagxarm` | `pyAgxArm` (`pyagxarm`) | yes | no | Piper default; `firmware_profile` probed or set; state `tcp_pose`, `joint_pos`, `joint_vel` (flange pose) |
| arm `piper_sdk` | `piper_sdk` (`piper-sdk`) | yes | no | vendor SDK, fixed at 1 Mbit/s; `stop()` holds by re-sending the measured joints |
| arm `fake` | none | yes | yes | deterministic first-order arm for tests and `--mock` |
| gripper `franky` | `franky` | - | - | Franka Hand; own session |
| gripper `serl_http` | none | - | - | binary only, through the bridge |
| gripper `pyagxarm` / `piper_sdk` | as the arm | - | - | requires `shares_arm_session: true` and the matching arm backend |
| camera `realsense` | `pyrealsense2` (`realsense`) | - | - | RGB (+ aligned depth in metres with `enable_depth`) |
| `fake` gripper / camera | none | - | - | tests and `--mock` |
| teleop `spacemouse` | `pyspacemouse>=2,<3` (`spacemouse`) | - | - | cartesian twist + sticky gripper; `Teleop.backends()` is `fake`, `spacemouse` |
| teleop `fake` | none | - | - | scripted samples, `push()` injection, `fail_after`; tests and `--mock` |

**`Arm` is a template-method class.** The public `servo_cartesian`, `servo_joints`,
`move_to_pose` and `move_to_joints` sanitize the input (shape, finiteness, unit quaternion),
apply the safety limits and only then call the backend hooks `_servo_cartesian`,
`_servo_joints`, `_move_to_pose` and `_move_to_joints`. A backend implements the hooks (plus
`read_state`, `stop`, `recover`, `_enter_control_mode`) and never overrides the public
methods, so no path to the motors skips the guard. A backend's `read_state()` must return its
result through `self._remember_state(state)`: the step cap reuses that cached state (at most
`MAX_DELTA_STATE_AGE_S` = 0.5 s old, else it re-reads) instead of querying the robot on every
servo call. `Arm.from_config(config)` builds the backend with `ArmSafety.from_config(config)`.
A third-party backend that skips `_remember_state` still works but costs one extra
`read_state()` per capped servo call.

### Safety rules

- **`disconnect()` never powers off.** Releasing an arm only closes the SDK session. The
  Piper backends never call `disable()`/`reset()`/`DisableArm`/`ResetPiper`, because a
  powered-down Piper falls. `stop()` brings motion to rest and leaves power and brakes as
  they are.
- **Servo commands are refused when the arm is not ready.** Every backend refuses servo
  commands outside the matching servo mode. franky, pyagxarm and piper_sdk also refuse them
  while the arm is faulted (call `recover()` first). The Piper backends also refuse on stale
  feedback: no new joint frame for 0.5 s on pyagxarm, 1.0 s on piper_sdk. On the hardware
  backends, `move_to_*` requires `IDLE`. Actions are checked for shape and finiteness, and
  `tcp_pose` quaternions are checked for unit norm.
- **Connecting enables the motors.** Connecting a Piper (directly, through the doctor, or
  through the daemon when a client connects) enables its motors, and they stay enabled
  after `disconnect()`. Clear the workspace before running the doctor.
- **Safety limits.** `safety_box`, `joint_limits` and `max_delta_per_step` in the robot yaml on
  the robot host are the source of truth. `Arm` enforces them in the public command methods, so
  they hold on every path: local `Robot`, the daemon (`arm.servo_*`, `arm.move_to_*`,
  `robot.send_action`, `robot.step`) and `send_action`. A `RemoteArm` has no guard of its own and
  forwards to the daemon's.
  - `servo_*`: the target is first shortened to `max_delta_per_step` from the measured state, then
    clipped into the box / joint limits. Any change counts one `safety_clips` and the command is
    still sent.
  - `move_to_*`: a target outside the box / joint limits raises `SafetyViolation`, counts one
    `safety_rejects`, and nothing is sent. `move_to_*` is not step-capped.
  - `SafetyViolation` is not a device fault: it is raised before the backend is touched, so it
    cannot latch a fault, needs no `recover()`, keeps the control mode, and through the daemon it
    does not refresh the watchdog. A client sees the same class.
  - `arm.health()` reports `safety_clips` / `safety_rejects`; `robot.describe` exposes the limits
    as `arm.safety`. The checks are validated at build time: `home_joints` must lie inside
    `joint_limits`, a box must have min <= max.
  - Limits apply to the position of `tcp_pose` only. The orientation is not limited and not step
    capped (see [Known limitations](#known-limitations-and-needs-hardware-verification)).
- **Watchdog.** The daemon idles a servoing arm whose client went quiet (see step 4).
- **`DeviceClaim`.** An advisory, node-local file lock in `~/.cache/robot_infra/claims/`,
  keyed `<type>:<robot_ip | can_interface | name>`. The daemon holds it while a client is
  connected, so a second daemon on the same robot gets `DeviceBusy`. It does not protect
  against other hosts or against code that does not take the claim. Set `claim: false`
  to opt out.

### Error model

Device errors subclass `DeviceError` (a `RuntimeError`) and carry `message`, `hint` and
`device`. `str(error)` prints the hint on a second line. Errors survive the
`to_dict()`/`from_dict()` round trip, so a client re-raises the same type the daemon raised.

| `kind` | Class | Typical cause |
|---|---|---|
| `sdk_missing` | `SdkMissing` | vendor package not installed |
| `device_not_found` | `DeviceNotFound` | robot unreachable, CAN interface or camera serial missing |
| `device_not_connected` | `DeviceNotConnected` | call before `connect()` |
| `device_already_connected` | `DeviceAlreadyConnected` | defined; not raised by current backends (`connect()` is idempotent) |
| `device_busy` | `DeviceBusy` | another process holds the claim |
| `device_fault` | `DeviceFault` (`recoverable: bool`) | reflex, e-stop, no feedback, wrong mode, watchdog |
| `version_mismatch` | `VersionMismatch` | libfranka vs robot server version; a `pyspacemouse` without the 2.x API (`open(device_index=...)`) |
| `safety_violation` | `SafetyViolation` | a `move_to_*` target outside `safety_box` / `joint_limits`; message lists the offending axes, hint `change the target, or the limits in the robot yaml on the robot host` |
| `transport` | `TransportError` (a `DeviceFault`) | daemon unreachable, timeout, token mismatch |

## Testing

```bash
pip install -e '.[dev,grpc]'
python -m pytest tests -q
```

The tests use the fake backends and SDK stubs. No hardware is needed. The gRPC tests are
skipped when `grpcio` is missing. Last run (remote container, Python 3.11): `486 passed`.
The maintainers run them in a remote container:

```bash
docker exec <container> bash -lc 'cd <checkout>/robot_infra && python -m pytest tests -q'
```

## Known limitations and needs hardware verification

Nothing below has been exercised on real hardware; the fakes cover the logic only.

- **SpaceMouse USB vendor ids.** `discover()` filters on `0x256f` (3Dconnexion) and `0x046d`
  (Logitech, older units). Both are unverified; without `hidapi` it lists every `/dev/hidraw*`.
- **udev and Docker.** The udev rule printed by `os:hid` and the `docker run --device=/dev/hidrawN`
  hint are untested on a robot PC. A container needs the node passed in before the daemon starts.
- **Does `pyspacemouse` `read()` block?** The backend polls at `poll_interval_s` (default 0.005 s)
  so a non-blocking `read()` does not spin a core; set it to `0` only after confirming that
  `read()` blocks on your pyspacemouse version.
- **Piper master arm** as a teleop (leader-follower) is not implemented; only `fake` and
  `spacemouse` exist. The `joint_abs` / `joint_delta` action spaces are exercised by the fake only.
- **Orientation is not limited.** `safety_box` limits the position, `max_delta_per_step` the
  position (`xyz_m`) and joints (`joint_rad`) only. A client that servos `tcp_pose` directly can
  command a large rotation in one step (the gym env bounds it with its own `rot_scale_rad`).
- **Step cap baseline.** The cap is measured from the last read state (at most 0.5 s old), not
  from the previous target, so the worst distance between a target and the true position is about
  twice the cap, without accumulating.
- **Bypassing the guard.** The legacy ROS1 bridge (`serl_http`) is an HTTP server: a client
  that talks to `bridge_url` directly does not pass through `Arm`, so none of the limits apply.
  Third-party backends must implement the `_servo_*`/`_move_*` hooks, not override the public methods.
- **A configured teleop is mandatory.** `connect()` fails and rolls the whole robot back when the
  teleop device is absent, including for deployment. Use a yaml without `teleop:` to deploy.
- **Version skew.** A daemon with these features breaks a pre-upgrade client at `arm.health()`
  (new fields), and an old client cannot read JPEG frames. Upgrade the client first.
- **`--jpeg-quality` is not recorded** in a training run's config; deploy cannot check that it
  matches the value used for training.
- **Teleop read clamp and clocks.** `teleop.<name>.read` over the daemon waits at most 2 s;
  `timestamp` is the robot PC's monotonic clock. `RemoteTeleop.get_observation()` returns the
  client's last sample, while the daemon's `robot.get_observation()` contains a `teleop` entry
  taken from the daemon's latest sample (events not consumed).

## Legacy

These modules predate the device layer and are kept for existing users. Planned work is in
[`docs/robot-infra-roadmap.md`](docs/robot-infra-roadmap.md).

- **`controller/`**: pure-torch Cartesian impedance and pose controllers (`core/`, `adapters/`),
  wrapped for ManiSkill in `controller/simulator/maniskill/` (extra `sim`). They serve
  GPU-batched simulation, not real hardware. See [`robot_infra/controller/README.md`](robot_infra/controller/README.md).
- **`controller/real/franka_bridge.py`** and **`gripper_server.py`**: a Flask/HTTP forwarder to
  the ROS1 `serl_franka_controllers` impedance controller and the Franka Hand. The `serl_http`
  backend is their client. Run the bridge on the ROS machine with
  `python -m robot_infra.controller.real.franka_bridge --robot_ip <ip> --port 5000` (extra
  `real`; ROS1 is installed separately, see the module docstring).
- **`teleop/`**: the pre-device-layer teleop (`teleop/spacemouse/` on the old `pyspacemouse`
  `read_all()` API, extra `real`; `teleop/utils/` twist teleop for Pico over ZeroMQ;
  `teleop/examples/`). The `Teleop` part ([step 7](#7-add-a-teleop-device)) supersedes the
  SpaceMouse code. The Pico path is not migrated yet and `rlgarden-real-world`'s
  `--teleop-device pico|spacemouse` still imports these modules, so they stay until it is.
