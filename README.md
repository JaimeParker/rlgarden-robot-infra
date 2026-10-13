# rlgarden-robot-infra

`robot_infra` is the real-robot layer of [rl-garden](https://github.com/JaimeParker/rl-garden).
It provides:

- a **device layer**: typed `Arm` / `Gripper` / `Camera` contracts with backends for
  Franka (franky, legacy ROS1 bridge), AgileX Piper (pyAgxArm, piper_sdk), and Intel RealSense;
- **discovery** of attached cameras, CAN interfaces, robot hosts, and HID devices;
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
| `grpc` | `grpcio>=1.60` | the daemon (`python -m robot_infra.server`) and `RemoteRobot` | robot PC **and** GPU machine |
| `real` | `flask`, `pyzmq`, `scipy`, `pyspacemouse` | legacy Flask bridge/gripper server and SpaceMouse teleop (ROS1 must be installed separately) | robot PC |
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
# safety_box: [xmin, ymin, zmin, xmax, ymax, zmax]          # validated, not enforced yet
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
claim: true
gripper:
  backend: pyagxarm           # must match the arm backend: pyagxarm | piper_sdk
  max_width_m: 0.07
  mode: continuous            # Piper default
  binary_threshold: 0.5
  shares_arm_session: true    # required: the Piper gripper borrows the arm's CAN session
cameras: []
```

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
can interfaces:
  none
hosts:
  172.16.0.2: ping=no reply open ports=none
hid devices:
  none
```

With hardware attached, the lines take the form `realsense: <serial>, <serial>`,
`can0: operstate=up bitrate=1000000`, `172.16.0.2: ping=ok open ports=<list>` and
`/dev/hidraw0`.

| Section | Source | What to look for |
|---|---|---|
| `cameras` | each camera backend's `discover()` (RealSense serials) | copy the serial into `cameras[].serial` |
| `can interfaces` | `/sys/class/net/*` with type CAN, bitrate from `ip -details link show` | `operstate=up` and the bitrate in your yaml |
| `hosts` | `ping` and TCP connect to the config's `robot_ip` | `ping=ok`. For Franka, 1337 open usually means FCI is reachable. |
| `hid devices` | `/dev/hidraw*` | SpaceMouse and other HID teleop devices |

Without `--config`, `hosts:` prints `none`.

### 3. Check with the doctor

```bash
python -m robot_infra.doctor --config robot.yaml [--mock] [--fix] [--json]
```

The doctor runs five layers in order:

| Layer | Checks | Notes |
|---|---|---|
| 1 `sdk` | each backend name resolves and its SDK imports | an unknown backend skips layers 3-5 |
| 2 `os` | ping `robot_ip` and port 1337 (informational); CAN interface exists / is up / bitrate; camera serial attached | driven by config fields; skipped for backends that need no SDK (fakes) |
| 3 `handshake` | takes the robot's device claim (unless `claim: false`), then `robot.connect()` | a failure (including a held claim) skips layers 4-5 |
| 4 `feedback` | the arm returns state; the gripper is readable; each camera delivers a frame within 2 s | |
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
| `--jpeg-quality` | none | accepted, but frames are not JPEG-encoded yet |

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
5. **Ctrl-C / SIGTERM.** The daemon stops the transport, stops the watchdog, disconnects the
   robot and releases the claim. It never powers the robot off and never disables the
   motors (see [Safety rules](#safety-rules)).

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

This path has no watchdog and takes no device claim. Those are daemon features.

## Architecture

```text
Connection (ABC)                 connect()/disconnect() lifecycle, owner borrowing, SDK, health()
└── Part                         observation_features, get_observation()
    ├── Camera   [registry]      background capture thread, read()/latest()/reopen()
    │   ├── realsense, fake
    │   └── (RemoteCamera is a Part, not a Camera)
    └── ControllablePart         action_features, send_action()
        ├── Arm      [registry]  read_state, servo_cartesian/joints, move_to_pose/joints,
        │   │                    stop, recover, set_control_mode, preflight
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

Backends register by name (`Arm.register("franky")`). `Arm.backends()`, `Gripper.backends()`
and `Camera.backends()` list them, and `Robot.types()` lists robot types. A gripper with
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
| `version_mismatch` | `VersionMismatch` | libfranka vs robot server version |
| `transport` | `TransportError` (a `DeviceFault`) | daemon unreachable, timeout, token mismatch |

## Testing

```bash
pip install -e '.[dev,grpc]'
python -m pytest tests -q
```

The tests use the fake backends and SDK stubs. No hardware is needed. The gRPC tests are
skipped when `grpcio` is missing. The maintainers run them in a remote container:

```bash
docker exec <container> bash -lc 'cd <checkout>/robot_infra && python -m pytest tests -q'
```

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
- **`teleop/`**: SpaceMouse teleop (`teleop/spacemouse/`, extra `real`) and twist-based
  teleop utilities (`teleop/utils/`, `teleop/examples/`). They are not yet integrated with
  the device layer.
