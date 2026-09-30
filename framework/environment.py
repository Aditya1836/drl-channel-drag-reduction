"""


Each wall cell is an independent mini-agent with 2 states (u, v) and 1 action.
The same neural network is shared across all 1064 wall cells.
"""

# from mpi4py import MPI  # Breaks nested mpirun
import gymnasium as gym
from gymnasium import spaces
import subprocess
import numpy as np
from pathlib import Path
import yaml
import os, time


class OFEnvironment(gym.Env):
    """
    Custom OpenFOAM Environment for SB3 - Channel Flow Control.
    Uses parameter sharing: observation_space=(2,), action_space=(1,).
    """

    def __init__(self, config_file, rank):
        super(OFEnvironment, self).__init__()
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Automatically assign variables from config to instance attributes
        for key, value in config.items():
            setattr(self, key, value)

        self.base_case = Path(self.name) / self.base_case
        self.base_policy_dir = self.base_case / self.policy_dir
        self.episode_number = -1
        self.case_number = rank
        self.run_case = None
        self.start_time = 0
        self.end_time = self.action_time_step * self.n_actions
        self.n_actions_all = self.n_actions * self.n_actuators

        # Parameter sharing: each wall cell sees 2 states (u, v) and outputs 1 action
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_states,),
            dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.episode_number += 1
        self.run_case = Path(self.name) / Path(
            self.run_cases + f"{self.case_number}" + "_ep" + str(self.episode_number)
        )

        # Clean up and prepare a fresh case
        self.clean_up()
        subprocess.run(f"cp -r {self.base_case} {self.run_case}", shell=True)
        self.converged = False

        # Preprocessing step
        log_path = self.run_case / "log.Allrun.pre"
        with open(log_path, "a+") as log_file:
            subprocess.run(
                f"./{self.run_case}/Allrun.pre",
                shell=True,
                stdout=log_file,
                stderr=log_file,
            )

    def clean_up(self):
        """
        Remove any previous case data.
        """
        if self.run_case and self.run_case.exists():
            subprocess.run(f"rm -rf {self.run_case}", shell=True)

    def run(self):
        """
        Run OpenFOAM for the entire episode, applying the RL policy.
        """
        self.configure_case()
        self.run_OpenFOAM()

    def postprocess(self):
        self.read_results()
        self.compute_reward()
        return (
            self.observation_values,
            self.action_values,
            self.reward_values,
            self.terminals,
        )

    def configure_case(self):
        self.set_foam_dict("system/controlDict", "endTime", self.end_time)
        self.set_foam_dict("system/controlDict", "writeInterval", self.end_time)

    def run_OpenFOAM(self):
        """
        Execute OpenFOAM for a full episode, with support for parallel simulations.
        """
        if self.parallel:
            # Get the number of subdomains
            output = subprocess.check_output(
                [
                    "foamDictionary",
                    "-entry",
                    "numberOfSubdomains",
                    "-value",
                    self.run_case / "system/decomposeParDict",
                ]
            )
            self.n_subdomains = int(output.split(b"\n")[0])

            # Run mpirun directly (Allrun scripts unreliable from Python subprocess)
            logf = open(self.run_case / "log.pimpleFoam", "w")
            cmd = f"cd {self.run_case} && mpirun -n {self.n_subdomains} pimpleFoam -parallel"
            subprocess.Popen(cmd, shell=True, stdout=logf, stderr=logf).wait()
            logf.close()
        else:
            logf = open(self.run_case / "log.Allrun", "a+")
            Allrun = f"./{self.run_case}/Allrun.run"
            subprocess.Popen(Allrun, shell=True, stdout=logf, stderr=logf).wait()

    def read_results(self):
        """
        Read simulation results from postProcessing directories.

        ActionState.dat format (from channelDRLControl C++ library):
        - Column 0: time
        - Then for each actuator j (j=0..1063):
            Column 1+3j: action_j
            Column 2+3j: state_u_j (streamwise velocity at ySample)
            Column 3+3j: state_v_j (wall-normal velocity at ySample)

        Friction data from surfaceFieldValue.dat:
        - Column 0: time
        - Column 1: area-averaged skin friction coefficient
        """
        start_time_str = (str(self.start_time) + "/").replace(".0/", "/")

        # Read friction coefficient data
        friction_path = os.path.join(
            self.run_case, "postProcessing", "friction",
            start_time_str, "surfaceFieldValue.dat"
        )

        try:
            times = []
            tau_x = []
            with open(friction_path) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('(')
                    t = float(parts[0].strip())
                    vec = parts[1].rstrip(')').split()
                    times.append(t)
                    tau_x.append(float(vec[0]))
            times = np.array(times)
            tau_x = np.array(tau_x)
            cf_vals = np.abs(tau_x) / (0.5 * self.U_bulk ** 2)
            cf_values = np.column_stack([times, cf_vals])
            if cf_values[-1, 0] == self.end_time:
                self.converged = True
            else:
                self.converged = False
        except Exception as e:
            print(f"Friction read error: {e}")
            self.converged = False

        if self.converged:
            # Read ActionState.dat from the DRL control BC
            episode_data_path = os.path.join(
                self.run_case,
                "postProcessing",
                "channelDRLControl",
                start_time_str,
                "ActionState.dat",
            )
            episode_data = np.loadtxt(episode_data_path, skiprows=2)

            # Extract states, actions, and friction for each actuator at each timestep
            state_values = []
            action_values = []
            cf = []

            for i in range(self.n_actions):
                # Friction: same reward for all actuators at this timestep
                cf.append(np.repeat(cf_values[i, 1], self.n_actuators))

                # Actions: every 3rd column starting from 1
                action_values.append(episode_data[i, 1::3])

                # States: (u, v) for each actuator
                for j in range(self.n_actuators):
                    state_values.append(episode_data[i, j * 3 + 2 : j * 3 + 4])

            self.cf = np.hstack(cf)
            self.action_values = np.hstack(action_values)
            self.observation_values = np.vstack(state_values)

            # Terminals: mark last timestep for each actuator as terminal
            self.terminals = np.zeros(self.n_actions_all, dtype=bool)
            # Mark the last n_actuators transitions as terminal
            # self.terminals[-self.n_actuators:] = True

        else:
            self.observation_values = np.empty((0, self.n_states), float)
            self.action_values = np.empty((0), float)
            self.cf = np.empty((0), float)
            self.terminals = np.empty((0), bool)

    def compute_reward(self):
        """
        Compute reward based on skin friction reduction.
        reward = 1 - Cf/Cf_uncontrolled

        If Cf < Cf_uncontrolled: positive reward (drag reduced)
        If Cf = Cf_uncontrolled: reward = 0 (no improvement)
        If Cf > Cf_uncontrolled: negative reward (drag increased)
        """
        if "friction" in self.reward_function:
            tau_w_uncontrolled = self.u_tau ** 2
            cf_uncontrolled = tau_w_uncontrolled / (0.5 * self.U_bulk ** 2)
            self.reward_values = 1 - self.cf / cf_uncontrolled
        else:
            raise RuntimeError(
                "reward function {} not implemented".format(self.reward_function)
            )

    def set_foam_dict(self, dict_file, entry, value):
        """
        Modify OpenFOAM dictionary files using foamDictionary.
        """
        command = (
            f"foamDictionary {self.run_case}/{dict_file} -entry {entry} -set {value}"
            f" -disableFunctionEntries"
        )
        subprocess.run(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
