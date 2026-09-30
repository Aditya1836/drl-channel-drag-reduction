"""
Channel Flow DRL Training Script using SB3 (SAC)
Adapted from cylinder SB3 training.py for parameter-sharing approach.

Key difference: parameter sharing means each wall cell is a mini-agent.
Each episode produces n_actions * n_actuators = 4 * 1064 = 4256 transitions.
"""


import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from utils.export_actor import export_actor
from utils.update_activation import update_activation_function
from utils.lr_scheduler import linear_schedule
from environment import OFEnvironment
from pathlib import Path
import numpy as np
import yaml
import time, os
# MPI: only import if running under mpirun (nprocs > 1)
try:
    if "PMI_SIZE" in os.environ or "OMPI_COMM_WORLD_SIZE" in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    else:
        raise ImportError("Not under mpirun")
except (ImportError, ModuleNotFoundError):
    import sys
    class MockComm:
        def gather(self, data, root=0): return [data]
        def bcast(self, data, root=0): return data
        def Barrier(self): pass
        def Get_rank(self): return 0
        def Abort(self): sys.exit(1)
    comm = MockComm()
    rank = 0
    nprocs = 1


def update_replay_buffer(replay_buffer, save_path, observations, actions,
                         rewards, terminals, n_actions, n_actuators):
    """
    Add transitions from all processes to the replay buffer.

    With parameter sharing, the data is flattened:
    - observations[p] shape: (n_actions * n_actuators, n_states)
    - actions[p] shape: (n_actions * n_actuators,)
    - rewards[p] shape: (n_actions * n_actuators,)
    - terminals[p] shape: (n_actions * n_actuators,)

    Within the flat array, data is ordered as:
    [t0_act0, t0_act1, ..., t0_act1063, t1_act0, ..., t3_act1063]

    For SAC we need (s, a, r, s', done) tuples. The next state for
    actuator j at timestep i is the state of actuator j at timestep i+1.
    For the last timestep, done=True so s' doesn't affect training.
    """
    for p in range(nprocs):
        if actions[p].size == 0:
            continue

        for i in range(n_actions):
            for j in range(n_actuators):
                idx = i * n_actuators + j
                obs = observations[p][idx]

                # Next observation: same actuator, next timestep
                if i < n_actions - 1:
                    next_idx = (i + 1) * n_actuators + j
                    next_obs = observations[p][next_idx]
                else:
                    # Last timestep: done=True, next_obs is dummy
                    next_obs = observations[p][idx]

                replay_buffer.add(
                    obs,
                    next_obs,
                    np.array([actions[p][idx]]),  # shape (1,) for Box action space
                    rewards[p][idx],
                    terminals[p][idx],
                    [{}],
                )

    torch.save(replay_buffer, save_path)


def main():

    time.sleep(3 * rank)

    current_path = Path(__file__).parent
    model_path = current_path / "model"
    config_path = current_path / "configs"
    env = OFEnvironment(config_path / "channel.yaml", rank)
    total_timesteps = 600_000

    if rank == 0:

        # Load SAC parameters
        with open(config_path / "sac.yaml", "r") as model_file:
            sac_config = yaml.safe_load(model_file)
        sac_params = sac_config["sac_params"]

        policy_kwargs = update_activation_function(sac_config["policy_kwargs"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SAC(
            "MlpPolicy",
            env,
            device=device,
            **sac_params,
            learning_rate=linear_schedule(5e-4, 5e-4),
            policy_kwargs=policy_kwargs,
            tensorboard_log=current_path / "logs" / "tensorboard",
        )
        model._logger = configure(
            folder="logs/", format_strings=["stdout", "csv", "tensorboard"]
        )
        export_actor(model.actor, env)

        replay_buffer = ReplayBuffer(
            buffer_size=sac_params["buffer_size"],
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        model.replay_buffer = replay_buffer

        os.makedirs("episodes", exist_ok=True)
        (model_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        model._total_timesteps = total_timesteps

    total_collected_steps = 0
    env_batch = 0

    while total_collected_steps < total_timesteps:
        if rank == 0:
            print("\nEnvironment batch {}, started".format(env_batch))
        episode = env_batch * nprocs + rank

        tic = time.time()
        env.reset()
        env.run()
        observations, actions, rewards, terminals = env.postprocess()
        env.clean_up()
        toc = time.time()
        sim_time = (toc - tic) / 60

        if actions.size != 0:
            avg_reward = np.sum(rewards) / env.n_actuators
            print(
                ("Episode {} completed, run time={:.1f} min, avg return={:.4f}").format(
                    episode, sim_time, avg_reward
                )
            )
        else:
            print(("Episode {} did not complete successfully").format(episode))

        observations_list = comm.gather(observations, root=0)
        actions_list = comm.gather(actions, root=0)
        rewards_list = comm.gather(rewards, root=0)
        terminals_list = comm.gather(terminals, root=0)

        if rank == 0:
            update_replay_buffer(
                replay_buffer,
                model_path / "replay_buffer.pt",
                observations_list,
                actions_list,
                rewards_list,
                terminals_list,
                env.n_actions,
                env.n_actuators,
            )
            total_collected_steps += sum(len(acts) for acts in actions_list)

            if replay_buffer.size() >= sac_params["batch_size"]:
                model._current_progress_remaining = (
                    total_timesteps - model.num_timesteps
                ) / total_timesteps
                model.train(
                    batch_size=sac_params["batch_size"],
                    gradient_steps=nprocs * sac_params["gradient_steps"],
                )
                model.num_timesteps = total_collected_steps

                export_actor(model.actor, env)
                model.save(current_path / f"model/checkpoints/batch_{env_batch}.zip")

            print(f"  Total Steps Collected: {total_collected_steps}")
            print(f"  Replay Buffer Size: {replay_buffer.size()}")
            print(f"  Total Updates: {model._n_updates}\n")

            env_batch += 1

        env_batch = comm.bcast(env_batch, root=0)
        total_collected_steps = comm.bcast(total_collected_steps, root=0)
        comm.Barrier()

    if rank == 0:
        print("\n\nTraining complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        comm.Abort()
