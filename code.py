import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================
# Configuration
# ============================================================

ENV_NAME = "Pendulum-v1"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 3
ACTION_DIM = 1

ACTION_LOW = -2.0
ACTION_HIGH = 2.0

EPISODE_LENGTH = 200

# Model-based agent
ENSEMBLE_SIZE = 5
MODEL_HIDDEN = 200
MODEL_EPOCHS = 20
MODEL_BATCH_SIZE = 256

CEM_HORIZON = 20
CEM_POPULATION = 500
CEM_ELITE = 50
CEM_ITERATIONS = 5

# Replay buffer
BUFFER_SIZE = 100_000

# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                float(reward),
                float(done)
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, next_states, rewards, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Probabilistic Dynamics Model
#
# Predicts:
#       p(next_state | state, action)
#
# Output:
#       mean and log variance
# ============================================================

class ProbabilisticDynamicsModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, MODEL_HIDDEN),
            nn.ReLU(),

            nn.Linear(MODEL_HIDDEN, MODEL_HIDDEN),
            nn.ReLU(),

            nn.Linear(MODEL_HIDDEN, MODEL_HIDDEN),
            nn.ReLU(),

            nn.Linear(MODEL_HIDDEN, STATE_DIM * 2)
        )

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)

        output = self.net(x)

        mean = output[:, :STATE_DIM]

        log_var = output[:, STATE_DIM:]

        # Prevent numerical instability
        log_var = torch.clamp(log_var, -10, 2)

        return mean, log_var


# ============================================================
# Probabilistic loss
# Negative log likelihood
# ============================================================

def probabilistic_loss(mean, log_var, target):

    variance = torch.exp(log_var)

    loss = 0.5 * (
        log_var +
        (target - mean) ** 2 / variance
    )

    return loss.mean()


# ============================================================
# Train one dynamics model
# ============================================================

def train_dynamics_model(model, dataset):

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    states = torch.tensor(
        np.array(dataset["states"]),
        dtype=torch.float32
    ).to(DEVICE)

    actions = torch.tensor(
        np.array(dataset["actions"]),
        dtype=torch.float32
    ).to(DEVICE)

    targets = torch.tensor(
        np.array(dataset["next_states"]),
        dtype=torch.float32
    ).to(DEVICE)

    n = len(states)

    for epoch in range(MODEL_EPOCHS):

        indices = torch.randperm(n)

        total_loss = 0

        for start in range(0, n, MODEL_BATCH_SIZE):

            idx = indices[
                start:start + MODEL_BATCH_SIZE
            ]

            s = states[idx]
            a = actions[idx]
            y = targets[idx]

            mean, log_var = model(s, a)

            loss = probabilistic_loss(
                mean,
                log_var,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if epoch == MODEL_EPOCHS - 1:
            print(
                f"    Model training loss: "
                f"{total_loss:.4f}"
            )


# ============================================================
# CEM Planner
# ============================================================

class CEMPlanner:

    def __init__(self, models):

        self.models = models

    @torch.no_grad()
    def predict_next_state(
        self,
        state,
        action
    ):

        predictions = []

        for model in self.models:

            mean, log_var = model(
                state,
                action
            )

            std = torch.exp(
                0.5 * log_var
            )

            # Sample probabilistic transition
            noise = torch.randn_like(std)

            next_state = (
                mean +
                std * noise
            )

            predictions.append(next_state)

        # Ensemble average
        return torch.stack(
            predictions
        ).mean(dim=0)

    @torch.no_grad()
    def evaluate_sequences(
        self,
        initial_state,
        action_sequences
    ):

        population = action_sequences.shape[0]

        states = initial_state.repeat(
            population,
            1
        )

        total_rewards = torch.zeros(
            population,
            device=DEVICE
        )

        for t in range(CEM_HORIZON):

            actions = action_sequences[:, t, :]

            next_states = []

            # Randomly choose models from ensemble
            model_indices = np.random.randint(
                0,
                len(self.models),
                population
            )

            for i in range(population):

                model = self.models[
                    model_indices[i]
                ]

                mean, log_var = model(
                    states[i:i+1],
                    actions[i:i+1]
                )

                std = torch.exp(
                    0.5 * log_var
                )

                next_state = (
                    mean +
                    std *
                    torch.randn_like(std)
                )

                next_states.append(
                    next_state
                )

            next_states = torch.cat(
                next_states,
                dim=0
            )

            # Pendulum reward approximation
            theta = torch.atan2(
                next_states[:, 1],
                next_states[:, 0]
            )

            theta_dot = next_states[:, 2]

            reward = -(
                theta ** 2
                + 0.1 * theta_dot ** 2
                + 0.001 *
                actions.squeeze(-1) ** 2
            )

            total_rewards += reward

            states = next_states

        return total_rewards

    def plan(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        mean = torch.zeros(
            CEM_HORIZON,
            ACTION_DIM,
            device=DEVICE
        )

        std = torch.ones(
            CEM_HORIZON,
            ACTION_DIM,
            device=DEVICE
        ) * 1.0

        for _ in range(CEM_ITERATIONS):

            # Sample action sequences
            noise = torch.randn(
                CEM_POPULATION,
                CEM_HORIZON,
                ACTION_DIM,
                device=DEVICE
            )

            actions = (
                mean.unsqueeze(0)
                +
                std.unsqueeze(0)
                * noise
            )

            actions = torch.clamp(
                actions,
                ACTION_LOW,
                ACTION_HIGH
            )

            scores = self.evaluate_sequences(
                state,
                actions
            )

            elite_indices = torch.topk(
                scores,
                CEM_ELITE
            ).indices

            elite_actions = actions[
                elite_indices
            ]

            mean = elite_actions.mean(
                dim=0
            )

            std = elite_actions.std(
                dim=0
            )

            std = torch.clamp(
                std,
                min=0.05
            )

        # Receding horizon control:
        # execute only the first action
        return mean[0].cpu().numpy()


# ============================================================
# Collect initial random data
# ============================================================

def collect_random_data(
    env,
    buffer,
    steps=5000
):

    state, _ = env.reset(
        seed=SEED
    )

    for _ in range(steps):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, _ = env.step(
            action
        )

        buffer.add(
            state,
            action,
            next_state,
            reward,
            terminated or truncated
        )

        state = next_state

        if terminated or truncated:
            state, _ = env.reset()


# ============================================================
# Train ensemble
# ============================================================

def train_ensemble(buffer):

    print(
        "\nTraining probabilistic "
        "dynamics ensemble..."
    )

    models = []

    data = list(buffer.buffer)

    for model_number in range(
        ENSEMBLE_SIZE
    ):

        print(
            f"\nTraining model "
            f"{model_number + 1}/"
            f"{ENSEMBLE_SIZE}"
        )

        # Bootstrap sample
        indices = np.random.randint(
            0,
            len(data),
            len(data)
        )

        sample = [
            data[i]
            for i in indices
        ]

        dataset = {
            "states": [
                x[0] for x in sample
            ],

            "actions": [
                x[1] for x in sample
            ],

            "next_states": [
                x[2] for x in sample
            ]
        }

        model = ProbabilisticDynamicsModel().to(
            DEVICE
        )

        train_dynamics_model(
            model,
            dataset
        )

        models.append(model)

    return models


# ============================================================
# Evaluate Model-Based Agent
# ============================================================

def evaluate_model_based_agent(
    env,
    total_steps=10000
):

    buffer = ReplayBuffer(
        BUFFER_SIZE
    )

    print(
        "Collecting initial "
        "random dataset..."
    )

    collect_random_data(
        env,
        buffer,
        steps=1000
    )

    models = train_ensemble(
        buffer
    )

    planner = CEMPlanner(
        models
    )

    returns = []
    steps = []

    total_environment_steps = 1000

    while total_environment_steps < total_steps:

        state, _ = env.reset()

        episode_return = 0

        for _ in range(
            EPISODE_LENGTH
        ):

            action = planner.plan(
                state
            )

            next_state, reward, terminated, truncated, _ = env.step(
                action
            )

            buffer.add(
                state,
                action,
                next_state,
                reward,
                terminated or truncated
            )

            state = next_state

            episode_return += reward

            total_environment_steps += 1

            if (
                total_environment_steps
                % 1000 == 0
            ):

                print(
                    f"MB steps: "
                    f"{total_environment_steps}, "
                    f"return: "
                    f"{episode_return:.2f}"
                )

            if terminated or truncated:
                break

        returns.append(
            episode_return
        )

        steps.append(
            total_environment_steps
        )

        # Periodically retrain models
        if (
            total_environment_steps
            % 2000 == 0
        ):

            models = train_ensemble(
                buffer
            )

            planner = CEMPlanner(
                models
            )

    return steps, returns


# ============================================================
# SAC NETWORKS
# ============================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(
            256,
            ACTION_DIM
        )

        self.log_std = nn.Linear(
            256,
            ACTION_DIM
        )

    def forward(self, state):

        x = self.net(state)

        mean = self.mean(x)

        log_std = torch.clamp(
            self.log_std(x),
            -20,
            2
        )

        std = torch.exp(
            log_std
        )

        normal = torch.distributions.Normal(
            mean,
            std
        )

        z = normal.rsample()

        action = torch.tanh(z)

        log_prob = (
            normal.log_prob(z)
            -
            torch.log(
                1 - action.pow(2)
                + 1e-6
            )
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True
        )

        action = action * ACTION_HIGH

        return action, log_prob


class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(
                STATE_DIM + ACTION_DIM,
                256
            ),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(
        self,
        state,
        action
    ):

        return self.q(
            torch.cat(
                [state, action],
                dim=-1
            )
        )


# ============================================================
# SAC Agent
# ============================================================

class SACAgent:

    def __init__(self):

        self.actor = Actor().to(
            DEVICE
        )

        self.critic1 = Critic().to(
            DEVICE
        )

        self.critic2 = Critic().to(
            DEVICE
        )

        self.target1 = Critic().to(
            DEVICE
        )

        self.target2 = Critic().to(
            DEVICE
        )

        self.target1.load_state_dict(
            self.critic1.state_dict()
        )

        self.target2.load_state_dict(
            self.critic2.state_dict()
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4
        )

        self.critic_optimizer = optim.Adam(
            list(
                self.critic1.parameters()
            )
            +
            list(
                self.critic2.parameters()
            ),
            lr=3e-4
        )

        self.gamma = 0.99

        self.tau = 0.005

        self.alpha = 0.2

    def update(
        self,
        replay,
        batch_size=256
    ):

        if len(replay) < batch_size:
            return

        (
            states,
            actions,
            next_states,
            rewards,
            dones
        ) = replay.sample(
            batch_size
        )

        with torch.no_grad():

            next_actions, next_logp = (
                self.actor(next_states)
            )

            target_q1 = self.target1(
                next_states,
                next_actions
            )

            target_q2 = self.target2(
                next_states,
                next_actions
            )

            target_q = torch.min(
                target_q1,
                target_q2
            )

            target = (
                rewards
                +
                self.gamma
                * (1 - dones)
                * (
                    target_q
                    -
                    self.alpha
                    * next_logp
                )
            )

        current_q1 = self.critic1(
            states,
            actions
        )

        current_q2 = self.critic2(
            states,
            actions
        )

        critic_loss = (
            (current_q1 - target).pow(2).mean()
            +
            (current_q2 - target).pow(2).mean()
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # Actor update
        new_actions, logp = self.actor(
            states
        )

        q1 = self.critic1(
            states,
            new_actions
        )

        q2 = self.critic2(
            states,
            new_actions
        )

        q = torch.min(
            q1,
            q2
        )

        actor_loss = (
            self.alpha * logp - q
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # Soft target update
        for target, source in [
            (self.target1, self.critic1),
            (self.target2, self.critic2)
        ]:

            for tp, sp in zip(
                target.parameters(),
                source.parameters()
            ):

                tp.data.copy_(
                    self.tau * sp.data
                    +
                    (1 - self.tau)
                    * tp.data
                )


# ============================================================
# Train SAC
# ============================================================

def train_sac(
    env,
    total_steps=50000
):

    agent = SACAgent()

    replay = ReplayBuffer(
        BUFFER_SIZE
    )

    state, _ = env.reset(
        seed=SEED + 1
    )

    episode_return = 0

    returns = []

    steps = []

    for step in range(
        1,
        total_steps + 1
    ):

        # Random exploration during warmup
        if step < 1000:

            action = env.action_space.sample()

        else:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                action, _ = agent.actor(
                    state_tensor
                )

            action = (
                action
                .cpu()
                .numpy()
                .flatten()
            )

        next_state, reward, terminated, truncated, _ = env.step(
            action
        )

        replay.add(
            state,
            action,
            next_state,
            reward,
            terminated or truncated
        )

        state = next_state

        episode_return += reward

        if step >= 1000:

            agent.update(
                replay
            )

        if terminated or truncated:

            returns.append(
                episode_return
            )

            steps.append(
                step
            )

            print(
                f"SAC steps: {step}, "
                f"return: "
                f"{episode_return:.2f}"
            )

            state, _ = env.reset()

            episode_return = 0

    return steps, returns


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "Device:",
        DEVICE
    )

    env_mb = gym.make(
        ENV_NAME
    )

    env_sac = gym.make(
        ENV_NAME
    )

    # --------------------------------------------------------
    # Model-Based + CEM
    # --------------------------------------------------------

    mb_steps, mb_returns = (
        evaluate_model_based_agent(
            env_mb,
            total_steps=10000
        )
    )

    # --------------------------------------------------------
    # SAC
    # --------------------------------------------------------

    sac_steps, sac_returns = train_sac(
        env_sac,
        total_steps=50000
    )

    # --------------------------------------------------------
    # Plot comparison
    # --------------------------------------------------------

    # Smooth returns
    def moving_average(
        values,
        window=5
    ):

        if len(values) < window:
            return values

        return np.convolve(
            values,
            np.ones(window) / window,
            mode="valid"
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        mb_steps[
            len(mb_steps)
            - len(moving_average(mb_returns))
        :],
        moving_average(mb_returns),
        label="Model-Based + CEM"
    )

    plt.plot(
        sac_steps[
            len(sac_steps)
            - len(moving_average(sac_returns))
        :],
        moving_average(sac_returns),
        label="SAC"
    )

    plt.xlabel(
        "Environment Steps"
    )

    plt.ylabel(
        "Episode Return"
    )

    plt.title(
        "Sample Efficiency: Model-Based RL vs SAC"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    env_mb.close()
    env_sac.close()