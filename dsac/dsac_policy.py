"""Construction of a discrete SAC policy using Tianshou's standard networks."""

import torch
from tianshou.policy import DiscreteSACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic


# =============================================================================
# Standard Tianshou discrete SAC networks
# =============================================================================

def build_discrete_sac_policy(
    state_shape,
    action_shape,
    device,
    learning_rate,
    hidden_sizes,
    gamma,
    alpha,
    tau,
):
    """Build DSAC with Tianshou's default MLP, actor and critic classes."""
    # The actor backbone extracts state features. Tianshou's Actor adds the
    # action head and normalizes its output into a categorical distribution.
    actor_backbone = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    )
    actor = Actor(
        preprocess_net=actor_backbone,
        action_shape=action_shape,
        device=device,
        # DiscreteSACPolicy passes this output to Categorical(logits=...).
        softmax_output=False,
    ).to(device)

    # Discrete SAC uses two independent state-action value estimators to reduce
    # positive bias in the bootstrapped target.
    critic1_backbone = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    )
    critic1 = Critic(
        preprocess_net=critic1_backbone,
        last_size=action_shape,
        device=device,
    ).to(device)

    critic2_backbone = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    )
    critic2 = Critic(
        preprocess_net=critic2_backbone,
        last_size=action_shape,
        device=device,
    ).to(device)

    # Each network owns a separate optimizer, as required by DiscreteSACPolicy.
    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=torch.optim.Adam(actor.parameters(), lr=learning_rate),
        critic1=critic1,
        critic1_optim=torch.optim.Adam(critic1.parameters(), lr=learning_rate),
        critic2=critic2,
        critic2_optim=torch.optim.Adam(critic2.parameters(), lr=learning_rate),
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=1,
    )
    return policy.to(device)
