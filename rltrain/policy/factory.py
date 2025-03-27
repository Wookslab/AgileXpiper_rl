# policy/factory.py

def create_policy(policy_name, obs_dim, act_dim):
    if policy_name == "mlp":
        from .mlp_policy import MLPPolicy
        return MLPPolicy(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
