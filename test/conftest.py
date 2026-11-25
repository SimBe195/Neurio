from jaxtyping import install_import_hook

# Runtime-check the jaxtyping annotations of these modules during tests. Installed before any test imports them.
# agents.ppo.advantage is excluded on purpose since its tests expect its own ValueErrors on malformed inputs.
install_import_hook(
    [
        "agents.ppo.agent",
        "agents.ppo.experience_buffer",
        "agents.ppo.self_imitation",
        "agents.ppo.actor_critic",
        "agents.ppo.impala_encoder",
    ],
    "beartype.beartype",
)
