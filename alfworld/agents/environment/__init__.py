def get_environment(env_type):
    match env_type:
        case 'AlfredTWEnv':
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
            return AlfredTWEnv
        case 'AlfredThorEnv':
            from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
            return AlfredThorEnv
        case 'AlfredHybrid':
            from alfworld.agents.environment.alfred_hybrid import AlfredHybrid
            return AlfredHybrid
        case _:
            raise NotImplementedError(f"Environment {env_type} is not implemented.")
