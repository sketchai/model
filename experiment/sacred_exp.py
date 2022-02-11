from sacred import Experiment
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# BEGIN Sacred setup
ex = Experiment()
ex.add_config('default.yaml')


@ex.config
def update_paths(root_dir, name):
    utils.check_dir(root_dir)
    log_dir = os.path.join(root_dir, name)


@ex.config_hook
def init_observers(config, command_name, logger):
    if command_name == 'training':
        neptune = config['observers']['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers.append(NeptuneObserver(project_name=neptune['project']))
    return config


@ex.post_run_hook
def clean_observer(observers):
    """ Observers that are added in a config_hook need to be cleaned """
    try:
        neptune = observers['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers = [obs for obs in ex.observers
                            if not isinstance(obs, NeptuneObserver)]
    except KeyError:
        pass


# END Sacred setup
