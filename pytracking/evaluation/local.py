from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = '/media/wcz/datasets/yang/benchmarks/Got10k/'
    settings.lasot_path = '/media/wcz/datasets/yang/benchmarks/Lasot/LaSOTTest/'
    settings.mobiface_path = ''
    settings.network_path = '/home/wcz/Yang/JCAT/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/media/wcz/datasets/yang/benchmarks/NFS/'
    settings.otb_path = '/media/wcz/datasets/yang/benchmarks/OTB100/'
    settings.results_path = '/home/wcz/Yang/JCAT/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = '/media/wcz/datasets/yang/benchmarks/TCL128/'
    settings.trackingnet_path = '/media/wcz/datasets/yang/benchmarks/TrackingNet/'
    settings.uav_path = '/media/wcz/datasets/yang/benchmarks/UAV123/'
    settings.vot16_path = '/media/wcz/datasets/yang/benchmarks/VOT2016/'
    settings.vot18_path = '/media/wcz/datasets/yang/benchmarks/VOT2018/'
    settings.vot19_path = '/media/wcz/datasets/yang/benchmarks/VOT2019/'
    settings.vot_path = ''

    return settings

