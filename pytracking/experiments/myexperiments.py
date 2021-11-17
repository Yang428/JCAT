from pytracking.evaluation import Tracker, OTBDataset, VOTDataset, TrackingNetDataset, GOT10KDatasetTest, VOT16Dataset, VOT18Dataset, VOT19Dataset, NFSDataset, UAVDataset, TPLDataset, LaSOTDataset

def got10k():
    # Run experiment on the Got10k dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = GOT10KDatasetTest()
    return trackers, dataset      

def trackingnet():
    # Run experiment on the TrackingNet dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = TrackingNetDataset()
    return trackers, dataset    

def otb():
    # Run experiment on the OTB100 dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = OTBDataset()
    return trackers, dataset

def nfs():
    # Run experiment on the NFS dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = NFSDataset()
    return trackers, dataset    

def tcl128():
    # Run experiment on the TCL128 dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = TPLDataset()
    return trackers, dataset   

def lasot():
    # Run experiment on the LaSOT dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = LaSOTDataset()
    return trackers, dataset 

def uav123():
    # Run experiment on the UAV123 dataset
    trackers = [Tracker('Jcat', 'Jcat')]

    dataset = UAVDataset()
    return trackers, dataset
