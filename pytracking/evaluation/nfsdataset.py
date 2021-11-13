import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def NFSDataset():
    return NFSDatasetClass().get_sequence_list()


class NFSDatasetClass(BaseDataset):
    """ NFS dataset.

    Publication:
        Need for Speed: A Benchmark for Higher Frame Rate Object Tracking
        H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, and S.Lucey
        ICCV, 2017
        http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Need_for_Speed_ICCV_2017_paper.pdf

    Download the dataset from http://ci2cv.net/nfs/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nfs_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=' ', dtype=np.str)
        ground_truth_rect = ground_truth_rect[:,1:5].astype(np.float64)
        ground_truth_rect[:,2] = ground_truth_rect[:,2]-ground_truth_rect[:,0]
        ground_truth_rect[:,3] = ground_truth_rect[:,3]-ground_truth_rect[:,1]

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,:])

    def __len__(self):
        '''Overload this function in your evaluation. This should return number of sequences in the evaluation '''
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Gymnastics", "path": "Gymnastics/30/Gymnastics", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "Gymnastics/30/Gymnastics.txt"},
            {"name": "MachLoop_jet", "path": "MachLoop_jet/30/MachLoop_jet", "startFrame": 1, "endFrame": 99, "nz": 5, "ext": "jpg", "anno_path": "MachLoop_jet/30/MachLoop_jet.txt"},
            {"name": "Skiing_red", "path": "Skiing_red/30/Skiing_red", "startFrame": 1, "endFrame": 69, "nz": 5, "ext": "jpg", "anno_path": "Skiing_red/30/Skiing_red.txt"},
            {"name": "Skydiving", "path": "Skydiving/30/Skydiving", "startFrame": 1, "endFrame": 196, "nz": 5, "ext": "jpg", "anno_path": "Skydiving/30/Skydiving.txt"},
            {"name": "airboard_1", "path": "airboard_1/30/airboard_1", "startFrame": 1, "endFrame": 425, "nz": 5, "ext": "jpg", "anno_path": "airboard_1/30/airboard_1.txt"},
            {"name": "airplane_landing", "path": "airplane_landing/30/airplane_landing", "startFrame": 1, "endFrame": 81, "nz": 5, "ext": "jpg", "anno_path": "airplane_landing/30/airplane_landing.txt"},
            {"name": "airtable_3", "path": "airtable_3/30/airtable_3", "startFrame": 1, "endFrame": 482, "nz": 5, "ext": "jpg", "anno_path": "airtable_3/30/airtable_3.txt"},
            {"name": "basketball_1", "path": "basketball_1/30/basketball_1", "startFrame": 1, "endFrame": 282, "nz": 5, "ext": "jpg", "anno_path": "basketball_1/30/basketball_1.txt"},
            {"name": "basketball_2", "path": "basketball_2/30/basketball_2", "startFrame": 1, "endFrame": 102, "nz": 5, "ext": "jpg", "anno_path": "basketball_2/30/basketball_2.txt"},
            {"name": "basketball_3", "path": "basketball_3/30/basketball_3", "startFrame": 1, "endFrame": 421, "nz": 5, "ext": "jpg", "anno_path": "basketball_3/30/basketball_3.txt"},
            {"name": "basketball_6", "path": "basketball_6/30/basketball_6", "startFrame": 1, "endFrame": 224, "nz": 5, "ext": "jpg", "anno_path": "basketball_6/30/basketball_6.txt"},
            {"name": "basketball_7", "path": "basketball_7/30/basketball_7", "startFrame": 1, "endFrame": 240, "nz": 5, "ext": "jpg", "anno_path": "basketball_7/30/basketball_7.txt"},
            {"name": "basketball_player", "path": "basketball_player/30/basketball_player", "startFrame": 1, "endFrame": 369, "nz": 5, "ext": "jpg", "anno_path": "basketball_player/30/basketball_player.txt"},
            {"name": "basketball_player_2", "path": "basketball_player_2/30/basketball_player_2", "startFrame": 1, "endFrame": 437, "nz": 5, "ext": "jpg", "anno_path": "basketball_player_2/30/basketball_player_2.txt"},
            {"name": "beach_flipback_person", "path": "beach_flipback_person/30/beach_flipback_person", "startFrame": 1, "endFrame": 61, "nz": 5, "ext": "jpg", "anno_path": "beach_flipback_person/30/beach_flipback_person.txt"},
            {"name": "bee", "path": "bee/30/bee", "startFrame": 1, "endFrame": 45, "nz": 5, "ext": "jpg", "anno_path": "bee/30/bee.txt"},
            {"name": "biker_acrobat", "path": "biker_acrobat/30/biker_acrobat", "startFrame": 1, "endFrame": 128, "nz": 5, "ext": "jpg", "anno_path": "biker_acrobat/30/biker_acrobat.txt"},
            {"name": "biker_all_1", "path": "biker_all_1/30/biker_all_1", "startFrame": 1, "endFrame": 113, "nz": 5, "ext": "jpg", "anno_path": "biker_all_1/30/biker_all_1.txt"},
            {"name": "biker_head_2", "path": "biker_head_2/30/biker_head_2", "startFrame": 1, "endFrame": 132, "nz": 5, "ext": "jpg", "anno_path": "biker_head_2/30/biker_head_2.txt"},
            {"name": "biker_head_3", "path": "biker_head_3/30/biker_head_3", "startFrame": 1, "endFrame": 254, "nz": 5, "ext": "jpg", "anno_path": "biker_head_3/30/biker_head_3.txt"},
            {"name": "biker_upper_body", "path": "biker_upper_body/30/biker_upper_body", "startFrame": 1, "endFrame": 194, "nz": 5, "ext": "jpg", "anno_path": "biker_upper_body/30/biker_upper_body.txt"},
            {"name": "biker_whole_body", "path": "biker_whole_body/biker_whole_body", "startFrame": 1, "endFrame": 572, "nz": 5, "ext": "jpg", "anno_path": "biker_whole_body/30/biker_whole_body.txt"},
            {"name": "billiard_2", "path": "billiard_2/30/billiard_2", "startFrame": 1, "endFrame": 604, "nz": 5, "ext": "jpg", "anno_path": "billiard_2/30/billiard_2.txt"},
            {"name": "billiard_3", "path": "billiard_3/30/billiard_3", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "billiard_3/30/billiard_3.txt"},
            {"name": "billiard_6", "path": "billiard_6/30/billiard_6", "startFrame": 1, "endFrame": 771, "nz": 5, "ext": "jpg", "anno_path": "billiard_6/30/billiard_6.txt"},
            {"name": "billiard_7", "path": "billiard_7/30/billiard_7", "startFrame": 1, "endFrame": 724, "nz": 5, "ext": "jpg", "anno_path": "billiard_7/30/billiard_7.txt"},
            {"name": "billiard_8", "path": "billiard_8/30/billiard_8", "startFrame": 1, "endFrame": 778, "nz": 5, "ext": "jpg", "anno_path": "billiard_8/30/billiard_8.txt"},
            {"name": "bird_2", "path": "bird_2/30/bird_2", "startFrame": 1, "endFrame": 476, "nz": 5, "ext": "jpg", "anno_path": "bird_2/30/bird_2.txt"},
            {"name": "book", "path": "book/30/book", "startFrame": 1, "endFrame": 288, "nz": 5, "ext": "jpg", "anno_path": "book/30/book.txt"},
            {"name": "bottle", "path": "bottle/30/bottle", "startFrame": 1, "endFrame": 2103, "nz": 5, "ext": "jpg", "anno_path": "bottle/30/bottle.txt"},
            {"name": "bowling_1", "path": "bowling_1/30/bowling_1", "startFrame": 1, "endFrame": 303, "nz": 5, "ext": "jpg", "anno_path": "bowling_1/30/bowling_1.txt"},
            {"name": "bowling_2", "path": "bowling_2/30/bowling_2", "startFrame": 1, "endFrame": 710, "nz": 5, "ext": "jpg", "anno_path": "bowling_2/30/bowling_2.txt"},
            {"name": "bowling_3", "path": "bowling_3/30/bowling_3", "startFrame": 1, "endFrame": 271, "nz": 5, "ext": "jpg", "anno_path": "bowling_3/30/bowling_3.txt"},
            {"name": "bowling_6", "path": "bowling_6/30/bowling_6", "startFrame": 1, "endFrame": 260, "nz": 5, "ext": "jpg", "anno_path": "bowling_6/30/bowling_6.txt"},
            {"name": "bowling_ball", "path": "bowling_ball/30/bowling_ball", "startFrame": 1, "endFrame": 275, "nz": 5, "ext": "jpg", "anno_path": "bowling_ball/30/bowling_ball.txt"},
            {"name": "bunny", "path": "bunny/30/bunny", "startFrame": 1, "endFrame": 705, "nz": 5, "ext": "jpg", "anno_path": "bunny/30/bunny.txt"},
            {"name": "car", "path": "car/30/car", "startFrame": 1, "endFrame": 2020, "nz": 5, "ext": "jpg", "anno_path": "car/30/car.txt"},
            {"name": "car_camaro", "path": "car_camaro/30/car_camaro", "startFrame": 1, "endFrame": 36, "nz": 5, "ext": "jpg", "anno_path": "car_camaro/30/car_camaro.txt"},
            {"name": "car_drifting", "path": "car_drifting/30/car_drifting", "startFrame": 1, "endFrame": 173, "nz": 5, "ext": "jpg", "anno_path": "car_drifting/30/car_drifting.txt"},
            {"name": "car_jumping", "path": "car_jumping/30/car_jumping", "startFrame": 1, "endFrame": 22, "nz": 5, "ext": "jpg", "anno_path": "car_jumping/30/car_jumping.txt"},
            {"name": "car_rc_rolling", "path": "car_rc_rolling/30/car_rc_rolling", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "car_rc_rolling/30/car_rc_rolling.txt"},
            {"name": "car_rc_rotating", "path": "car_rc_rotating/30/car_rc_rotating", "startFrame": 1, "endFrame": 80, "nz": 5, "ext": "jpg", "anno_path": "car_rc_rotating/30/car_rc_rotating.txt"},
            {"name": "car_side", "path": "car_side/30/car_side", "startFrame": 1, "endFrame": 108, "nz": 5, "ext": "jpg", "anno_path": "car_side/30/car_side.txt"},
            {"name": "car_white", "path": "car_white/30/car_white", "startFrame": 1, "endFrame": 2063, "nz": 5, "ext": "jpg", "anno_path": "car_white/30/car_white.txt"},
            {"name": "cheetah", "path": "cheetah/30/cheetah", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg", "anno_path": "cheetah/30/cheetah.txt"},
            {"name": "cup", "path": "cup/30/cup", "startFrame": 1, "endFrame": 1281, "nz": 5, "ext": "jpg", "anno_path": "cup/30/cup.txt"},
            {"name": "cup_2", "path": "cup_2/30/cup_2", "startFrame": 1, "endFrame": 182, "nz": 5, "ext": "jpg", "anno_path": "cup_2/30/cup_2.txt"},
            {"name": "dog", "path": "dog/30/dog", "startFrame": 1, "endFrame": 1030, "nz": 5, "ext": "jpg", "anno_path": "dog/30/dog.txt"},
            {"name": "dog_1", "path": "dog_1/30/dog_1", "startFrame": 1, "endFrame": 168, "nz": 5, "ext": "jpg", "anno_path": "dog_1/30/dog_1.txt"},
            {"name": "dog_2", "path": "dog_2/30/dog_2", "startFrame": 1, "endFrame": 594, "nz": 5, "ext": "jpg", "anno_path": "dog_2/30/dog_2.txt"},
            {"name": "dog_3", "path": "dog_3/30/dog_3", "startFrame": 1, "endFrame": 200, "nz": 5, "ext": "jpg", "anno_path": "dog_3/30/dog_3.txt"},
            {"name": "dogs", "path": "dogs/30/dogs", "startFrame": 1, "endFrame": 198, "nz": 5, "ext": "jpg", "anno_path": "dogs/30/dogs.txt"},
            {"name": "dollar", "path": "dollar/30/dollar", "startFrame": 1, "endFrame": 1426, "nz": 5, "ext": "jpg", "anno_path": "dollar/30/dollar.txt"},
            {"name": "drone", "path": "drone/30/drone", "startFrame": 1, "endFrame": 70, "nz": 5, "ext": "jpg", "anno_path": "drone/30/drone.txt"},
            {"name": "ducks_lake", "path": "ducks_lake/30/ducks_lake", "startFrame": 1, "endFrame": 107, "nz": 5, "ext": "jpg", "anno_path": "ducks_lake/30/ducks_lake.txt"},
            {"name": "exit", "path": "exit/30/exit", "startFrame": 1, "endFrame": 359, "nz": 5, "ext": "jpg", "anno_path": "exit/30/exit.txt"},
            {"name": "first", "path": "first/30/first", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg", "anno_path": "first/30/first.txt"},
            {"name": "flower", "path": "flower/30/flower", "startFrame": 1, "endFrame": 448, "nz": 5, "ext": "jpg", "anno_path": "flower/30/flower.txt"},
            {"name": "footbal_skill", "path": "footbal_skill/30/footbal_skill", "startFrame": 1, "endFrame": 131, "nz": 5, "ext": "jpg", "anno_path": "footbal_skill/30/footbal_skill.txt"},
            {"name": "helicopter", "path": "helicopter/30/helicopter", "startFrame": 1, "endFrame": 310, "nz": 5, "ext": "jpg", "anno_path": "helicopter/30/helicopter.txt"},
            {"name": "horse_jumping", "path": "horse_jumping/30/horse_jumping", "startFrame": 1, "endFrame": 117, "nz": 5, "ext": "jpg", "anno_path": "horse_jumping/30/horse_jumping.txt"},
            {"name": "horse_running", "path": "horse_running/30/horse_running", "startFrame": 1, "endFrame": 139, "nz": 5, "ext": "jpg", "anno_path": "horse_running/30/horse_running.txt"},
            {"name": "iceskating_6", "path": "iceskating_6/30/iceskating_6", "startFrame": 1, "endFrame": 603, "nz": 5, "ext": "jpg", "anno_path": "iceskating_6/30/iceskating_6.txt"},
            {"name": "jellyfish_5", "path": "jellyfish_5/30/jellyfish_5", "startFrame": 1, "endFrame": 746, "nz": 5, "ext": "jpg", "anno_path": "jellyfish_5/30/jellyfish_5.txt"},
            {"name": "kid_swing", "path": "kid_swing/30/kid_swing", "startFrame": 1, "endFrame": 169, "nz": 5, "ext": "jpg", "anno_path": "kid_swing/30/kid_swing.txt"},
            {"name": "motorcross", "path": "motorcross/30/motorcross", "startFrame": 1, "endFrame": 39, "nz": 5, "ext": "jpg", "anno_path": "motorcross/30/motorcross.txt"},
            {"name": "motorcross_kawasaki", "path": "motorcross_kawasaki/30/motorcross_kawasaki", "startFrame": 1, "endFrame": 65, "nz": 5, "ext": "jpg", "anno_path": "motorcross_kawasaki/30/motorcross_kawasaki.txt"},
            {"name": "parkour", "path": "parkour/30/parkour", "startFrame": 1, "endFrame": 58, "nz": 5, "ext": "jpg", "anno_path": "parkour/30/parkour.txt"},
            {"name": "person_scooter", "path": "person_scooter/30/person_scooter", "startFrame": 1, "endFrame": 413, "nz": 5, "ext": "jpg", "anno_path": "person_scooter/30/person_scooter.txt"},
            {"name": "pingpong_2", "path": "pingpong_2/30/pingpong_2", "startFrame": 1, "endFrame": 1277, "nz": 5, "ext": "jpg", "anno_path": "pingpong_2/30/pingpong_2.txt"},
            {"name": "pingpong_7", "path": "pingpong_7/30/pingpong_7", "startFrame": 1, "endFrame": 1290, "nz": 5, "ext": "jpg", "anno_path": "pingpong_7/30/pingpong_7.txt"},
            {"name": "pingpong_8", "path": "pingpong_8/30/pingpong_8", "startFrame": 1, "endFrame": 296, "nz": 5, "ext": "jpg", "anno_path": "pingpong_8/30/pingpong_8.txt"},
            {"name": "purse", "path": "purse/30/purse", "startFrame": 1, "endFrame": 968, "nz": 5, "ext": "jpg", "anno_path": "purse/30/purse.txt"},
            {"name": "rubber", "path": "rubber/30/rubber", "startFrame": 1, "endFrame": 1328, "nz": 5, "ext": "jpg", "anno_path": "rubber/30/rubber.txt"},
            {"name": "running", "path": "running/30/running", "startFrame": 1, "endFrame": 677, "nz": 5, "ext": "jpg", "anno_path": "running/30/running.txt"},
            {"name": "running_100_m", "path": "running_100_m/30/running_100_m", "startFrame": 1, "endFrame": 313, "nz": 5, "ext": "jpg", "anno_path": "running_100_m/30/running_100_m.txt"},
            {"name": "running_100_m_2", "path": "running_100_m_2/30/running_100_m_2", "startFrame": 1, "endFrame": 337, "nz": 5, "ext": "jpg", "anno_path": "running_100_m_2/30/running_100_m_2.txt"},
            {"name": "running_2", "path": "running_2/30/running_2", "startFrame": 1, "endFrame": 363, "nz": 5, "ext": "jpg", "anno_path": "running_2/30/running_2.txt"},
            {"name": "shuffleboard_1", "path": "shuffleboard_1/30/shuffleboard_1", "startFrame": 1, "endFrame": 42, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_1/30/shuffleboard_1.txt"},
            {"name": "shuffleboard_2", "path": "shuffleboard_2/30/shuffleboard_2", "startFrame": 1, "endFrame": 41, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_2/30/shuffleboard_2.txt"},
            {"name": "shuffleboard_4", "path": "shuffleboard_4/30/shuffleboard_4", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_4/30/shuffleboard_4.txt"},
            {"name": "shuffleboard_5", "path": "shuffleboard_5/30/shuffleboard_5", "startFrame": 1, "endFrame": 32, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_5/30/shuffleboard_5.txt"},
            {"name": "shuffleboard_6", "path": "shuffleboard_6/30/shuffleboard_6", "startFrame": 1, "endFrame": 52, "nz": 5, "ext": "jpg", "anno_path": "shuffleboard_6/30/shuffleboard_6.txt"},
            {"name": "shuffletable_2", "path": "shuffletable_2/30/shuffletable_2", "startFrame": 1, "endFrame": 372, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_2/30/shuffletable_2.txt"},
            {"name": "shuffletable_3", "path": "shuffletable_3/30/shuffletable_3", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_3/30/shuffletable_3.txt"},
            {"name": "shuffletable_4", "path": "shuffletable_4/30/shuffletable_4", "startFrame": 1, "endFrame": 101, "nz": 5, "ext": "jpg", "anno_path": "shuffletable_4/30/shuffletable_4.txt"},
            {"name": "ski_long", "path": "ski_long/30/ski_long", "startFrame": 1, "endFrame": 274, "nz": 5, "ext": "jpg", "anno_path": "ski_long/30/ski_long.txt"},
            {"name": "soccer_ball", "path": "soccer_ball/30/soccer_ball", "startFrame": 1, "endFrame": 163, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball/30/soccer_ball.txt"},
            {"name": "soccer_ball_2", "path": "soccer_ball_2/30/soccer_ball_2", "startFrame": 1, "endFrame": 1934, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball_2/30/soccer_ball_2.txt"},
            {"name": "soccer_ball_3", "path": "soccer_ball_3/30/soccer_ball_3", "startFrame": 1, "endFrame": 1381, "nz": 5, "ext": "jpg", "anno_path": "soccer_ball_3/30/soccer_ball_3.txt"},
            {"name": "soccer_player_2", "path": "soccer_player_2/30/soccer_player_2", "startFrame": 1, "endFrame": 475, "nz": 5, "ext": "jpg", "anno_path": "soccer_player_2/30/soccer_player_2.txt"},
            {"name": "soccer_player_3", "path": "soccer_player_3/30/soccer_player_3", "startFrame": 1, "endFrame": 319, "nz": 5, "ext": "jpg", "anno_path": "soccer_player_3/30/soccer_player_3.txt"},
            {"name": "stop_sign", "path": "stop_sign/30/stop_sign", "startFrame": 1, "endFrame": 302, "nz": 5, "ext": "jpg", "anno_path": "stop_sign/30/stop_sign.txt"},
            {"name": "suv", "path": "suv/30/suv", "startFrame": 1, "endFrame": 2584, "nz": 5, "ext": "jpg", "anno_path": "suv/30/suv.txt"},
            {"name": "tiger", "path": "tiger/30/tiger", "startFrame": 1, "endFrame": 1556, "nz": 5, "ext": "jpg", "anno_path": "tiger/30/tiger.txt"},
            {"name": "walking", "path": "walking/30/walking", "startFrame": 1, "endFrame": 555, "nz": 5, "ext": "jpg", "anno_path": "walking/30/walking.txt"},
            {"name": "walking_3", "path": "walking_3/30/walking_3", "startFrame": 1, "endFrame": 1427, "nz": 5, "ext": "jpg", "anno_path": "walking_3/30/walking_3.txt"},
            {"name": "water_ski_2", "path": "water_ski_2/30/water_ski_2", "startFrame": 1, "endFrame": 47, "nz": 5, "ext": "jpg", "anno_path": "water_ski_2/30/water_ski_2.txt"},
            {"name": "yoyo", "path": "yoyo/30/yoyo", "startFrame": 1, "endFrame": 67, "nz": 5, "ext": "jpg", "anno_path": "yoyo/30/yoyo.txt"},
            {"name": "zebra_fish", "path": "zebra_fish/30/zebra_fish", "startFrame": 1, "endFrame": 671, "nz": 5, "ext": "jpg", "anno_path": "zebra_fish/30/zebra_fish.txt"},
        ]

        return sequence_info_list
