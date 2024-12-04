from wis3d import Wis3D
from pathlib import Path
from datetime import datetime

from easyvolcap.engine import *
from easyvolcap.utils.data_utils import to_numpy


def make_wis3d(name="debug", output_dir="data/wis3d", time_postfix=False) -> Wis3D:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if time_postfix:
        time_str = datetime.now().strftime("%m%d-%H%M-%S")
        name = f"{name}_{time_str}"
    log_dir = output_dir / name
    if log_dir.exists():
        log(f'remove contents of directory {log_dir}')
        os.system(f"rm -rf {log_dir}")
    log(f"Creating Wis3D {log_dir}")
    wis3d = Wis3D(output_dir.absolute(), name)
    return wis3d


def wis3d_add_skeleton(vis3d: Wis3D, t: int, joints, parents: list, name: str):
    # joints: (J, 3)
    vis3d.set_scene_id(t)
    joints = to_numpy(joints)
    start_points = joints[1:]
    end_points = [joints[parents[i]] for i in range(1, len(joints))]
    end_points = np.stack(end_points, axis=0)
    vis3d.add_lines(start_points=start_points, end_points=end_points, name=name)


def wis3d_add_coord(wis3d: Wis3D, t: int, axis, origin, name: str):
    '''
        axis: (3, 3)
        origin: (1, 3)
    '''
    wis3d.set_scene_id(t)
    axis, origin = to_numpy(axis), to_numpy(origin)
    
    start_points = np.repeat(origin, 3, axis=0).reshape(3, 3)
    end_points = start_points + axis.T * 0.1

    wis3d.add_lines(start_points=start_points[0:1], end_points=end_points[0:1], name=f'{name}_x')
    wis3d.add_lines(start_points=start_points[1:2], end_points=end_points[1:2], name=f'{name}_y')
    wis3d.add_lines(start_points=start_points[2:3], end_points=end_points[2:3], name=f'{name}_z')