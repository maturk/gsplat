import json
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import open3d
import torch


def load_from_json(filename: Path):
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

class BlenderParser:
    """Blender dataparser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        split: Literal["train","val","test"] = "train",
        num_points: int = 100_000,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        c2w_mats = []
        meta = load_from_json(self.data_dir + f"/transforms_{split}.json")
        image_names = []
        image_paths = []
        for camera_idx, frame in enumerate(meta["frames"]):
            image_name = frame["file_path"]
            image_names.append(image_name)
            fname = self.data_dir / Path(frame["file_path"].replace("./", "") + ".png")
            image_paths.append(fname)

            c2w_mat = np.array(frame["transform_matrix"])
            c2w_mat[:3, 1:3] *= -1
            c2w_mats.append(c2w_mat)

        c2w_mats = np.array(c2w_mats).astype(np.float32)
        img_0 = imageio.v2.imread(image_paths[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0

        Ks_dict = {}
        imsize_dict = {}
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        K[:2, :] /= factor
        camera_ids = [id for id in range(len(image_names))]
        params_dict = {}
        for camera_id in camera_ids:
            Ks_dict[camera_id] = K
            imsize_dict[camera_id] = (image_width // factor, image_height // factor)
            params_dict[camera_id] = np.empty(0, dtype=np.float32)
        
        # size of the scene measured by cameras
        camera_locations = c2w_mats[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


        print(f"Generating random point cloud ({num_points})...")
        num_points = 100_000_0
        # We create random points inside the bounds of the synthetic Blender scenes
        # points = np.random.random((num_points, 3)) * 2.6 - 1.3
        points = np.random.random((num_points, 3)) * self.scene_scale - scene_center
        points_rgb = np.random.random((num_points, 3)) * 255
        points_rgb = points_rgb.astype(np.uint8)
        print(points_rgb)
        points_err = np.zeros_like(points)[:,0]

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = c2w_mats  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = None  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform =  np.eye(4)  # np.ndarray, (4, 4)


    def _load_3D_points(self, ply_file_path: Path):
        pcd = open3d.io.read_point_cloud(str(ply_file_path))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.config.scale_factor)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out
    
if __name__ == "__main__":
    parser = BlenderParser(data_dir="/home/turkulm1/data/nerf_synthetic/lego")