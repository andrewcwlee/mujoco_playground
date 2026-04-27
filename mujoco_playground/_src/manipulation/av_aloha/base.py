# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for AV-ALOHA."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.av_aloha import (
    av_aloha_constants as consts,
)


def get_assets() -> Dict[str, bytes]:
  """Returns a dictionary of all assets used by the environment.

  Pulls VX300S meshes and the table/scene assets from the menagerie ALOHA
  directory, then layers AV-ALOHA's own XMLs and the third-arm meshes
  (vx300s_7dof_wrist_*.stl, zedm.stl) from the local av_aloha/xmls/ tree.
  """
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "aloha"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  path = mjx_env.ROOT_PATH / "manipulation" / "av_aloha" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class AvAlohaEnv(mjx_env.MjxEnv):
  """Base class for AV-ALOHA environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path
    self._renderer = None  # Lazily created for active-vision RGB rendering.
    self._depth_renderer = None  # Lazily created for active-vision depth.

  def _post_init_av_aloha(self, keyframe: str = "home"):
    """Initializes helpful robot properties (mirrors AlohaEnv post-init)."""
    self._left_gripper_site = self._mj_model.site("left/gripper").id
    self._right_gripper_site = self._mj_model.site("right/gripper").id
    self._middle_camera_site = self._mj_model.site(
        consts.MIDDLE_CAMERA_SITE
    ).id
    self._table_geom = self._mj_model.geom("table").id
    self._finger_geoms = [
        self._mj_model.geom(geom_id).id for geom_id in consts.FINGER_GEOMS
    ]
    self._init_q = jp.array(self._mj_model.keyframe(keyframe).qpos)
    self._init_ctrl = jp.array(self._mj_model.keyframe(keyframe).ctrl)
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

    arm_joint_ids = [self._mj_model.joint(j).id for j in consts.ARM_JOINTS]
    self._arm_qadr = jp.array(
        [self._mj_model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids]
    )
    middle_joint_ids = [
        self._mj_model.joint(j).id for j in consts.MIDDLE_ARM_JOINTS
    ]
    self._middle_arm_qadr = jp.array(
        [self._mj_model.jnt_qposadr[joint_id] for joint_id in middle_joint_ids]
    )
    self._finger_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in consts.FINGER_JOINTS
    ])

    self._table_finger_found_sensor = [
        self._mj_model.sensor("table_" + geom + "_found").id
        for geom in consts.FINGER_GEOMS
    ]

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def hand_table_collision(self, data) -> jp.ndarray:
    hand_table_collisions = [
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._table_finger_found_sensor
    ]
    return (sum(hand_table_collisions) > 0).astype(float)

  def _sync_mj_data(self, data) -> mujoco.MjData:
    mj_data = mujoco.MjData(self._mj_model)
    mj_data.qpos[:] = np.asarray(data.qpos)
    mj_data.qvel[:] = np.asarray(data.qvel)
    mujoco.mj_forward(self._mj_model, mj_data)
    return mj_data

  def render_active_vision(
      self, data, width: int = 640, height: int = 480
  ) -> np.ndarray:
    """Render an RGB image from the middle-arm ZED left camera.

    Eval-only: not JIT-safe. Copies mjx data back to the CPU MjData and
    runs MuJoCo's offscreen renderer. Returns uint8 (H, W, 3).
    """
    if self._renderer is None or self._renderer.height != height or (
        self._renderer.width != width
    ):
      self._renderer = mujoco.Renderer(
          self._mj_model, height=height, width=width
      )
    mj_data = self._sync_mj_data(data)
    self._renderer.update_scene(mj_data, camera=consts.MIDDLE_CAMERA_LEFT)
    return self._renderer.render()

  def render_active_vision_depth(
      self, data, width: int = 640, height: int = 480
  ) -> np.ndarray:
    """Render a metric-depth image from the middle-arm ZED left camera.

    Eval-only. Returns float32 (H, W) where each pixel is depth in meters
    along the camera's view direction. Background pixels (no hit) report
    the model's far clip distance.
    """
    if (
        self._depth_renderer is None
        or self._depth_renderer.height != height
        or self._depth_renderer.width != width
    ):
      self._depth_renderer = mujoco.Renderer(
          self._mj_model, height=height, width=width
      )
      self._depth_renderer.enable_depth_rendering()
    mj_data = self._sync_mj_data(data)
    self._depth_renderer.update_scene(mj_data, camera=consts.MIDDLE_CAMERA_LEFT)
    return self._depth_renderer.render()
