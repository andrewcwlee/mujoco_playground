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
"""Bimanual cube pickup with active-vision tracking for AV-ALOHA."""

import os
from typing import Any, Dict, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.av_aloha import (
    av_aloha_constants as consts,
)
from mujoco_playground._src.manipulation.av_aloha import base as av_aloha_base


def default_config() -> config_dict.ConfigDict:
  # Mac dev: export MJX_IMPL=jax. Workstation default is warp.
  impl = os.environ.get("MJX_IMPL", "warp")
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.015,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_cube=2.0,
              cube_lifted=4.0,
              vision_track=2.0,
              no_table_collision=0.3,
          ),
      ),
      impl=impl,
      naconmax=24 * 2048,
      njmax=88,
  )


class CubePickup(av_aloha_base.AvAlohaEnv):
  """Bimanual cube pickup; middle arm camera must keep the cube in view."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[
          Dict[str, Union[str, int, list[Any]]]
      ] = None,
  ):
    super().__init__(
        xml_path=(consts.XML_PATH / "mjx_cube_pickup.xml").as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self):
    self._post_init_av_aloha(keyframe="home")
    self._mocap_target = self._mj_model.body("mocap_target").mocapid
    self._cube_body = self._mj_model.body("cube").id
    self._cube_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._cube_body]
    ]
    self._cube_geom = self._mj_model.geom("cube").id
    # Cube center at rest = table top + half cube extent. Set so cube_lifted
    # reads ~0 at rest and saturates at 1 once the cube is ~15 cm higher.
    self._table_z = float(self._mj_model.geom_size[self._cube_geom][2])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_cube_x, rng_cube_y, rng_target_z = jax.random.split(rng, 4)
    cube_xy = jp.array([
        jax.random.uniform(rng_cube_x, (), minval=-0.10, maxval=0.10),
        jax.random.uniform(rng_cube_y, (), minval=-0.05, maxval=0.10),
    ])
    init_q = self._init_q.at[self._cube_qadr : self._cube_qadr + 2].add(cube_xy)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    target_z = jax.random.uniform(rng_target_z, (), minval=0.18, maxval=0.28)
    target_pos = jp.array([0.0, 0.0, 0.0]).at[2].set(target_z)
    target_pos = target_pos.at[:2].set(cube_xy)
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target].set(target_pos)
    )

    info = {
        "rng": rng,
        "target_pos": target_pos,
        "prev_potential": jp.array(0.0, dtype=float),
        "_steps": jp.array(0, dtype=int),
    }

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    newly_reset = state.info["_steps"] == 0
    state.info["prev_potential"] = jp.where(
        newly_reset, 0.0, state.info["prev_potential"]
    )

    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    potential = sum(rewards.values()) / sum(
        self._config.reward_config.scales.values()
    )
    reward = jp.maximum(
        potential - state.info["prev_potential"], jp.zeros_like(potential)
    )
    state.info["prev_potential"] = jp.maximum(
        potential, state.info["prev_potential"]
    )
    reward = jp.where(newly_reset, 0.0, reward)

    cube_pos = data.xpos[self._cube_body]
    out_of_bounds = jp.any(jp.abs(cube_pos) > 1.0)
    out_of_bounds |= cube_pos[2] < -0.05
    done = (
        out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

    state.info["_steps"] += self._config.action_repeat
    state.info["_steps"] = jp.where(
        done | (state.info["_steps"] >= self._config.episode_length),
        0,
        state.info["_steps"],
    )
    state.metrics.update(**rewards, out_of_bounds=out_of_bounds.astype(float))

    obs = self._get_obs(data, state.info)
    return mjx_env.State(
        data, obs, reward, done.astype(float), state.metrics, state.info
    )

  def _get_reward(
      self, data: mjx.Data, info: Dict[str, Any]
  ) -> Dict[str, Any]:
    def distance(x, y):
      return jp.exp(-10.0 * jp.linalg.norm(x - y))

    cube = data.xpos[self._cube_body]
    l_gripper = data.site_xpos[self._left_gripper_site]
    r_gripper = data.site_xpos[self._right_gripper_site]

    # Both grippers approach cube.
    gripper_cube = 0.5 * (distance(l_gripper, cube) + distance(r_gripper, cube))

    # Smooth height-based lift bonus over a 15 cm range above the table.
    cube_lifted = jp.clip((cube[2] - self._table_z) / 0.15, 0.0, 1.0)

    # Active-vision tracking: align middle camera's forward axis to the cube.
    # Camera forward axis is the third column of site_xmat (z-axis).
    cam_pos = data.site_xpos[self._middle_camera_site]
    cam_xmat = data.site_xmat[self._middle_camera_site].reshape(3, 3)
    cam_forward = cam_xmat[:, 2]
    to_cube = cube - cam_pos
    to_cube_norm = to_cube / (jp.linalg.norm(to_cube) + 1e-6)
    cos_angle = jp.dot(cam_forward, to_cube_norm)
    # Soft bonus that ramps up as alignment improves; ~1.0 inside an 18-deg cone.
    vision_track = jp.clip((cos_angle - 0.6) / 0.35, 0.0, 1.0)

    table_collision = self.hand_table_collision(data)

    return {
        "gripper_cube": gripper_cube,
        "cube_lifted": cube_lifted,
        "vision_track": vision_track,
        "no_table_collision": 1.0 - table_collision,
    }

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    left_gripper_pos = data.site_xpos[self._left_gripper_site]
    left_gripper_mat = data.site_xmat[self._left_gripper_site]
    right_gripper_pos = data.site_xpos[self._right_gripper_site]
    right_gripper_mat = data.site_xmat[self._right_gripper_site]
    middle_cam_pos = data.site_xpos[self._middle_camera_site]
    middle_cam_mat = data.site_xmat[self._middle_camera_site]
    cube_pos = data.xpos[self._cube_body]
    cube_mat = data.xmat[self._cube_body]
    finger_qposadr = data.qpos[self._finger_qposadr]
    cube_half_width = self.mjx_model.geom_size[self._cube_geom][1]

    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        finger_qposadr - cube_half_width,
        cube_pos,
        cube_mat.ravel()[3:],  # 6D rotation (drop first row).
        left_gripper_pos,
        left_gripper_mat.ravel()[3:],
        right_gripper_pos,
        right_gripper_mat.ravel()[3:],
        middle_cam_pos,
        middle_cam_mat.ravel()[3:],
        cube_pos - info["target_pos"],
        (info["_steps"].reshape((1,)) / self._config.episode_length).astype(
            float
        ),
    ])
    return obs
