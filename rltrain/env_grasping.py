import numpy as np
from isaacgym import gymapi
import torch


class GraspingEnv:
    def __init__(self, gym, sim, num_envs, robot_asset):
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = []
        self.robots = []
        self.boxes = []

        self._create_envs(robot_asset)

        # === 관측 및 행동 차원 수 설정 ===
        self.num_obs = self.get_obs_size()
        self.num_act = self.get_act_size()

    def _create_envs(self, robot_asset):
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = False

        box_dims = gymapi.Vec3(0.05, 0.05, 0.05)
        self.box_asset = self.gym.create_box(
            self.sim, box_dims.x, box_dims.y, box_dims.z, asset_options
        )

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, self.num_envs)

            # === 로봇 생성 ===
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0, 0, 0)
            robot = self.gym.create_actor(env, robot_asset, robot_pose, f"robot_{i}", i, 1)

            if robot is None:
                print(f"[ERROR] Failed to create robot actor in env {i}")
                continue
            print(f"[DEBUG] Robot actor: {robot}")

            # === 관절 초기화 ===
            dof_states = self.gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
            for j in range(dof_states.shape[0]):
                dof_states['pos'][j] = 0.0
                dof_states['vel'][j] = 0.0
            self.gym.set_actor_dof_states(env, robot, dof_states, gymapi.STATE_ALL)

            # === 박스 생성 ===
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0.3, 0.0, 0.05)
            box = self.gym.create_actor(env, self.box_asset, box_pose, f"box_{i}", i, 0)

            self.envs.append(env)
            self.robots.append(robot)
            self.boxes.append(box)

    def reset(self):
        observations = torch.zeros((self.num_envs, self.num_obs), device=self.device)  # 예시
        return observations

    def step(self, actions):
        # 강화학습 환경에서 매 타임스텝마다 실행할 동작
        # 현재는 간단히 관측값, 보상, 완료 여부 반환하는 형식으로 예시
        observations = self.reset()
        rewards = torch.zeros((self.num_envs,), device=self.device)
        dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return observations, rewards, dones

    def get_obs_size(self):
        # 관측 벡터의 크기 반환
        # 예: 모든 DOF의 개수라고 가정
        dof_state = self.gym.get_actor_dof_states(self.envs[0], self.robots[0], gymapi.STATE_ALL)
        return dof_state['pos'].shape[0]

    def get_act_size(self):
        # 행동 벡터의 크기 반환
        # 예: 관절 수와 동일하게 설정
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.robots[0])
        return dof_props.shape[0]

    def get_observation(self):       
        # 관측 반환 (예시)
        obs = []
        for env, robot in zip(self.envs, self.robots):
            dof_states = self.gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
            obs.append(dof_states['pos'])
        return np.array(obs)
