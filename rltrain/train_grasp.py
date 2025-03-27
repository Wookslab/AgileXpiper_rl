from isaacgym import gymapi, gymtorch
import torch
from env_grasping import GraspingEnv
from policy.mlp_policy import MLPPolicy
import os

# Isaac Gym 초기화
gym = gymapi.acquire_gym()

# 시뮬레이션 설정
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 로봇 asset 불러오기
asset_root = os.path.join(os.path.dirname(__file__), "assets")
robot_asset = gym.load_asset(sim, asset_root, "piper_description.urdf", gymapi.AssetOptions())

# 환경 객체 생성
env = GraspingEnv(gym, sim, num_envs=10, robot_asset=robot_asset)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 정책 생성
obs_dim = env.get_obs_size()
act_dim = env.get_act_size()
policy = MLPPolicy(obs_dim, act_dim).to(device)

# 옵티마이저
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 학습 루프
for epoch in range(100):
    observations = env.reset().to(device)
    for step in range(100):
        actions = policy.act(observations)
        next_obs, rewards, dones = env.step(torch.tensor(actions, device=device))
        observations = next_obs.to(device)
    optimizer.step()
    print(f"Epoch {epoch} complete")
    torch.cuda.synchronize()

gym.destroy_sim(sim)
