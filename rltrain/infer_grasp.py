from isaacgym import gymapi, gymtorch
import torch
from env_grasping import GraspingEnv
from policy.mlp_policy import MLPPolicy
import os
import glob

# Isaac Gym 초기화
gym = gymapi.acquire_gym()

# 시뮬레이션 설정
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Viewer 생성
camera_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, camera_props)

# 로봇 asset 불러오기
asset_root = os.path.join(os.path.dirname(__file__), "assets")
robot_asset = gym.load_asset(sim, asset_root, "piper_description.urdf", gymapi.AssetOptions())

# 환경 객체 생성
NUM_ENVS = 10
env = GraspingEnv(gym, sim, num_envs=NUM_ENVS, robot_asset=robot_asset)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 정책 초기화
obs_dim = env.get_obs_size()
act_dim = env.get_act_size()
policy = MLPPolicy(obs_dim, act_dim).to(device)

# 최신 모델 자동 불러오기
model_files = sorted(glob.glob("models/trained_policy_*.pth"))
if not model_files:
    raise FileNotFoundError("No policy found in models/")
latest_model_path = model_files[-1]
print(f"📂 최신 정책 불러오기: {latest_model_path}")
policy.load_state_dict(torch.load(latest_model_path))
policy.eval()

# 추론 루프
while not gym.query_viewer_has_closed(viewer):
    observations = env.reset().to(device)
    total_rewards = torch.zeros(NUM_ENVS, device=device)

    for step in range(100):
        actions = policy.act(observations)
        next_obs, rewards, dones = env.step(torch.tensor(actions, device=device))
        observations = next_obs.to(device)

        # 누적 보상 기록
        total_rewards += rewards

        # Viewer 시각화
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    print(f"🔎 평균 보상: {total_rewards.mean().item():.3f}")

# 정리
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
