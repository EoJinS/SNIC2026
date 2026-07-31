import numpy as np

num_steps = 20
car_2_pos = np.array([25.0, 15.6, 1.8])
car_2_traj_x = np.linspace(car_2_pos[0], car_2_pos[0] + 0, num_steps)
car_2_traj_y = np.linspace(car_2_pos[1], car_2_pos[1] - 5.0, num_steps)
car_2_traj_z = np.linspace(car_2_pos[2], car_2_pos[2] + 0, num_steps)
car_2_traj = np.stack([car_2_traj_x, car_2_traj_y, car_2_traj_z], axis=1)

for i, pos in enumerate(car_2_traj):
    ue_pos = pos + [0, 0, 1.0]
    print(f"step {i}: car_2 {pos} ue {ue_pos}")
