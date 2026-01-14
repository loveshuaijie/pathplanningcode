"""
3D APF (Artificial Potential Field) planner for MapEnv

Usage:
    from AxisPathPlanEnv.MapEnv import MapEnv
    from planners.apf_3d import apf_plan

    env = MapEnv(config)
    path = apf_plan(env, max_iters=2000, step_size=0.5, k_att=1.0, k_rep=50.0)
    # path: list of waypoints (each is np.array length 6: x,y,z,rx,ry,rz) if successful; else None
"""

import copy
import math
import numpy as np
from typing import List, Optional

def _clone_env_for_planning(env):
    """
    Create a fresh MapEnv instance with same configuration and obstacles,
    and reset it to the same start/target as env. This avoids mutating the
    original env during planning (MapEnv.plan_step mutates).
    """
    # Reconstruct config from env attributes
    cfg = {
        "envxrange": env.xrange,
        "envyrange": env.yrange,
        "envzrange": env.zrange,
        "obstacles_num": env.obstacles_num,
        "start": np.concatenate([env.startpos, env.startges]).tolist(),
        "target": np.concatenate([env.targetpos, env.targetges]).tolist(),
        "tool_size": env.tool_size,
        "maxstep": env.maxstep,
        "period": env.period,
        "safe_distance": env.safe_distance,
        "alpha_max": env.alpha_max,
        "reachpos_scale": env.reachpos_scale,
        "reachges_scale": env.reachges_scale,
        "Vmax": env.Vmax,
        # preserve goal_conditioned if present
        "goal_conditioned": getattr(env, "goal_conditioned", False)
    }
    from AxisPathPlanEnv.MapEnv import MapEnv
    plan_env = MapEnv(cfg)
    # set exact same obstacles (reuse objects)
    try:
        plan_env.set_obstacles(env.obstacles)
    except Exception:
        # fallback: try append
        plan_env.clear_obstacles()
        for o in env.obstacles:
            plan_env.append_obstacle(o)
    # reset with same start/target and no random obstacle generation
    plan_env.reset(start=np.concatenate([env.startpos, env.startges]),
                   target=np.concatenate([env.targetpos, env.targetges]),
                   generate_obstacles=False)
    return plan_env

def _to_waypoint(pos3, ori3):
    arr = np.zeros(6, dtype=float)
    arr[0:3] = pos3
    arr[3:6] = ori3
    return arr

def apf_plan(env,
             max_iters: int = 2000,
             step_size: float = 0.5,
             k_att: float = 1.0,
             k_rep: float = 50.0,
             repulse_range: float = 3.0,
             tol: float = 0.5,
             random_escape_prob: float = 0.1,
             max_random_escapes: int = 20) -> Optional[List[np.ndarray]]:
    """
    APF planner in 3D.

    Args:
      env: existing MapEnv instance (used to get config/obstacles/start/target)
      max_iters: max iterations for APF updates
      step_size: movement per iteration
      k_att: attractive force coefficient
      k_rep: repulsive force coefficient
      repulse_range: distance within which obstacles repel
      tol: distance to goal threshold to stop (meters)
      random_escape_prob: probability to attempt small random perturbation when stuck
      max_random_escapes: max number of random escape attempts total

    Returns:
      path: list of 6-dim waypoints (x,y,z,rx,ry,rz) or None if planning failed
    """
    plan_env = _clone_env_for_planning(env)
    start_pos = plan_env.startpos.copy()
    start_ori = plan_env.startges.copy()
    goal_pos = plan_env.targetpos.copy()
    goal_ori = plan_env.targetges.copy()

    current = start_pos.copy()
    path = [_to_waypoint(current, start_ori)]
    prev_norm = None
    escapes = 0

    for it in range(max_iters):
        # attractive force: proportional to vector to goal
        diff = goal_pos - current
        dist_to_goal = np.linalg.norm(diff)
        if dist_to_goal < tol:
            # reached
            path.append(_to_waypoint(goal_pos, goal_ori))
            return path

        f_att = k_att * diff  # vector

        # repulsive forces from obstacles (point obstacles approximated by center)
        f_rep = np.zeros(3, dtype=float)
        for obs in plan_env.obstacles:
            # distance from current to obstacle center (approx)
            obs_center = np.array(obs.centerPoint, dtype=float)
            d = np.linalg.norm(current - obs_center)
            if d < 1e-6:
                d = 1e-6
            if d <= repulse_range:
                # direction away from obstacle
                dir_away = (current - obs_center) / d
                # magnitude
                mag = k_rep * (1.0 / d - 1.0 / repulse_range) / (d * d)
                f_rep += mag * dir_away

        f_total = f_att + f_rep
        # normalize and scale by step_size (to get next position)
        norm = np.linalg.norm(f_total)
        if norm < 1e-6:
            # likely local minimum; try random escape
            if escapes < max_random_escapes and np.random.rand() < random_escape_prob:
                rand_dir = np.random.randn(3)
                rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-9)
                candidate = current + rand_dir * step_size
                escapes += 1
            else:
                # stuck -> fail
                return None
        else:
            direction = f_total / norm
            candidate = current + direction * step_size

        # enforce bounds
        candidate[0] = np.clip(candidate[0], plan_env.xrange[0], plan_env.xrange[1])
        candidate[1] = np.clip(candidate[1], plan_env.yrange[0], plan_env.yrange[1])
        candidate[2] = np.clip(candidate[2], plan_env.zrange[0], plan_env.zrange[1])

        # test collision by plan_step on cloned env (we use same orientation)
        res = plan_env.plan_step(candidate, plan_env.nowges)
        if res.get('success', False) and not res.get('collision', False):
            current = candidate
            path.append(_to_waypoint(current, start_ori))
            # continue
        else:
            # collision - try small orthogonal perturbations (simple local navigation)
            collided = True
            found = False
            for attempt in range(6):
                angle = (attempt / 6.0) * 2 * math.pi
                # create orthogonal small offset
                offset = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=float) * (step_size * 0.5)
                candidate2 = current + offset
                candidate2[0] = np.clip(candidate2[0], plan_env.xrange[0], plan_env.xrange[1])
                candidate2[1] = np.clip(candidate2[1], plan_env.yrange[0], plan_env.yrange[1])
                candidate2[2] = np.clip(candidate2[2], plan_env.zrange[0], plan_env.zrange[1])
                res2 = plan_env.plan_step(candidate2, plan_env.nowges)
                if res2.get('success', False) and not res2.get('collision', False):
                    current = candidate2
                    path.append(_to_waypoint(current, start_ori))
                    found = True
                    break
            if not found:
                # try random escape if allowed
                if escapes < max_random_escapes:
                    rand_dir = np.random.randn(3)
                    rand_dir /= (np.linalg.norm(rand_dir) + 1e-9)
                    candidate3 = current + rand_dir * step_size
                    candidate3[0] = np.clip(candidate3[0], plan_env.xrange[0], plan_env.xrange[1])
                    candidate3[1] = np.clip(candidate3[1], plan_env.yrange[0], plan_env.yrange[1])
                    candidate3[2] = np.clip(candidate3[2], plan_env.zrange[0], plan_env.zrange[1])
                    res3 = plan_env.plan_step(candidate3, plan_env.nowges)
                    if res3.get('success', False) and not res3.get('collision', False):
                        current = candidate3
                        path.append(_to_waypoint(current, start_ori))
                        escapes += 1
                        continue
                # cannot find non-colliding candidate -> fail
                return None

    # max_iters reached
    return None


# Example quick test (not executed on import)
if __name__ == "__main__":
    
    import sys

    sys.path.append('E:\pathplanning\pathplanningcode')
    from AxisPathPlanEnv.MapEnv import MapEnv
    # Provide your env_config path or dict here for testing
    cfg = {
        "envxrange": [-10, 10],
        "envyrange": [-10, 10],
        "envzrange": [-10, 10],
        "obstacles_num": 8,
        "start": [0,0,0,0,0,0],
        "target": [6,6,6,0,0,0],
        "tool_size": [2.0, 0.5],
        "maxstep": 1000,
        "period": 0.1,
        "safe_distance": 1.0,
        "alpha_max": math.pi/4,
        "reachpos_scale": 10.0,
        "reachges_scale": 10.0,
        "Vmax": 1.0
    }
    env = MapEnv(cfg)
    p = apf_plan(env)
    print("APF path:", p)