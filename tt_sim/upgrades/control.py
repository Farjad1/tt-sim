"""High-level controller upgrades for tt-sim."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tt_sim.interfaces import (
    Aimer,
    BallObservation,
    HighLevelController,
    Perceiver,
    Predictor,
    SpinEstimator,
    SwingPlanner,
)

if TYPE_CHECKING:
    pass


class ReplanController(HighLevelController):
    """Stage 1 – Re-planning closed-loop controller.

    Replans the full pipeline (predict → aim → swing) every REPLAN_INTERVAL
    frames (~31 Hz at dt=0.008s).  Between replans, holds the last commanded
    position.  Each replan generates a fresh quintic trajectory from the
    current joint state and returns the position at t = replan_dt into that
    trajectory (Option A: fixed lookahead clamped to trajectory duration).

    This provides closed-loop correction for prediction errors: as more ball
    observations accumulate, the predicted intercept becomes more accurate and
    the swing trajectory adapts.
    """

    REPLAN_INTERVAL = 4   # frames between replans (~31 Hz)
    MIN_OBS = 4           # fewer than open_loop since we keep correcting
    DT = 0.008            # sim timestep

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x

        self._observations: list[BallObservation] = []
        self._frame_count = 0
        self._last_action: np.ndarray | None = None
        self._prev_q: np.ndarray | None = None

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X (drag-aware)."""
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        # Drag-aware: sample predictor at fine time steps
        try:
            dt_sample = 0.004
            n_samples = int(self.t_predict / dt_sample)
            for i in range(1, n_samples + 1):
                t_probe = t_now + i * dt_sample
                state = self.predictor.predict(self._observations, t_probe)
                if state.position[0] >= self.ROBOT_X:
                    return t_probe
        except Exception:
            pass

        # Fallback: linear extrapolation
        o0 = self._observations[0]
        dt = o_last.timestamp - o0.timestamp
        if dt < 1e-6:
            return t_now + self.t_predict
        vx = (o_last.position[0] - o0.position[0]) / dt
        if vx < 0.1:
            return t_now + self.t_predict
        dx = self.ROBOT_X - o_last.position[0]
        return t_now + dx / vx

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        self._frame_count += 1

        should_replan = (
            len(self._observations) >= self.MIN_OBS
            and self._frame_count % self.REPLAN_INTERVAL == 0
        )

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            # Override t_contact with actual remaining time to intercept
            paddle_target.t_contact = max(t_intercept - obs.timestamp, 0.05)

            # Estimate current joint velocity for warm-start
            dq_current = None
            if self._prev_q is not None:
                dq_current = (q_current - self._prev_q) / (self.REPLAN_INTERVAL * self.DT)

            # Pass dq_current if swing planner supports it (QuinticSwingPlanner)
            try:
                trajectory = self.swing_planner.plan(paddle_target, q_current, dq_current=dq_current)
            except TypeError:
                trajectory = self.swing_planner.plan(paddle_target, q_current)

            self._prev_q = q_current.copy()

            # Look ahead: use a fraction of time-to-contact, not just one
            # replan interval.  This commits to more of the swing while still
            # allowing course correction at the next replan.
            replan_dt = self.REPLAN_INTERVAL * self.DT
            traj_times = trajectory.times
            traj_duration = traj_times[-1] - traj_times[0]
            # Look ahead by 2x replan interval or 30% of trajectory, whichever is larger
            dt_ahead = min(max(2 * replan_dt, 0.3 * traj_duration), traj_duration)
            t_query = traj_times[0] + dt_ahead

            n_joints = trajectory.positions.shape[1]
            self._last_action = np.array([
                np.interp(t_query, traj_times, trajectory.positions[:, j])
                for j in range(n_joints)
            ])

        if self._last_action is not None:
            return self._last_action

        return q_current

    def reset(self) -> None:
        """Clear observation history between rallies."""
        self._observations.clear()
        self._frame_count = 0
        self._last_action = None
        self._prev_q = None


class MPCController(HighLevelController):
    """Stage 2a – Receding-horizon MPC controller.

    Uses CasADi to solve a joint-space NLP at each replan cycle.
    Warm-starts from previous solution to maintain trajectory continuity.
    Replaces both swing planner and replan controller.
    """

    REPLAN_INTERVAL = 4   # frames between NLP re-solves
    MIN_OBS = 4
    DT = 0.008            # sim timestep

    def __init__(
        self,
        perceiver: Perceiver,
        predictor: Predictor,
        spin_estimator: SpinEstimator,
        aimer: Aimer,
        swing_planner: SwingPlanner,  # kept for interface compat, not used
        fk_dict: dict | None = None,
        t_predict: float = 0.5,
        robot_x: float = 1.3,
        horizon_N: int = 15,
        dt_mpc: float = 0.032,
        max_iter: int = 80,
        w_pos: float = 1000.0,
        w_ori: float = 100.0,
        w_vel: float = 1.0,
        w_smooth: float = 0.01,
        w_effort: float = 0.001,
        dq_max: float = 5.0,
    ) -> None:
        self.perceiver = perceiver
        self.predictor = predictor
        self.spin_estimator = spin_estimator
        self.aimer = aimer
        self.swing_planner = swing_planner
        self.t_predict = t_predict
        self.ROBOT_X = robot_x

        self._observations: list[BallObservation] = []
        self._frame_count = 0
        self._prev_q: np.ndarray | None = None

        # MPC params
        self.N = horizon_N
        self.dt_mpc = dt_mpc
        self.max_iter = max_iter
        self.dq_max = dq_max
        self.weights = {
            'pos': w_pos, 'ori': w_ori, 'vel': w_vel,
            'smooth': w_smooth, 'effort': w_effort,
        }

        # FK and NLP (built lazily)
        self._fk = fk_dict
        self._nlp_solver = None
        self._n_joints: int | None = None
        self._joint_ranges: np.ndarray | None = None

        # Warm-start state
        self._prev_solution: np.ndarray | None = None
        self._solution_trajectory: tuple | None = None
        self._solution_start_time: float | None = None

    def _build_nlp(self) -> None:
        """Build the CasADi NLP (called once)."""
        import casadi as ca

        nj = self._n_joints
        N = self.N
        dt = self.dt_mpc
        w = self.weights

        # Decision variables: q_1, ..., q_N (q_0 is parameter = current state)
        Q = ca.SX.sym('Q', N * nj)
        q0_param = ca.SX.sym('q0', nj)
        target_pos = ca.SX.sym('target_pos', 3)
        target_normal = ca.SX.sym('target_normal', 3)
        target_dq = ca.SX.sym('target_dq', nj)
        dt_param = ca.SX.sym('dt_param')  # adaptive timestep

        params = ca.vertcat(q0_param, target_pos, target_normal, target_dq, dt_param)

        fk_fn = self._fk['fk_fn']
        fk_rot_fn = self._fk['fk_rot_fn']

        cost = 0
        g = []  # constraints
        lbg = []
        ubg = []

        def get_q(k):
            if k == 0:
                return q0_param
            return Q[(k - 1) * nj: k * nj]

        # Smoothness and effort over horizon
        for k in range(1, N + 1):
            qk = get_q(k)
            qk_prev = get_q(k - 1)
            dq = (qk - qk_prev) / dt_param

            # Effort: minimize velocity
            cost += w['effort'] * ca.dot(dq, dq)

            # Velocity limits
            g.append(dq)
            lbg += [-self.dq_max] * nj
            ubg += [self.dq_max] * nj

            # Smoothness (acceleration penalty) for k >= 2
            if k >= 2:
                qk_prev2 = get_q(k - 2)
                ddq = (qk - 2 * qk_prev + qk_prev2) / (dt_param * dt_param)
                cost += w['smooth'] * ca.dot(ddq, ddq)

        # Terminal costs
        q_end = get_q(N)
        p_ee = fk_fn(q_end)
        _, R_flat = fk_rot_fn(q_end)
        # R_flat is row-major [R00,R01,R02,R10,...,R22]
        # Body Z-axis (column 2 of R) = [R02, R12, R22] = indices 2,5,8
        z_ee = ca.vertcat(R_flat[2], R_flat[5], R_flat[8])

        # Position tracking
        pos_err = p_ee - target_pos
        cost += w['pos'] * ca.dot(pos_err, pos_err)

        # Orientation tracking (cross product error)
        ori_err = ca.cross(z_ee, target_normal)
        cost += w['ori'] * ca.dot(ori_err, ori_err)

        # Velocity at end
        q_end_prev = get_q(N - 1)
        dq_end = (q_end - q_end_prev) / dt_param
        vel_err = dq_end - target_dq
        cost += w['vel'] * ca.dot(vel_err, vel_err)

        # Joint limits as bounds
        lbx = []
        ubx = []
        for k in range(N):
            for j in range(nj):
                lbx.append(float(self._joint_ranges[j, 0]))
                ubx.append(float(self._joint_ranges[j, 1]))

        # Build NLP
        nlp = {'x': Q, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
        opts = {
            'ipopt.max_iter': self.max_iter,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-3,
        }
        self._nlp_solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._n_params = params.shape[0]

    def _solve(
        self,
        q_current: np.ndarray,
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        target_dq: np.ndarray,
        dt_mpc: float,
    ) -> np.ndarray:
        """Solve the NLP, return (N, n_joints) trajectory."""
        nj = self._n_joints
        N = self.N

        p_val = np.concatenate([
            q_current, target_pos, target_normal, target_dq, [dt_mpc]
        ])

        # Initial guess: warm-start or hold
        if self._prev_solution is not None:
            x0 = np.zeros(N * nj)
            prev = self._prev_solution
            for k in range(N):
                src = min(k + 1, N - 1)
                x0[k * nj: (k + 1) * nj] = prev[src]
        else:
            x0 = np.tile(q_current, N)

        sol = self._nlp_solver(
            x0=x0, lbx=self._lbx, ubx=self._ubx,
            lbg=self._lbg, ubg=self._ubg, p=p_val,
        )

        Q_opt = np.array(sol['x']).flatten()
        trajectory = Q_opt.reshape(N, nj)
        self._prev_solution = trajectory.copy()
        return trajectory

    def _estimate_intercept_time(self) -> float:
        """Estimate when the ball will reach ROBOT_X."""
        if len(self._observations) < 2:
            return self._observations[-1].timestamp + self.t_predict

        o_last = self._observations[-1]
        t_now = o_last.timestamp

        try:
            dt_sample = 0.004
            n_samples = int(self.t_predict / dt_sample)
            for i in range(1, n_samples + 1):
                t_probe = t_now + i * dt_sample
                state = self.predictor.predict(self._observations, t_probe)
                if state.position[0] >= self.ROBOT_X:
                    return t_probe
        except Exception:
            pass

        o0 = self._observations[0]
        dt = o_last.timestamp - o0.timestamp
        if dt < 1e-6:
            return t_now + self.t_predict
        vx = (o_last.position[0] - o0.position[0]) / dt
        if vx < 0.1:
            return t_now + self.t_predict
        dx = self.ROBOT_X - o_last.position[0]
        return t_now + dx / vx

    def step(self, env_state: dict, q_current: np.ndarray) -> np.ndarray:
        obs = self.perceiver.observe(env_state)
        self._observations.append(obs)
        self._frame_count += 1

        if self._n_joints is None:
            self._n_joints = len(q_current)
            if self._fk is not None:
                self._joint_ranges = self._fk['joint_ranges']
                self._build_nlp()

        should_replan = (
            self._fk is not None
            and self._nlp_solver is not None
            and len(self._observations) >= self.MIN_OBS
            and self._frame_count % self.REPLAN_INTERVAL == 0
        )

        if should_replan:
            t_intercept = self._estimate_intercept_time()
            ball_state = self.predictor.predict(self._observations, t_intercept)
            ball_state.position[2] = np.clip(ball_state.position[2], 0.76, 2.0)
            spin = self.spin_estimator.estimate(self._observations)
            paddle_target = self.aimer.aim(ball_state, spin)
            t_remaining = max(t_intercept - obs.timestamp, 0.05)
            paddle_target.t_contact = t_remaining

            # Adaptive dt so horizon covers time to intercept
            dt_mpc = max(t_remaining / self.N, 0.008)

            # Compute desired joint velocity from Cartesian paddle velocity
            target_dq = np.zeros(self._n_joints)
            if (paddle_target.velocity is not None
                    and np.linalg.norm(paddle_target.velocity) > 1e-8):
                J_val = np.array(self._fk['fk_fn'].jacobian()(
                    q_current, np.zeros(3)))
                J = J_val[:3, :self._n_joints]
                damping = 0.01
                JJT = J @ J.T + damping**2 * np.eye(3)
                target_dq = J.T @ np.linalg.solve(JJT, paddle_target.velocity)

            trajectory = self._solve(
                q_current,
                paddle_target.position,
                paddle_target.normal,
                target_dq,
                dt_mpc,
            )

            # Store trajectory with timestamps for interpolation
            # Prepend current state so interp works between replans
            times = np.array([obs.timestamp] + [
                obs.timestamp + (k + 1) * dt_mpc for k in range(self.N)
            ])
            full_traj = np.vstack([q_current.reshape(1, -1), trajectory])
            self._solution_trajectory = (times, full_traj)

        # Interpolate from current solution with lookahead
        # PD control needs q_desired ahead of q_current to generate torque
        if self._solution_trajectory is not None:
            times, traj = self._solution_trajectory
            t_now = obs.timestamp
            t_end = times[-1]
            t_remain = max(t_end - t_now, 0.01)
            t_query = min(t_now + 0.6 * t_remain, t_end)
            result = np.array([
                np.interp(t_query, times, traj[:, j])
                for j in range(self._n_joints)
            ])
            return result

        return q_current

    def reset(self) -> None:
        """Clear state between rallies."""
        self._observations.clear()
        self._frame_count = 0
        self._prev_q = None
        self._prev_solution = None
        self._solution_trajectory = None
        self._solution_start_time = None
