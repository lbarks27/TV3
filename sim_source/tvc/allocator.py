"""Simple collective + differential control mixer.

Replaces the former Levenberg–Marquardt NLS allocator with a fixed structure:

1. **Collective a1** — uniform nested angle chosen so axial force matches
   ``force_target`` (binary search when force coupling is active; rest pose
   otherwise).
2. **Differential a2** — primary gimbal angles from a linear torque map
   ``τ ≈ τ0 + B · Δa2`` evaluated at the collective seed (one finite-diff
   column per active mount), solved with a small ridge regularizer.

This is intentionally small, deterministic, and PX4-friendly: no multi-seed
search, no iterative Jacobian, no line search.
"""

from __future__ import annotations

import numpy as np

from .geometry import Mount, thrust_direction, unit
from .parameters import SolverConfig


class TorqueAllocator:
    def __init__(
        self,
        mounts: list[Mount],
        solver: SolverConfig,
        force_axis: np.ndarray | list[float] | None = None,
        center_of_mass: np.ndarray | list[float] | None = None,
    ) -> None:
        self.mounts = mounts
        self.center_of_mass = np.asarray(
            [0.0, 0.0, 1.0] if center_of_mass is None else center_of_mass,
            dtype=float,
        )
        self.force_axis = unit(
            np.asarray(
                [0.0, 0.0, 1.0] if force_axis is None else force_axis,
                dtype=float,
            )
        )
        self.tol = float(solver.tol)
        self.finite_diff_eps = float(solver.finite_diff_eps)
        # Ridge on the a2 least-squares solve (legacy ``damping`` scaled).
        self.torque_ridge = max(float(solver.damping) ** 2, 1e-12)
        self.force_search_iters = max(1, int(getattr(solver, "force_search_iters", 16)))
        # Extra a2 re-linearizations after the first solve (still no LM).
        self.a2_polish_iters = max(0, int(getattr(solver, "a2_polish_iters", 2)))
        # Weights on τx/τz in the a2 LS solve (<1 → that axis yields to pitch).
        self.roll_torque_weight = float(
            np.clip(getattr(solver, "roll_torque_weight", 1.0), 0.0, 1e3)
        )
        self.yaw_torque_weight = float(
            np.clip(getattr(solver, "yaw_torque_weight", 1.0), 0.0, 1e3)
        )
        # Kept for call-site / JSON compatibility; unused by this mixer.
        self.damping = float(solver.damping)
        self.step_scale = float(solver.step_scale)
        self.max_iters = int(solver.max_iters)
        self.tracking_iters = max(0, int(solver.tracking_iters))
        self.motion_damping = float(np.clip(solver.motion_damping, 0.0, 1.0))
        self.reseed_interval = max(0, int(solver.reseed_interval))
        self.refine_passes = max(0, int(solver.refine_passes))

        self.rest_angles = np.vstack([mount.rest_angles for mount in mounts])
        self.angles = self.rest_angles.copy()
        self.allocate_count = 0
        self.min_angles = np.vstack([mount.min_angles for mount in mounts])
        self.max_angles = np.vstack([mount.max_angles for mount in mounts])
        self.motion_scale = max(
            float(np.linalg.norm(self.max_angles - self.min_angles)), 1e-9
        )
        self.torque_upper_bound = sum(
            np.linalg.norm(mount.position - self.center_of_mass) * mount.thrust
            for mount in mounts
        )
        self.force_upper_bound = sum(mount.thrust for mount in mounts)

    def _a1_search_span_deg(self) -> tuple[float, float, float]:
        """Return (a1_min, a1_max, a1_rest) in degrees for collective search."""
        a1_min = float(np.min(np.rad2deg(self.min_angles[:, 1])))
        a1_max = float(np.max(np.rad2deg(self.max_angles[:, 1])))
        a1_rest = float(np.rad2deg(self.rest_angles[0, 1]))
        return a1_min, a1_max, a1_rest

    @staticmethod
    def _force_coupling_active(
        force_target: np.ndarray | float | None,
        force_weight: float,
    ) -> bool:
        return force_weight > 0.0 and force_target is not None

    @staticmethod
    def _force_target_magnitude(force_target: np.ndarray | float | None) -> float:
        if force_target is None:
            return 0.0
        return float(np.linalg.norm(np.asarray(force_target, dtype=float)))

    def refresh_limits(self) -> None:
        self.torque_upper_bound = sum(
            np.linalg.norm(mount.position - self.center_of_mass) * mount.thrust
            for mount in self.mounts
        )
        self.force_upper_bound = sum(mount.thrust for mount in self.mounts)
        self.min_angles = np.vstack([mount.min_angles for mount in self.mounts])
        self.max_angles = np.vstack([mount.max_angles for mount in self.mounts])
        self.motion_scale = max(
            float(np.linalg.norm(self.max_angles - self.min_angles)), 1e-9
        )

    def clip_angles(self, angles: np.ndarray) -> np.ndarray:
        return np.clip(angles, self.min_angles, self.max_angles)

    def twr_lock_angles(
        self,
        thrusts: np.ndarray,
        weight: float,
        twr: float = 0.75,
    ) -> np.ndarray:
        """Uniform a1, a2=0 so axial force is ``twr * weight``. Attitude off."""
        thrusts_arr = np.asarray(thrusts, dtype=float)
        target = float(twr) * max(float(weight), 1e-6)
        a1_min, a1_max, a1_rest = self._a1_search_span_deg()
        f_rest = self._uniform_a1_axial_force(a1_rest, thrusts_arr)
        f_fold = self._uniform_a1_axial_force(a1_max, thrusts_arr)
        if target >= f_rest - 1e-9:
            return self._uniform_a1_angles(a1_rest)
        if target <= f_fold + 1e-9:
            return self._uniform_a1_angles(a1_max)
        lo, hi = a1_rest, a1_max
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if self._uniform_a1_axial_force(mid, thrusts_arr) > target:
                lo = mid
            else:
                hi = mid
        return self._uniform_a1_angles(0.5 * (lo + hi))

    def constrain_angles_for_thrusts(
        self, angles: np.ndarray, thrusts: np.ndarray | None = None
    ) -> np.ndarray:
        if thrusts is None:
            return angles
        constrained = np.array(angles, copy=True)
        inactive = np.asarray(thrusts, dtype=float) <= 1e-9
        if np.any(inactive):
            constrained[inactive] = self.rest_angles[inactive]
        return constrained

    def angle_utilization(self, angles: np.ndarray) -> float:
        up_deflect = np.maximum(angles - self.rest_angles, 0.0)
        down_deflect = np.maximum(self.rest_angles - angles, 0.0)
        up_travel = np.maximum(self.max_angles - self.rest_angles, 1e-9)
        down_travel = np.maximum(self.rest_angles - self.min_angles, 1e-9)
        utilization = np.maximum(up_deflect / up_travel, down_deflect / down_travel)
        return float(np.max(utilization))

    def torque_force_dirs(
        self,
        angles: np.ndarray,
        thrusts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_torque = np.zeros(3, dtype=float)
        total_force = np.zeros(3, dtype=float)
        dirs = np.zeros((len(self.mounts), 3), dtype=float)
        for i, mount in enumerate(self.mounts):
            ax, ay = angles[i]
            direction = thrust_direction(ax, ay, mount)
            dirs[i] = direction
            thrust = float(mount.thrust if thrusts is None else thrusts[i])
            force = thrust * direction
            total_force += force
            total_torque += np.cross(mount.position - self.center_of_mass, force)
        return total_torque, total_force, dirs

    def torque_and_dirs(
        self, angles: np.ndarray, thrusts: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        torque, _, dirs = self.torque_force_dirs(angles, thrusts)
        return torque, dirs

    def force_and_dirs(
        self, angles: np.ndarray, thrusts: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        _, force, dirs = self.torque_force_dirs(angles, thrusts)
        return force, dirs

    def axial_force(self, force: np.ndarray) -> float:
        return float(np.dot(force, self.force_axis))

    def rest_axial_force(self, thrusts: np.ndarray) -> float:
        _, force, _ = self.torque_force_dirs(self.rest_angles, thrusts)
        return self.axial_force(force)

    def _resolve_thrusts(self, thrusts: np.ndarray | None) -> np.ndarray:
        if thrusts is None:
            return np.array([mount.thrust for mount in self.mounts], dtype=float)
        return np.asarray(thrusts, dtype=float)

    def _torque_tolerance(self, target_torque: np.ndarray) -> float:
        target_norm = float(np.linalg.norm(target_torque))
        # Slightly looser than LM: linear a2 map has small nonlinear residual.
        return max(self.tol * 10.0, 5e-3, 0.03 * max(target_norm, 1.0))

    def _uniform_a1_angles(self, a1_deg: float) -> np.ndarray:
        candidate = self.rest_angles.copy()
        candidate[:, 0] = 0.0
        candidate[:, 1] = np.deg2rad(a1_deg)
        return self.clip_angles(candidate)

    def _uniform_a1_axial_force(self, a1_deg: float, thrusts: np.ndarray) -> float:
        _, force, _ = self.torque_force_dirs(self._uniform_a1_angles(a1_deg), thrusts)
        return self.axial_force(force)

    def _axial_force_after_torque_null(
        self,
        a1_deg: float,
        thrusts: np.ndarray,
        target_torque: np.ndarray,
    ) -> float:
        """Axial force at uniform a1 after differential a2 tracks ``target_torque``.

        Under engine-out / thrust imbalance, a2=0 axial search picks a low-a1
        pose with huge bias torque that a2 cannot cancel. Evaluating force
        *after* the torque solve keeps the collective search on feasible poses.
        """
        seed = self.constrain_angles_for_thrusts(
            self._uniform_a1_angles(a1_deg), thrusts
        )
        angles = self._solve_differential_a2(seed, target_torque, thrusts)
        _, force, _ = self.torque_force_dirs(angles, thrusts)
        return self.axial_force(force)

    def _collective_a1_deg(
        self,
        force_target_mag: float,
        thrusts: np.ndarray,
        *,
        force_active: bool,
        target_torque: np.ndarray | None = None,
    ) -> float:
        """Pick uniform nested angle for axial force (degrees)."""
        a1_min, a1_max, a1_rest = self._a1_search_span_deg()
        if not force_active:
            return a1_rest

        tau_ref = (
            np.zeros(3, dtype=float)
            if target_torque is None
            else np.asarray(target_torque, dtype=float)
        )
        # Use torque-null-aware force when any mount is dead/weak; otherwise the
        # cheap a2=0 model matches the symmetric-thrust case closely.
        thrusts_arr = np.asarray(thrusts, dtype=float)
        active = thrusts_arr > 1e-9
        n_active = int(np.count_nonzero(active))
        imbalance = False
        if n_active > 0:
            t_live = thrusts_arr[active]
            imbalance = n_active < len(self.mounts) or (
                float(np.max(t_live)) > 1e-9
                and float(np.min(t_live)) / float(np.max(t_live)) < 0.85
            )

        def pose_metrics(a1: float) -> tuple[float, float]:
            """Return (Fax, |τ − τ_ref|) after differential a2 (or a2=0 model)."""
            if imbalance:
                seed = self.constrain_angles_for_thrusts(
                    self._uniform_a1_angles(a1), thrusts_arr
                )
                angles = self._solve_differential_a2(seed, tau_ref, thrusts_arr)
                tau, force, _ = self.torque_force_dirs(angles, thrusts_arr)
                return self.axial_force(force), float(np.linalg.norm(tau - tau_ref))
            fax = self._uniform_a1_axial_force(a1, thrusts_arr)
            return fax, 0.0

        # With unequal motor thrust, the differential axes can reach their travel
        # limits. Use the torque tolerance to find the achievable axial force range.
        n_sample = max(self.force_search_iters * 2, 24)
        a1_grid = np.linspace(a1_min, a1_max, n_sample)
        samples: list[tuple[float, float, float]] = []
        for a1 in a1_grid:
            fax, tau_err = pose_metrics(float(a1))
            samples.append((float(a1), float(fax), float(tau_err)))

        tau_tol = self._torque_tolerance(tau_ref)
        feasible = [s for s in samples if s[2] <= tau_tol]
        if imbalance and feasible:
            faxes = [s[1] for s in feasible]
            f_low, f_high = min(faxes), max(faxes)
            target = float(np.clip(force_target_mag, f_low, f_high))
            # Nearest feasible sample to the (clipped) force target.
            best = min(feasible, key=lambda s: abs(s[1] - target))
            return best[0]

        # Symmetric / no feasible-torque subset: classic axial binary search.
        f_lo = samples[0][1]
        f_hi = samples[-1][1]
        f_low = min(f_lo, f_hi)
        f_high = max(f_lo, f_hi)
        if force_target_mag <= f_low + 1e-9:
            return a1_min if f_lo <= f_hi else a1_max
        if force_target_mag >= f_high - 1e-9:
            return a1_max if f_hi >= f_lo else a1_min
        target = float(np.clip(force_target_mag, f_low, f_high))
        lo = a1_min
        hi = a1_max
        increasing = f_hi >= f_lo
        for _ in range(self.force_search_iters):
            mid = 0.5 * (lo + hi)
            f_mid, _ = pose_metrics(mid)
            if increasing:
                if f_mid < target:
                    lo = mid
                else:
                    hi = mid
            else:
                if f_mid < target:
                    hi = mid
                else:
                    lo = mid
        return 0.5 * (lo + hi)

    def _a2_effectiveness(
        self, seed: np.ndarray, thrusts: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (τ0, B 3×k, active_indices) for primary-axis differentials."""
        tau0, _, _ = self.torque_force_dirs(seed, thrusts)
        indices = [i for i, is_active in enumerate(active) if is_active]
        if not indices:
            return tau0, np.zeros((3, 0), dtype=float), indices

        eps = self.finite_diff_eps
        b_cols: list[np.ndarray] = []
        for i in indices:
            perturbed = seed.copy()
            perturbed[i, 0] += eps
            tau_eps, _, _ = self.torque_force_dirs(perturbed, thrusts)
            b_cols.append((tau_eps - tau0) / eps)
        return tau0, np.column_stack(b_cols), indices

    def _solve_a2_step(
        self,
        current: np.ndarray,
        target_torque: np.ndarray,
        thrusts: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """One ridge least-squares update of primary angles from current pose."""
        tau0, b_mat, indices = self._a2_effectiveness(current, thrusts, active)
        angles = current.copy()
        if not indices:
            return self.constrain_angles_for_thrusts(angles, thrusts)

        residual = np.asarray(target_torque, dtype=float) - tau0
        # Weighted LS: pitch full priority; roll/yaw scaled (heading/roll last).
        w = np.array(
            [
                float(self.roll_torque_weight),
                1.0,
                float(self.yaw_torque_weight),
            ],
            dtype=float,
        )
        b_w = b_mat * w[:, None]
        r_w = residual * w
        bt = b_w.T
        gram = bt @ b_w + self.torque_ridge * np.eye(b_w.shape[1])
        rhs = bt @ r_w
        try:
            delta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        for col, mount_idx in enumerate(indices):
            angles[mount_idx, 0] = current[mount_idx, 0] + float(delta[col])
        angles = self.clip_angles(angles)
        return self.constrain_angles_for_thrusts(angles, thrusts)

    def _solve_differential_a2(
        self,
        seed: np.ndarray,
        target_torque: np.ndarray,
        thrusts: np.ndarray,
    ) -> np.ndarray:
        """Apply linear Δa2 on top of collective seed; optional re-linearize."""
        active = np.asarray(thrusts, dtype=float) > 1e-9
        angles = self._solve_a2_step(seed, target_torque, thrusts, active)
        for _ in range(self.a2_polish_iters):
            angles = self._solve_a2_step(angles, target_torque, thrusts, active)
        return angles

    def _retrim_collective_a1(
        self,
        angles: np.ndarray,
        force_target_mag: float,
        thrusts: np.ndarray,
    ) -> np.ndarray:
        """Hold a2 fixed; binary-search uniform a1 to restore axial force.

        Differential a2 slightly drops axial force; this recovers F without a
        full NLS solve. Inactive mounts stay at rest.
        """
        a1_min, a1_max, _a1_rest = self._a1_search_span_deg()
        active = np.asarray(thrusts, dtype=float) > 1e-9
        if not np.any(active):
            return angles

        def axial_at(a1_deg: float) -> float:
            trial = angles.copy()
            trial[active, 1] = np.deg2rad(a1_deg)
            trial = self.clip_angles(trial)
            trial = self.constrain_angles_for_thrusts(trial, thrusts)
            _, force, _ = self.torque_force_dirs(trial, thrusts)
            return self.axial_force(force)

        f_lo = axial_at(a1_min)
        f_hi = axial_at(a1_max)
        f_low = min(f_lo, f_hi)
        f_high = max(f_lo, f_hi)
        if force_target_mag <= f_low + 1e-9:
            a1_deg = a1_min if f_lo <= f_hi else a1_max
        elif force_target_mag >= f_high - 1e-9:
            a1_deg = a1_max if f_hi >= f_lo else a1_min
        else:
            target = float(np.clip(force_target_mag, f_low, f_high))
            lo, hi = a1_min, a1_max
            increasing = f_hi >= f_lo
            # Half the iters of the free collective search — a2 already near the answer.
            iters = max(4, self.force_search_iters // 2)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                f_mid = axial_at(mid)
                if increasing:
                    if f_mid < target:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if f_mid < target:
                        hi = mid
                    else:
                        lo = mid
            a1_deg = 0.5 * (lo + hi)
        out = angles.copy()
        out[active, 1] = np.deg2rad(a1_deg)
        out = self.clip_angles(out)
        return self.constrain_angles_for_thrusts(out, thrusts)

    def mix(
        self,
        target_torque: np.ndarray,
        force_target: np.ndarray | float | None = None,
        force_weight: float = 0.0,
        thrusts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Collective a1 + differential a2 mixer (single shot).

        Pipeline:
        1. Uniform a1 for force (or rest if force coupling off)
        2. Linear a2 map for torque (+ polish re-linearizations)
        3. If force-coupled: re-trim uniform a1 with a2 held, then one a2 polish
        """
        thrusts = self._resolve_thrusts(thrusts)
        force_target_mag = self._force_target_magnitude(force_target)
        force_active = self._force_coupling_active(force_target, force_weight)
        target_torque = np.asarray(target_torque, dtype=float)

        a1_deg = self._collective_a1_deg(
            force_target_mag,
            thrusts,
            force_active=force_active,
            target_torque=target_torque,
        )
        seed = self._uniform_a1_angles(a1_deg)
        seed = self.constrain_angles_for_thrusts(seed, thrusts)
        angles = self._solve_differential_a2(seed, target_torque, thrusts)

        if force_active:
            # Torque-priority re-trim: uniform a1 after a2 restores force but can
            # reintroduce large bias torque under engine-out / thrust imbalance.
            # Keep the force re-trim only when residual |τ| does not get worse
            # beyond a small slack (attitude authority wins over altitude mix).
            active = np.asarray(thrusts, dtype=float) > 1e-9
            tau_before, _, _ = self.torque_force_dirs(angles, thrusts)
            err_before = float(np.linalg.norm(target_torque - tau_before))

            candidate = self._retrim_collective_a1(angles, force_target_mag, thrusts)
            # Extra a2 polish under imbalance (dead/weak mounts shrink B rank).
            polish = max(1, self.a2_polish_iters)
            n_active = int(np.count_nonzero(active))
            if n_active < len(self.mounts):
                polish = max(polish, 3)
            for _ in range(polish):
                candidate = self._solve_a2_step(
                    candidate, target_torque, thrusts, active
                )

            tau_after, _, _ = self.torque_force_dirs(candidate, thrusts)
            err_after = float(np.linalg.norm(target_torque - tau_after))
            # Accept force re-trim if torque tracking stays comparable.
            if err_after <= err_before * 1.15 + 1e-3:
                angles = candidate
            # else: keep pre-re-trim angles (torque priority)

        torque, force, dirs = self.torque_force_dirs(angles, thrusts)
        torque_error = float(np.linalg.norm(target_torque - torque))
        saturated = torque_error > self._torque_tolerance(target_torque)
        if force_active:
            force_error = abs(force_target_mag - self.axial_force(force))
            force_tol = max(0.08, 0.02 * max(force_target_mag, 1.0))
            if force_error > force_tol:
                saturated = True
        return angles, torque, dirs, saturated

    def solve(
        self,
        target_torque: np.ndarray,
        angles: np.ndarray,
        direction: np.ndarray | None = None,
        parallel_weight: float = 1.0,
        perp_weight: float = 1.0,
        force_target: np.ndarray | float | None = None,
        force_weight: float = 0.0,
        use_line_search: bool = False,
        thrusts: np.ndarray | None = None,
        fast: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """API-compatible entry: ignores seed/LM kwargs; runs the simple mixer."""
        del direction, parallel_weight, perp_weight, use_line_search, fast, angles
        mixed, torque, dirs, _ = self.mix(
            target_torque,
            force_target=force_target,
            force_weight=force_weight,
            thrusts=thrusts,
        )
        return mixed, torque, dirs

    def allocate_force_track(
        self,
        target_torque: np.ndarray,
        force_target: np.ndarray | float | None,
        force_weight: float,
        thrusts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Flight-path mixer (same algorithm as ``allocate``)."""
        angles, torque, dirs, saturated = self.mix(
            target_torque,
            force_target=force_target,
            force_weight=force_weight,
            thrusts=thrusts,
        )
        self.angles = angles
        return angles, torque, dirs, saturated

    def allocate(
        self,
        target_torque: np.ndarray,
        force_target: np.ndarray | float | None = None,
        force_weight: float = 0.0,
        thrusts: np.ndarray | None = None,
        *,
        fast: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Interactive / UI path — identical to the simple flight mixer."""
        del fast
        self.allocate_count += 1
        return self.allocate_force_track(
            target_torque,
            force_target=force_target,
            force_weight=force_weight,
            thrusts=thrusts,
        )

    def max_torque_along(
        self,
        direction: np.ndarray,
        angles: np.ndarray,
        force_target: np.ndarray | float | None = None,
        force_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Scale torque along ``direction`` until the mixer saturates."""
        del angles
        direction_unit = unit(direction)
        if float(np.linalg.norm(direction_unit)) <= 0.0:
            mixed, torque, dirs, _ = self.mix(
                np.zeros(3, dtype=float),
                force_target=force_target,
                force_weight=force_weight,
            )
            return mixed, torque, dirs, 0.0

        lo = 0.0
        hi = max(self.torque_upper_bound, 1.0)
        best_angles = self.rest_angles.copy()
        best_torque = np.zeros(3, dtype=float)
        best_dirs = np.zeros((len(self.mounts), 3), dtype=float)
        best_proj = 0.0

        for _ in range(20):
            mid = 0.5 * (lo + hi)
            target = direction_unit * mid
            trial_angles, trial_torque, trial_dirs, saturated = self.mix(
                target,
                force_target=force_target,
                force_weight=force_weight,
            )
            proj = float(np.dot(trial_torque, direction_unit))
            err = float(np.linalg.norm(target - trial_torque))
            if not saturated and err <= self._torque_tolerance(target):
                best_angles = trial_angles
                best_torque = trial_torque
                best_dirs = trial_dirs
                best_proj = proj
                lo = mid
            else:
                hi = mid

        if best_proj <= 0.0:
            trial_angles, trial_torque, trial_dirs, _ = self.mix(
                direction_unit * hi,
                force_target=force_target,
                force_weight=force_weight,
            )
            best_angles = trial_angles
            best_torque = trial_torque
            best_dirs = trial_dirs
            best_proj = float(np.dot(trial_torque, direction_unit))

        self.angles = best_angles
        return best_angles, best_torque, best_dirs, best_proj
