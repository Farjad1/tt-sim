"""Trajectory prediction upgrades for tt-sim."""

from __future__ import annotations

import numpy as np

from tt_sim.interfaces import (
    BallObservation,
    BallState,
    Predictor,
)


# ── Physics constants ────────────────────────────────────────────────────────

C_D: float = 0.4          # drag coefficient (sphere in turbulent regime)
RHO_AIR: float = 1.225    # kg/m^3, air density at sea level
BALL_MASS: float = 0.0027 # kg, ITTF regulation ball mass
BALL_RADIUS: float = 0.02 # m, ITTF regulation ball radius
BALL_AREA: float = np.pi * BALL_RADIUS ** 2  # m^2, cross-sectional area
RESTITUTION: float = 0.91 # coefficient of restitution (ball–table)
C_L: float = 0.55         # lift (Magnus) coefficient
GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81])


class DragBouncePredictor(Predictor):
    """Stage 1 – Analytic gravity + aerodynamic drag predictor.

    Algorithm
    ---------
    Integrates the ODE for a sphere under gravity and quadratic drag::

        dv/dt = g - (C_D * rho_air * A) / (2 * m) * |v| * v

    where *A* is the cross-sectional area of the ball.  Table bounces are
    detected via event handling (z ≤ 0) and resolved with the coefficient
    of restitution (e = 0.91).

    The drag coefficient C_D can optionally be fit from trajectory data
    using least-squares minimisation of position residuals.

    Mathematics
    -----------
    State vector y = [x, v] ∈ ℝ⁶.

    dy/dt = [v,  g - (C_D ρ A / 2m) |v| v]

    Bounce condition:  z(t) = 0  →  v_z ← -e * v_z

    Implementation
    --------------
    * ``scipy.integrate.solve_ivp`` with RK45 and dense output.
    * Terminal event for table-plane crossing.

    References
    ----------
    * Huang, Y. et al. (2011). "Trajectory prediction of spinning ball for
      ping-pong player robot." *IEEE/RSJ IROS*.

    Datasets
    --------
    Fit / validate with any recorded ball-trajectory dataset (≥ 50 samples).

    Getting Started
    ---------------
    1. Instantiate with default constants or supply measured C_D.
    2. Call ``predict(observations, t_future)`` with ≥ 2 observations.
    3. Internally performs linear velocity bootstrap then integrates forward.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("DragBouncePredictor.predict")


class ResidualMLPPredictor(Predictor):
    """Stage 2a – Physics-informed residual MLP predictor.

    Algorithm
    ---------
    Augments the analytic drag ODE with a learned residual from a small
    multi-layer perceptron (MLP)::

        dv/dt = physics(x, v) + f_θ(x, v)

    where *physics* is the gravity + drag term and *f_θ* is a 2-layer MLP
    (hidden dims 64) with ReLU activations trained to minimise position
    prediction error over recorded trajectories.

    Mathematics
    -----------
    dy/dt = [v,  g - k_drag |v| v + f_θ(x, v)]

    Loss = Σ_i || x̂(t_i) - x(t_i) ||²

    Implementation
    --------------
    * PyTorch for the MLP.
    * ``torchdiffeq.odeint`` for differentiable ODE integration during
      training.

    References
    ----------
    * Achterhold, J. et al. (2023). "Physics-informed residual learning for
      table tennis trajectory prediction." *L4DC*.

    Datasets
    --------
    Kienzle 50 k trajectory dataset (position + timestamp).

    Getting Started
    ---------------
    1. Pre-train by loading a ``DragBouncePredictor`` as the physics prior.
    2. Train the MLP residual on recorded data via ``train()`` method.
    3. Call ``predict()`` for combined physics + learned prediction.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("ResidualMLPPredictor.predict")


class NeuralODEPredictor(Predictor):
    """Stage 2b – Fully learned Neural ODE predictor.

    Algorithm
    ---------
    Learns the full dynamics from data without an explicit physics prior::

        dx/dt = f_θ(x)

    where *f_θ* is a 3-layer MLP with 64 hidden units per layer and
    SiLU/Swish activations.  Training uses the adjoint sensitivity method
    for memory-efficient backpropagation through the ODE solver.

    Mathematics
    -----------
    State y = [x, v] ∈ ℝ⁶.

    dy/dt = f_θ(y),   f_θ : ℝ⁶ → ℝ⁶

    Loss = Σ_i || ŷ(t_i) - y(t_i) ||²

    Implementation
    --------------
    * ``torchdiffeq.odeint_adjoint`` for training.
    * 3-layer MLP, 64 hidden units, SiLU activation.

    References
    ----------
    * Chen, R. T. Q. et al. (2018). "Neural Ordinary Differential
      Equations." *NeurIPS*.
    * Rubanova, Y. et al. (2019). "Latent ODEs for Irregularly-Sampled
      Time Series." *NeurIPS*.

    Datasets
    --------
    Kienzle 120 k trajectory dataset (full state: position + velocity).

    Getting Started
    ---------------
    1. Prepare dataset of (t, y) pairs.
    2. Train with ``odeint_adjoint`` and Adam optimiser, lr=1e-3.
    3. Call ``predict()`` — internally integrates the learned ODE.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("NeuralODEPredictor.predict")


class GPPredictor(Predictor):
    """Stage 2c – Gaussian Process predictor with physics mean function.

    Algorithm
    ---------
    Places a GP prior over trajectory residuals with respect to a physics
    mean function (gravity + drag).  The kernel is a scaled RBF
    (squared-exponential) with automatic relevance determination (ARD).

    Posterior predictions yield both a mean trajectory and calibrated
    uncertainty (covariance) over future ball states.

    Mathematics
    -----------
    f(t) ~ GP( m(t), k(t, t') )

    m(t) = physics_prediction(t)   (drag + gravity)

    k(t, t') = σ² exp( -||t - t'||² / (2 ℓ²) )

    Posterior:  p(f* | X, y) = N( μ*, Σ* )

    Implementation
    --------------
    * ``gpytorch`` exact GP with ``ScaleKernel(RBFKernel(ard_num_dims=…))``.
    * Physics mean via ``gpytorch.means.ConstantMean`` replaced with custom
      ``PhysicsMean``.

    References
    ----------
    * Deisenroth, M. P. & Rasmussen, C. E. (2011). "PILCO: A Model-Based
      and Data-Efficient Approach to Policy Search." *ICML*.

    Datasets
    --------
    Kienzle 50 k trajectory dataset.

    Getting Started
    ---------------
    1. Fit hyperparameters on training trajectories via marginal likelihood.
    2. Call ``predict()`` — returns ``BallState`` with covariance populated.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("GPPredictor.predict")


class EnsemblePredictor(Predictor):
    """Stage 3 – Deep ensemble predictor.

    Algorithm
    ---------
    Maintains 3–5 ``ResidualMLPPredictor`` instances, each trained with a
    different random seed and/or data shuffle.  At inference time the
    ensemble mean serves as the point prediction and the ensemble variance
    provides a calibrated uncertainty estimate.

    Mathematics
    -----------
    μ_ens = (1/K) Σ_k μ_k

    σ²_ens = (1/K) Σ_k (σ²_k + μ²_k) - μ²_ens

    Implementation
    --------------
    * Wraps a list of ``ResidualMLPPredictor`` members.
    * Aggregates predictions via NumPy mean / variance.

    References
    ----------
    * Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive
      Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

    Getting Started
    ---------------
    1. Train K=5 ``ResidualMLPPredictor`` instances with different seeds.
    2. Pass them to the constructor.
    3. ``predict()`` returns mean state with covariance from ensemble spread.
    """

    def predict(
        self,
        observations: list[BallObservation],
        t_future: float,
    ) -> BallState:
        raise NotImplementedError("EnsemblePredictor.predict")
