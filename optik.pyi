from typing import Literal

import numpy as np
import numpy.typing as npt

VectorXd = list[float] | npt.NDArray[np.float64]
MatrixXd = list[list[float]] | npt.NDArray[np.float64]

class SolverConfig:
    def __init__(
        self,
        solution_mode: Literal["speed", "quality"] = ...,
        max_time: float = ...,
        max_restarts: int = ...,
        tol_f: float = ...,
        tol_df: float = ...,
        tol_dx: float = ...,
        linear_weight: VectorXd = ...,
        angular_weight: VectorXd = ...,
    ): ...

class Robot:
    @staticmethod
    def from_urdf_file(path: str, base_link: str, ee_link: str) -> Robot: ...
    def set_parallelism(self, n: int) -> None: ...
    def num_positions(self) -> int: ...
    def joint_limits(self) -> tuple[list[float], list[float]]: ...
    def joint_jacobian(
        self,
        x: VectorXd,
        ee_offset: MatrixXd | None = ...,
    ) -> list[list[float]]: ...
    def fk(
        self, x: VectorXd, ee_offset: MatrixXd | None = ...
    ) -> list[list[float]]: ...
    def ik(
        self,
        config: SolverConfig,
        target: MatrixXd,
        x0: VectorXd,
        ee_offset: MatrixXd | None = ...,
    ) -> tuple[list[float], float] | None: ...
    def diff_ik(
        self,
        x0: VectorXd,
        V_WE: VectorXd,
        v_max: VectorXd,
        ee_offset: MatrixXd | None = ...,
    ) -> tuple[float, list[float]] | None: ...
