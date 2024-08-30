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
        ee_offset: MatrixXd,
    ) -> list[list[float]]: ...
    def fk(self, x: VectorXd, ee_offset: MatrixXd) -> list[list[float]]: ...
    def ik(
        self,
        config: SolverConfig,
        target: MatrixXd,
        ee_offset: MatrixXd,
        x0: VectorXd,
    ) -> tuple[list[float], float] | None: ...
