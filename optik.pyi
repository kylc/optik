from typing import Literal

class SolverConfig:
    def __init__(
        self,
        solution_mode: Literal["speed", "quality"] = ...,
        max_time: float = ...,
        max_restarts: int = ...,
        tol_f: float = ...,
        tol_df: float = ...,
        tol_dx: float = ...,
        linear_weight: list[float] = ...,
        angular_weight: list[float] = ...,
    ): ...

class Robot:
    @staticmethod
    def from_urdf_file(path: str, base_link: str, ee_link: str) -> Robot: ...
    def set_parallelism(self, n: int) -> None: ...
    def num_positions(self) -> int: ...
    def joint_limits(self) -> tuple[list[float], list[float]]: ...
    def joint_jacobian(
        self,
        x: list[float],
        ee_offset: list[list[float]],
    ) -> list[list[float]]: ...
    def fk(self, x: list[float], ee_offset: list[list[float]]) -> list[list[float]]: ...
    def ik(
        self,
        config: SolverConfig,
        target: list[list[float]],
        ee_offset: list[list[float]],
        x0: list[float],
    ) -> tuple[list[float], float] | None: ...
