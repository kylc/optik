from typing import Literal

def set_parallelism(n: int) -> None: ...

class SolverConfig:
    def __init__(
        self,
        gradient_mode: Literal["numerical", "analytical"] = ...,
        solution_mode: Literal["speed", "quality"] = ...,
        max_time: float = ...,
        max_restarts: int = ...,
        tol_f: float = ...,
        tol_dx: float = ...,
        tol_df: float = ...,
    ): ...

class Robot:
    @staticmethod
    def from_urdf_file(path: str, base_link: str, ee_link: str) -> Robot: ...
    def num_positions(self) -> int: ...
    def joint_limits(self) -> tuple[list[float], list[float]]: ...
    def fk(self, x: list[float]) -> list[list[float]]: ...
    def ik(
        self, config: SolverConfig, target: list[float[float]], x0: list[float]
    ) -> tuple[list[float], float] | None: ...
