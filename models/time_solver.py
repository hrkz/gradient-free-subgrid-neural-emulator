import numpy as np
import jax
import jax.numpy as jnp

from typing import Callable, List, Tuple

class ImexScheme:
    def coef(self) -> float:
        pass
    def __call__(
        self,
        eq,
        system: jnp.ndarray,
        source: Callable,
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ) -> Callable:
        pass

class OdeScheme:
    def __call__(
        self,
        eq,
        source: Callable,
        explicit: Callable,
    ) -> Callable:
        pass

class BPR353(ImexScheme):
    def __init__(self, dt: float):
        self.dt = dt
    def coef(self) -> float:
        return 0.5 * self.dt
    def __call__(
        self,
        eq,
        source: Callable,
        eq_system: List[jnp.ndarray],
        implicit: Callable,
        explicit: Callable,
        solve: Callable,
    ):
        def __imex_step__(
            state_s: jnp.ndarray,
            t: float
        ) -> Tuple[float, jnp.ndarray]:
            expl_0, c = explicit(state_s, source, t)
            impl_0    = implicit(state_s)
            step_0    = state_s + self.dt * expl_0 + self.dt * 0.5 * impl_0
            state_0   = solve(eq_system, step_0)
            
            expl_1, _ = explicit(state_0, source, t)
            impl_1    = implicit(state_0)
            step_1    = state_s + self.dt * (4.0/9.0 * expl_0 + 2.0/9.0 * expl_1) + self.dt * (5.0/18.0 * impl_0 - 1.0/9.0 * impl_1)
            state_1   = solve(eq_system, step_1)
            
            expl_2, _ = explicit(state_1, source, t)
            impl_2    = implicit(state_1)
            step_2    = state_s + self.dt * (0.25 * expl_0 + 0.75 * expl_2) + self.dt * 0.5 * impl_0
            state_2   = solve(eq_system, step_2)
            
            impl_3    = implicit(state_2)
            step_3    = state_s + self.dt * (0.25 * expl_0 + 0.75 * expl_2) + self.dt * (0.25 * impl_0 + 0.75 * impl_2 - 0.5 * impl_3)
            state_s   = solve(eq_system, step_3)
            
            return (
              c, state_s
            )
        return __imex_step__

class RK4(OdeScheme):
    def __init__(self, dt: float):
        self.dt = dt
    def __call__(
        self,
        source: Callable,
        explicit: Callable
    ):
        def __ode_step__(
            x_k: jnp.ndarray,
            y_j: jnp.ndarray,
            t: float
        ) -> Tuple[float, jnp.ndarray]:
            x_dot0, y_dot0 = explicit(x_k, y_j, source(x_k, t))

            x_s1 = x_k + self.dt * 0.5 * x_dot0
            x_dot1, y_dot1 = explicit(
                x_s1, 
                y_j + self.dt * 0.5 * y_dot0, 
                source(x_s1, t + self.dt * 0.5)
            )

            x_s2 = x_k + self.dt * 0.5 * x_dot1
            x_dot2, y_dot2 = explicit(
                x_s2, 
                y_j + self.dt * 0.5 * y_dot1, 
                source(x_s2, t + self.dt * 0.5)
            )

            x_s3 = x_k + self.dt * x_dot2
            x_dot3, y_dot3 = explicit(
                x_s3, 
                y_j + self.dt * y_dot2, 
                source(x_s3, t + self.dt)
            )

            x_k = x_k + self.dt * (x_dot0 / 6. + x_dot1 / 3. + x_dot2 / 3. + x_dot3 / 6.)
            y_j = y_j + self.dt * (y_dot0 / 6. + y_dot1 / 3. + y_dot2 / 3. + y_dot3 / 6.)
            return (
              x_k, y_j
            )
        return __ode_step__
