
# Stiff ODEs from the document

This markdown file lists four ODE systems known to be stiff or challenging to integrate, based on the 28 given in ODEBench document.

## 1. Van der Pol Oscillator (standard form and simplified form)
- The **Van der Pol oscillator** is a well-known example of a stiff system, especially for large values of the parameter \( c_0 \). It becomes more challenging to integrate as \( c_0 \) increases because the dynamics exhibit rapid oscillations, creating stiff behavior that requires specialized stiff solvers (e.g., backward differentiation formula).
- **Equations**:
  - Standard form: \( x_1 = -c_0 x_1 (x_0^2 - 1) - x_0 \)
  - Simplified form: \( c_0 \left( -rac{x_0^3}{3} + x_0 + x_1 
ight) - x_0 \)

## 2. Brusselator
- The **Brusselator** is a chemical oscillation model, and in certain parameter regimes, it can exhibit stiff dynamics due to the presence of both slow and fast reactions.
- **Equation**: \( c_1 x_0^2 x_1 - x_0 (c_0 + 1) + 1, c_0 x_0 - c_1 x_0^2 x_1 \)

## 3. Glycolytic Oscillator
- This oscillator describes biochemical oscillations and can become stiff depending on the system parameters, especially when modeling rapid interactions in metabolic pathways.
- **Equation**: \( c_0 x_1 + x_0^2 x_1 - x_0, -c_0 x_0 + c_1 - x_0^2 x_1 \)

## 4. Lotka-Volterra Models (variants)
- Depending on the parameter values, these predator-prey models can also exhibit stiffness, especially when modeling sharp predator-prey dynamics or competition models. Stiffness arises when one population grows much faster than the other.
- **Equations**:
  - Lotka-Volterra competition model: \( x_0 (c_0 - c_1 x_1 - x_0), x_1 (c_2 - x_0 - x_1) \)
  - Lotka-Volterra simple model: \( x_0 (c_0 - c_1 x_1), -x_1 (c_2 - c_3 x_0) \)
