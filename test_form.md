# Matrix form test

$$
\begin{aligned}
cost = F(\mathbf{X})&=f(\mathbf{X}) + g(\mathbf{X})\\
&= \operatorname{tr}(\mathbf{X}^\top\mathbf{A}\mathbf{X}) + 0.1 \|\mathbf{X}\|_{2,1}\\
&= x_{11}^2 + 4x_{12}^2 + x_{21}^2 +4x_{22}^2 + 0.1 \|\mathbf{X}\|_{2,1}
\end{aligned}
$$

where $\mathbf{X} \in \mathbb{R}^{2\times 2}$, $\mathbf{A} \in \mathbb{R}^{2\times 2}$, and
$$
\mathbf{A}=\left[\matrix{1&0\\0&4}\right]
$$

the derivative of cost function over $\mathbf{X}$ should be
$$
\nabla f(\mathbf{X}) = \mathbf{AX}+\mathbf{A}^\top\mathbf{X}
$$

# Vector form test

$$
\begin{aligned}
cost = F({\boldsymbol{x}}) &= f(\boldsymbol{x}) + g(\boldsymbol{x})\\
&=\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x}+0.1\|\boldsymbol{x}\|_1\\
&=x_1^2+2x_2^2+3x_3^2+0.1\|\boldsymbol{x}\|_1
\end{aligned}
$$

where
$$
\mathbf{A}=\left[\matrix{1&0&0\\0&2&0\\0&0&3}\right]
$$
the derivative should be
$$
\begin{aligned}
\nabla f(\boldsymbol{x}) &= (\mathbf{A} + \mathbf{A}^\top)\boldsymbol{x}\\
&= [\matrix{2x_1&4x_2&6x_3}]^\top
\end{aligned}
$$

# Benchmark function: Ackey N. 2 Function

$$
f(x, y) = -200e^{-0.2\sqrt{x^2 + y^2}}
$$

$$
\nabla f = 
40 \frac{e^{-0.2\sqrt{x^2+y^2}}}{\sqrt{x^2+y^2}}
\left[
\matrix{x & y}
\right]^\top
$$