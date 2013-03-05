% Declarative Infrastructure for Automated Scientific Computing
% Matthew Rocklin
% March 5th, 2013

Expertise
---------

Expertise is important for efficient computation 


Expertise
---------

Most scientific programmers aren't expert in all requisite disciplines 


Expertise
---------

~~~~~~~~~C
// A: Naive             // B: Programmer    // C: Mathematician
int fact(int n){        int fact(int n){    int fact(int n){
    if (n == 0)             prod = n;           // n! = Gamma(n+1)   
        return 1;           while(n--)          return lround(exp(lgamma(n+1)));
    return n*fact(n-1);         prod *= n;  }
}                           return prod;
                        }
~~~~~~~~~

*   Modern compilers automate `B`
*   Humans do `C` by hand
*   The people who know `C` are rarely trained to build compilers

*   Both `B` and `C` are commonly necessary within one project

Composable Software - Monolithic
--------------------------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/nwp_monolithic}
\end{figure}

Composable Software - Composable
--------------------------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/nwp_orthogonal}
\end{figure}

Outline
-------

*   Probability Modeling 
    *   Bayesian inference simulations (MCMC)
*   Matrix algebra          
    *   BLAS/LAPACK computations
*   Static Scheduling       
    *   Blocked Matrix Algorithms


Probability Modeling
====================

Computer Algebra - SymPy
------------------------

~~~~~~~~Python
>>> expr = log(3*exp(x + 2))
>>> print simplify(expr)
x + 2 + log(3)
~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.3\textwidth]{images/expr.pdf}
\includegraphics[width=.3\textwidth]{images/sexpr.pdf}
\end{figure}

Random Variables
----------------

~~~~~~Python
>>> x = Normal('x', 0, 1)

>>> expr = log(3*exp(x + 2))
>>> print simplify(expr)
x + log(3) + 2
~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.5\textwidth]{images/pexpr.pdf}
\end{figure}


Random Variables
----------------

~~~~~~Python
>>> x = Normal('x', 0, 1)

>>> expr = log(3*exp(x + 2))
>>> print simplify(expr)
x + log(3) + 2

>>> P(expr > 4)
~~~~~~

$$ \int_{0}^{\infty} \frac{\sqrt{2} e^{- \frac{1}{2} \left(z - \log{\left (3
\right )} + 2\right)^{2}}}{2 \sqrt{\pi}}\, dz $$

1.  Uncertainty doesn't interfere
2.  Probability query $\rightarrow$ integral expression 
    is a simple transformation
3.  Integral problems have mature solutions


Random Variables
----------------

![](images/stats_ecosystem.pdf)


Bayesian Inference
------------------

~~~~~~~~Python
>>> rate = Beta('lambda', a, b)
>>> count = Poisson('count', rate)
~~~~~~~~

$$ p(x \vert \lambda) = \frac{\lambda^{x}}{e^{\lambda} x!} \;\;\;\;
p(\lambda) = \frac{\lambda^{a - 1} \left(- \lambda + 1\right)^{b - 1} \Gamma\left(a + b\right)}{\Gamma\left(a\right) \Gamma\left(b\right)}$$ 

Infer `rate` given many observations for `count`

$$ \textrm{Maximize }\;\; p(\lambda \vert x_i) \propto \prod_i p(x_i \vert \lambda) \cdot p(\lambda) $$
$$ 0 = \frac{d}{d\lambda} \log\left( \prod_i p(x_i \vert \lambda) \cdot p(\lambda)\right) =
\frac{d}{d\lambda} \sum_i \log(p(x_i \vert \lambda) \cdot p(\lambda)) $$

Need to find the roots of 
$$ \sum_{i=1}^{n} \frac{a \left(\lambda - 1\right) + b \lambda - \lambda \left(\lambda - 1\right) - 2 \lambda + \left(\lambda - 1\right) \operatorname{data}{\left [i \right ]} + 1}{\lambda \left(\lambda - 1\right)} = 0 $$ 

Bayesian Inference
----------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.4\textwidth]{images/pymc_ecosystem}
\end{figure}


How do we solve math problems?
-------------------

~~~~~~~~~Python
>>> A = Normal('a', 0, 1)
>>> density(A)
~~~~~~~~~

$$ \frac{\sqrt{2} e^{- \frac{1}{2} z^{2}}}{2 \sqrt{\pi}} $$

~~~~~~~~~Python
>>> density(A**2)
~~~~~~~~~

Use generic transformations taught in Statistics 101, e.g.
$$ f_Y(y) = f_X(g^{-1}(y)) \left\vert\frac{d g^{-1}(y)}{d y}\right\vert $$


$$ \frac{\sqrt{2} e^{- \frac{1}{2} z} \lvert{\frac{1}{\sqrt{z}}}\rvert}{2\sqrt{\pi}} $$ 


How do we solve math problems?
-------------------

~~~~~~~~~Python
>>> A = Normal('a', 0, 1)
>>> B = Normal('b', 0, 1)
>>> density(A**2 + B**2)
~~~~~~~~~

$$ \int_{-\infty}^{\infty} \frac{e^{- \frac{1}{2} \left(b - a\right)^{2}} e^{- \frac{1}{2} z + \frac{1}{2} b^{2}}}{2 \pi \lvert{\sqrt{z - b^{2}}}\rvert}\, db $$


How do we solve math problems?
----------------------------

~~~~~~~~~Python
>>> A = Normal('a', 0, 1)
>>> B = Normal('b', 0, 1)
>>> density(A**2 + B**2)
~~~~~~~~~

$$ \frac{1}{2} e^{- \frac{1}{2} z} $$ 

Phenomenological relations:

$$ N(0, 1)^2 \sim \chi^2(1) $$
$$ \chi^2(n) + \chi^2(m) \sim \chi^2(n + m) $$


Term Rewrite System
-------------------

Rewrite rule:

---------------  -------------------
Source pattern   $tan(x)$
Target pattern   $sin(x) / cos(x)$
Varibles         $x,$
---------------  -------------------

Example: 

    From:   3 + tan(a + b)**2     
    
    To:     3 + (sin(a + b) / cos(a + b))**2


Term Rewrite System
-------------------

Rules:

$$ tan(a) \rightarrow sin(a) / cos(a) $$
$$ sin^2(a) \rightarrow \frac{1-cos(2a)}{2}$$
$$ cos^2(a) \rightarrow \frac{1+cos(2a)}{2}$$
$$ sin(a) + sin(b) \rightarrow 2 sin(\frac{a+b}{2}) cos(\frac{a+b}{2})$$
$$ sin^2(a) + cos^2(a) \rightarrow 1 $$
$$ sin(a) / cos(a) \rightarrow tan(a) $$
$$ ... $$

Simplify:
    
$$ sin^2(y) + \frac{sin(z)}{cos(z)} + cos^2(y)  $$


Encode Statistical Rewrite Rules
--------------------------------

Express patterns:

~~~~~~~~~~python
patterns = [
    (Normal(0, 1),                      StandardNormal(),    []),
    (StandardNormal()**2,               ChiSquared(1),       []),
    (ChiSquared(m) + ChiSquared(n),     ChiSquared(n + m),   [n, m]),
    ...
]
~~~~~~~~~~

Define control flow:

~~~~~~~~~~python
canonicalize = chain(repeat, bottom_up, choose)
~~~~~~~~~~

Combine:

~~~~~~~~~~python
stat_simplify = canonicalize(patterns)
~~~~~~~~~~


Status and Evaluation
---------------------

*   Software:   
    *   Fully functional
    *   Lacks efficient matching for many patterns
    *   Maybe integrate into PyMC
*   Evaluation: 
    *   Compare numeric runtimes
    *   Compare complexity of problem description


Related Work
------------

Symbolic Statistics - APPL

Markov chain Monte Carlo - PyMC, WinBUGS, JAGS

Term Rewrite Systems - Elan, Maude, Mathematica, Stratego/XT, Coq


Numerical Linear Algebra
========================

The need for a high level array compiler
----------------------------------------

    x = ones(10000, 1)

    (x*x')*x            Elapsed time is 0.337711 seconds.
    x*(x'*x)            Elapsed time is 0.000956 seconds.

\begin{figure}[htbp]
\centering
\includegraphics[width=.6\textwidth]{images/xxtrans}
\end{figure}


The need for a high level array compiler
----------------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

    beta = (X.T*X).I*X.T*y

The need for a high level array compiler
----------------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

    beta = solve(X.T*X, X.T*y)

The need for a high level array compiler
----------------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

    beta = spd_solve(X.T*X, X.T*y)

BLAS/LAPACK
-----------

Numeric libraries for dense linear algebra

>*  `DGEMM` - **D**ouble precision **GE**neral **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

>*  `DSYMM` - **D**ouble precision **SY**mmetric **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

>*  ...

>*  `DPOSV` - **D**ouble symmetric **PO**sitive definite matrix **S**ol**V**e  -- $A^{-1}y$
    *   `SUBROUTINE DPOSV( UPLO, N, NRHS, A, LDA, B, LDB, INFO )`

Necessary Definitions
---------------------

**Language**: Multiply, addition, inverse, transpose, trace, determinant, blocks, etc...

    beta = (X.T*X).I*X.T*y              X.I*X -> Identity

**Predicates**: symmetric, positive_definite, full_rank, orthogonal, lower_triangular, etc....

    fullrank(X)                         fullrank(X) -> positive_definite(X.T*X)

**Computations**:

    class SYMM(BLAS):
        inputs    = [alpha, A, B, beta, C]
        outputs   = [alpha*A*B + beta*C]
        condition = symmetric(A) or symmetric(B)
        inplace   = {0: 4} # 0th output stored in 4th input
        template  = ....


Mathematical code
-----------------

Original language definition in Maude 

    eq A (B + C) = (A B) + (A C) .
    eq (alpha A)' = alpha A' .
    eq A'' = A .
    eq inverse(A) A = Identity .
    ...
    eq  X X' is symmetric = True .
    ceq X Y  is positive-definite = True  if X is positive-definite
                                         and Y is positive-definite .
    ceq X X' is positive-definite = True  if X is full-rank .

Eventually moved to SymPy for distribution reasons

Computation
-----------

**Given**:

    (X.T*X).I*X.T*y
    full_rank(X)

**Produce**:

![](images/hat-comp.pdf)


Computation
-----------

\begin{figure}[htbp]
\centering
\includegraphics<1->[width=.24\textwidth]{images/hat0}
\includegraphics<2->[width=.24\textwidth]{images/hat1}
\includegraphics<3->[width=.24\textwidth]{images/hat2}
\includegraphics<4->[width=.24\textwidth]{images/hat3}
\end{figure}

User Experience
---------------

~~~~~~~~Python
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)

inputs  = [X, y]
outputs = [(X.T*X).I*X.T*y]
facts   = Q.fullrank(X)

f = f2py(next(compile(inputs, outputs, facts)))
~~~~~~~~~

\hrule

~~~~~~~~Fortran
subroutine f(X, y, var_7, m, n)
implicit none

integer, intent(in) :: m
integer, intent(in) :: n
real*8, intent(in) :: y(n)          !  y
real*8, intent(in) :: X(n, m)       !  X
real*8, intent(out) :: var_7(m)     !  0, X'*y, (X'*X)^-1*X'*y
real*8 :: var_8(m, m)               !  0, X'*X
integer :: INFO                     !  INFO

call dgemm('N', 'N', m, 1, n, 1, X, n, y, n, 0, var_7, m)
call dgemm('N', 'N', m, m, n, 1, X, n, X, n, 0, var_8, m)
call dposv('U', m, 1, var_8, m, var_7, m, INFO)

RETURN
END
~~~~~~~~~

Multiple Results
----------------

\begin{figure}[htbp]
\centering
\includegraphics<1->[width=.24\textwidth]{images/hat0}
\includegraphics<1->[width=.24\textwidth]{images/hat1}
\includegraphics<1->[width=.24\textwidth]{images/hat2}
\includegraphics<1->[width=.24\textwidth]{images/hat3}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics<2>[width=.23\textwidth]{images/hat_gesv1}
\includegraphics<2>[width=.23\textwidth]{images/hat_gesv2}
\includegraphics<2>[width=.53\textwidth]{images/hat_gesv3}
\end{figure}

Status and Evaluation
---------------------



Static Scheduling
=================

Static Scheduling
-----------------

    newmu    = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
    newSigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

![](images/kalman.pdf)

Static Scheduling
-----------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.48\textwidth]{images/kalman_cpu1}
\includegraphics[width=.48\textwidth]{images/kalman_cpu0}
\end{figure}


Static Scheduling
-----------------

**Given**:
\begin{columns}
\column{.5\textwidth}

Computation Graph

\column{.5\textwidth}

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/hat-comp}
\end{figure}

\end{columns}

\begin{columns}
\column{.5\textwidth}
Worker network 

\column{.5\textwidth}

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/worker}
\end{figure}
\end{columns}


\begin{columns}
\column{.5\textwidth}
Computation times

Communication times 

\column{.5\textwidth}
task, worker $\rightarrow$ time

variable, source, target $\rightarrow$ time
\end{columns}

**Produce**:

Set of computation subgraphs to minimize total runtime


Application - Blocked Cholesky Decomposition
--------------------------------------------

Math Problem:

$$ Ax = y \;\; \rightarrow  \;\; LL^Tx = y \;\; \rightarrow \;\; x = L^{-T}L^{-1}y $$
$$A \textrm{ symmetric positive definite, } L \textrm{ lower triangular}$$

$$\left[\begin{smallmatrix}A_{11} & A_{21}^T \\
                           A_{21} & A_{22} \end{smallmatrix}\right]
= 
\left[\begin{smallmatrix}L_{11} & 0 \\
                         L_{21} & L_{22} \end{smallmatrix}\right] 
\left[\begin{smallmatrix}L_{11} & 0 \\
                         L_{21} & L_{22} \end{smallmatrix}\right]^T$$

$$ L_{11} := \operatorname{cholesky}(A_{11}) $$
$$ L_{21} := A_{21}L_{11}^{-T} $$
$$ L_{22} := \operatorname{cholesky}(A_{22} - L_{21}L_{21}^{T}) $$


Compute Resources:

E.g. Four node system with two GPUs on Infiniband


Related work
------------

*   Heterogeneous Static Scheduling
    *   Integer Programming - Tompkins 2003
    *   Heterogeneous Earliest Finish Time - Topcuoglu 2002 
    *   Suggestions?
*   Automated Dense Linear Algebra
    *   ScaLAPACK, PlaLAPACK, BLACS
    *   FLAME - Language for blocked matrix algorithms
        -   SuperMatrix - Dynamic shared memory variant
        -   Elemental - Distributed memory variant
    *   Magma - Hybrid LAPACK
    *   Spiral - Hardware specific numeric code generation with 
        internal computation language


Status and Evaluation
---------------------

*   Software 
    *   Implemented Schedulers
    *   Implemented rudimentary model
    *   Everything hooks up well
*   Comparison
    *   Scheduling time
    *   Execution
    *   Comparison against Magma?


Recap
=====

What will I do?
---------------

*   Solve a specific numerical problem 
    *   Distributed Blocked Linear Algebra
*   Without writing anything by hand
*   Keeping mathematical and algorithmic code separate


Extras
======


Inplace
-------

![](images/kalman.pdf)

Inplace
-------

![](images/kalman-inplace.pdf)
