import Random
import PyPlot
import Printf


import Statistics
import SpecialFunctions

import Calculus
import ForwardDiff

import Optim

using LinearAlgebra

import BSON


include("../src/declarations.jl")



include("../src/retractions/Rp.jl")

include("../src/manifold/vector_transport.jl")

include("../src/matrices/numerical.jl")

include("../src/VGP/objective.jl")

include("../src/optimization/CG.jl")
include("../src/optimization/Rp/engine_Rp.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1



## load data.
data = BSON.load("../data/density_fit_data1.bson")

y = data[:y]
X = data[:X]
K = data[:K]
α_SDP = data[:α_SDP]
μ = data[:μ]



f = aa->RKHSfitdensitycostfunc(aa, K, y, μ)
df_Euc_ND = xx->Calculus.gradient(f,xx)
df_Euc = aa->gradientRKHSfitdensitycostfunc(y, K, μ, aa)
df = df_Euc

d2f_ND = xx->Calculus.hessian(f,xx)

### verify df.
x0 = randn(length(y))
discrepancy = norm(df_Euc_ND(x0)- df_Euc(x0))
println("discrepancy between AN and ND for df is ", discrepancy)

### verify d2f is constant.
q = d2f_ND(x0)
H = (K*K + μ .* K) .* 2.0

discrepancy = norm(d2f_ND(x0)- H)
println("discrepancy between AN and ND for d2f is ", discrepancy)

q = d2f_ND(randn(length(y)))
H = gethessianRKHSfitdensitycostfunc(y, K, μ)

discrepancy = norm(d2f_ND(x0)- H)
println("discrepancy between AN and ND for d2f is ", discrepancy)


### verify trust-region subproblem.

N = length(y)

x0 = rand(N) .* 2.3
df_p = df(x0)

γ_p = ones(Float64, N) .* rand() .* 5.0
G = diagm(γ_p)
Bp = H'*G

𝓛 = xx->(dot(df_p, xx) + 0.5* dot(xx, Bp, xx))

B = Bp + Bp'
d𝓛 = xx->(df_p + 0.5 .* B*xx)
x_star = -2.0 .* (B\df_p)

import Optim
max_iters_optim = 10000
N_particles_acq = 3
op = Optim.Options( iterations = max_iters_optim,
                         store_trace = false,
                         show_trace = false)

#lower = zeros(Float64, N)
lower = ones(Float64, N) .* -Inf
upper = ones(Float64, N) .* Inf
swarm = Optim.ParticleSwarm()
#
x_optim_start = x_star #rand(N)
results = Optim.optimize(𝓛, x_optim_start, swarm, op)

x_optim = results.minimizer

println("𝓛(x_star)  = ", 𝓛(x_star))
println("𝓛(x_optim) = ", 𝓛(x_optim))
println()

println("𝓛(x_optim)-𝓛(x_star) = ", 𝓛(x_optim)-𝓛(x_star))
println()

# next, come up with Riemannian metric that makes sense for this optim problem.
