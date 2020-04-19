## test opt



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

include("../src/optimization/CG.jl")
include("../src/optimization/Rp/engine_Rp.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1


## load set up parameters and oracle solution.
data = BSON.load("../data/density_fit_data1.bson")

y = data[:y]
X = data[:X]
K = data[:K]
μ = data[:μ]

α_SDP = data[:α_SDP] # this is the oracle solution.

f = aa->RKHSfitdensitycostfunc(aa, K, y, μ)
df_Euc = aa->gradientRKHSfitdensitycostfunc(y, K, μ, aa)
H = gethessianRKHSfitdensitycostfunc(y, K, μ)
#fill!(H, 0.0) # force H to not be posdef. This triggers a Hessain approximnation in the engine.

selfmetricfunc = (XX,pp)->dot(XX,XX)
metricfunc = (XX,YY,pp)->dot(XX,YY)

## initial guess.
α_initial = ones(Float64, length(y))

## optimization configuration.
max_iter = 400
verbose_flag = true #  false
max_iter_tCG = 100
ρ_lower_acceptance = 0.2 # recommended to be less than 0.25
ρ_upper_acceptance = 5.0
TR_config = TrustRegionConfigType(  1e-3,
                                    10.0,
                                    max_iter_tCG,
                                    verbose_flag,
                                    ρ_lower_acceptance,
                                    ρ_upper_acceptance)
#
norm_df_tol = 1e-5
objective_tol = 1e-5
avg_Δf_tol = 0.0 #1e-12 #1e-5
avg_Δf_window = 10
max_idle_update_count = 50
opt_config = OptimizationConfigType( max_iter,
                                        verbose_flag,
                                        norm_df_tol,
                                        objective_tol,
                                        avg_Δf_tol,
                                        avg_Δf_window,
                                        max_idle_update_count)


α_star, f_α_array, norm_df_array, num_iters = engineRp(f,
                                        df_Euc,
                                        α_initial,
                                        randn(length(α_initial)),
                                        metricfunc,
                                        selfmetricfunc,
                                        TR_config,
                                        opt_config,
                                        H)
#
discrepancy = norm(α_SDP-α_star)
println("discrepancy between another solver's solution and the RiemannianOptim solution: ", discrepancy)

println("f(α_SDP) = ", f(α_SDP))
println("f(α_star) = ", f(α_star))
println()

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(collect(1:num_iters), log.(f_α_array))

title_string = "log f(α) history vs. iterations"
PyPlot.title(title_string)
