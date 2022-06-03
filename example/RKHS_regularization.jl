## test opt



import Random
import PyPlot
import Printf
using LinearAlgebra
import BSON

# include("../src/declarations.jl")
#
# include("../src/retractions/Rp.jl")
#
# include("../src/manifold/vector_transport.jl")
#
# include("../src/optimization/CG.jl")
#
# #include("../src/optimization/Rp/engine_Rp.jl")
# include("../src/optimization/vectorspace/engine_array.jl")
#
# include("../src/optimization/TRS/trustregion.jl")
# include("../src/optimization/TRS/trhelpers.jl")
#
# include("../src/problems/RKHS_positive_coefficients.jl")

include("../src/RiemannianOptim.jl")
import .RiemannianOptim

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

f = aa->RiemannianOptim.RKHSfitdensitycostfunc(aa, K, y, μ)
df_Euc = aa->RiemannianOptim.gradientRKHSfitdensitycostfunc(y, K, μ, aa)
H = RiemannianOptim.gethessianRKHSfitdensitycostfunc(y, K, μ)
#fill!(H, 0.0) # force H to not be posdef. This triggers a Hessain approximnation in the engine.

# ## Euclidean metric.
# g = pp->1.0

# ## prop. metric.
# g = pp->dot(pp,pp)

## inv. prop. metric.
g = pp->1.0/(dot(pp,pp)+1.0)

# ## fancy inv. prop. metric.
# g = pp->1.0/(dot(pp,pp)+norm(pp)+1.0)

## initial guess.
α_initial = ones(Float64, length(y))

## optimization configuration.
max_iter = 1600 # 400
verbose_flag = false #  false
max_iter_tCG = 100
ρ_lower_acceptance = 0.2 # recommended to be less than 0.25
ρ_upper_acceptance = 5.0
TR_config = RiemannianOptim.TrustRegionConfigType(  1e-3,
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
𝑟 = 1e-2
opt_config = RiemannianOptim.OptimizationConfigType( max_iter,
                                        verbose_flag,
                                        norm_df_tol,
                                        objective_tol,
                                        avg_Δf_tol,
                                        avg_Δf_window,
                                        max_idle_update_count,
                                        𝑟 )

# TODO get this retraction lower bound sorted out.
#retraction_lower_bound = 1e-10
#ℜ =  xx->ℝ₊₊arrayexpquadraticretraction(xx...; lower_bound = retraction_lower_bound)
ℜ = RiemannianOptim.ℝ₊₊arrayexpquadraticretraction
@time α_star, f_α_array, norm_df_array,
    num_iters = RiemannianOptim.engineArray(    f,
                                df_Euc,
                                α_initial,
                                copy(α_initial),
                                TR_config,
                                opt_config,
                                H,
                                ℜ;
                                𝑔 = g)
#
discrepancy = norm(α_SDP-α_star)
println("discrepancy between another solver's solution and the RiemannianOptim solution: ", discrepancy)

println("f(α_SDP)  = ", f(α_SDP))
println("f(α_star) = ", f(α_star))
println()

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(collect(1:num_iters), log.(f_α_array))

title_string = "log f(α) history vs. iterations"
PyPlot.title(title_string)
