
"""
Consists of the following fields:

max_iter::Int
verbose_flag::Bool
norm_df_tol::T
objective_tol::T
avg_Δf_tol::T
avg_Δf_window::Int
max_idle_update_count::Int

Note: the optimizer must run for at least avg_Δf_window iterations before avg_Δf_tol is checked as a stopping condition.




Example:

norm_df_tol = 1e-5
objective_tol = 1e-5
avg_Δf_tol = 0.0 # 1e-12
avg_Δf_window = 10
max_idle_update_count = 50
opt_config = OptimizationConfigType( max_iter,
                                        verbose_flag,
                                        norm_df_tol,
                                        objective_tol,
                                        avg_Δf_tol,
                                        avg_Δf_window,
                                        max_idle_update_count)
"""
mutable struct OptimizationConfigType{T}
    max_iter::Int
    verbose_flag::Bool
    norm_df_tol::T
    objective_tol::T

    # the optimizer must run for at least avg_Δf_window iterations before
    #   avg_Δf_tol is checked as a stopping condition.
    avg_Δf_tol::T
    avg_Δf_window::Int
    max_idle_update_count::Int

    # # for metric switch.
    # mean_x_tol_for_metric_switch::T
    # metric_a::T
    # metric_c::T

    # Hessian posdef approximation parameter.
    𝑟::T
end


"""
Consists of the following fields:

minimum_TR_radius::T
maximum_TR_radius::T
max_iter_tCG::Int
verbose_flag::Bool
ρ_lower_acceptance::T # below 1.
ρ_upper_acceptance::T


Example:

max_iter = 400
verbose_flag = false #true
max_iter_tCG = 100
ρ_lower_acceptance = 0.2 # recommended to be less than 0.25
ρ_upper_acceptance = 5.0
TR_config = TrustRegionConfigType(  1e-3,
                                    10.0,
                                    max_iter_tCG,
                                    verbose_flag,
                                    ρ_lower_acceptance,
                                    ρ_upper_acceptance)
"""
mutable struct TrustRegionConfigType{T}
    minimum_TR_radius::T
    maximum_TR_radius::T
    max_iter_tCG::Int
    verbose_flag::Bool
    ρ_lower_acceptance::T # below 1.
    ρ_upper_acceptance::T # higher than 1.
end

mutable struct CholeskyType{T}
    c::Vector{T} # diagonal entries.
    a::Vector{T} # offdiagonal entries.
end

mutable struct MVNType{T}
    μ::Vector{T} # The mean.
    L::CholeskyType{T}
    #L::Matrix{T} # Cholesky decomposition of the covmat.
end

mutable struct Rank1PositiveDiagmType{T}
    c::Vector{T} # diagonal matrix of strictly positive values. Length n.
    A::Matrix{T} # n x p matrix of p rank-1 updates.
end

# vector transport.
mutable struct VectorTransportType{T,D,M}
    #ℜ::Function # The retraction at p.
    p::M
    X::Array{T,D} # The fixed vector at p.
    Y::Array{T,D} # The input vector.
    𝑉::Function # the vector transport.
end


function MVNType(m::Vector{T}, S::Matrix{T})::MVNType{T} where T
    L = cholesky(S).L
    return MVNType(m, packCholeskyType(L))
end
