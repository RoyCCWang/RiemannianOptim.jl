### compositions of elemnetary retractions.
# separate versions required for AD, since tuples are not supported in
#   existing Julia AD packages.

# this is for 𝓜 := ℝ_+^N x Stiefel(N,p). No quotient manifold. No for AD.
function rank1positivediagmretraction(p::Rank1PositiveDiagmType,
                                      X::Rank1PositiveDiagmType,
                                      t::T)::Tuple{Vector{T},Matrix{T}} where T

    ### check.
    @assert size(p.c) == size(X.c)
    @assert size(p.A) == size(X.A)


    p_c_next = ℝ₊₊arrayexpquadraticretraction(p.c, X.c, t)
    p_A_next = fixedrankretractiontall(p.A, X.A, t)

    return Rank1PositiveDiagmType(p_c_next, p_A_next)
end

# for vector transport. Only for automatic differentiation.
function rank1positivediagmretraction(p::Rank1PositiveDiagmType,
                                      X::Matrix{T},
                                      Y::Matrix{T},
                                      t::T2)::Matrix{T2} where {  T <: Real,
                                                                  T2 <: Real}
    ### check.
    @assert ( length(p.c), size(p.A)[2] + 1 ) == size(X)

    p_c_next = ℝ₊₊arrayexpquadraticretraction(p.c, X[:,1], Y[:,1], t)
    p_A_next = fixedrankretractiontall(p.A, X[:,2:end], Y[:,2:end], t)

    return [p_c_next p_A_next]
end






function choleskyretraction(p::CholeskyType{T},
                            X::Vector{T},
                            t::T)::CholeskyType{T} where T

    ### check.
    n = length(p.c)

    @assert length(X) == round(Int, n*(n+1)/2)

    p_c_next = ℝ₊₊arrayexpquadraticretraction(p.c, X[1:n], t)
    p_a_next = p.a + t .* X[n+1:end]

    return CholeskyType(p_c_next, p_a_next)
end

# for vector transport, automatic differentiation only.
function choleskyretraction(p::CholeskyType{T},
                            X::Vector{T},
                            Y::Vector{T},
                            t::T2)::Vector{T2} where {T,T2}

    ### check.
    n = length(p.c)

    @assert length(X) == round(Int, n*(n+1)/2)

    p_c_next = ℝ₊₊arrayexpquadraticretraction(p.c, X[1:n], Y[1:n], t)
    p_a_next = p.a + X[n+1:end] + t .* Y[n+1:end]

    return [p_c_next; p_a_next]
end


function MVNretraction( p::MVNType{T},
                        X::Vector{T},
                        t::T2)::MVNType{T2} where {T,T2}
#
    #
    p_m_next = p.μ + t .* X[1:n]

    p_L_next = choleskyretraction(p.L, X[n+1:end], t)

    return MVNType(p_m_next,p_L_next)
end

# vector transport. AD.
function MVNretraction( p::MVNType{T},
                        X::Vector{T},
                        Y::Vector{T},
                        t::T2)::Vector{T2} where {T,T2}
#
    #
    p_m_next = p.μ + X[1:n] + t .* Y[1:n]

    p_L_next = choleskyretraction(p.L, X[n+1:end], Y[n+1:end], t)

    return [p_m_next; p_L_next]
end
