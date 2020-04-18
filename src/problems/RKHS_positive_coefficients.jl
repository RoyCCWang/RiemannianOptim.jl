
function RKHSfitdensitycostfunc(α::Vector{T}, Kp::Matrix{T}, y::Vector{T}, μ::T)::T where T
    N = length(y)
    @assert length(α) == size(Kp,1) == size(Kp,2)

    r = Kp*α-y
    term1 = dot(r,r)
    term2 = μ*dot(α, Kp, α)
    obj = term1 + term2

    return obj
end

"""
gradientRKHSfitdensitycostfunc( y::Vector{T},
                                K::Matrix{T},
                                μ::T,
                                a::Vector{T})
"""
function gradientRKHSfitdensitycostfunc(y::Vector{T},
                                        K::Matrix{T},
                                        μ::T,
                                        a::Vector{T}) where T <: Real
    #
    v = K*a
    return 2.0 .* (K*v .- K*y .+ μ .* v)
end


"""
gethessianRKHSfitdensitycostfunc(  y::Vector{T},
                                    K::Matrix{T},
                                    μ::T)
"""
function gethessianRKHSfitdensitycostfunc(  y::Vector{T},
                                            K::Matrix{T},
                                            μ::T) where T <: Real
    #
    out = (K*K + μ .* K) .* 2.0
    return forcesymmetric(out)
end

function forcesymmetric(K::Matrix{T})::Matrix{T} where T <: Real
    N = size(K,1)
    @assert N == size(K,2)

    for j = 1:N
        for i = 1:j-1
            K[i,j] = (K[i,j]+K[j,i])/2.0
        end
    end

    for i = 1:N
        for j = i+1:N
            K[i,j] = K[j,i]
        end
    end

    return K
end


"""
x here is p in my notes.
"""
function TRstepdensity( x_manifold::Vector{T},
                f_x,
                df♯_x::Array{T,D},
                𝐻::Function,
                f::Function,
                R::Function,
                metricfunc::Function,
                selfmetricfunc::Function,
                radius,
                config)::Tuple{Vector{T},Bool,T} where {T,D}
    ### setup
    x = x_manifold
    #min_retraction_bound = minimum( x ./ df♯_x )
    #radius = min(radius, min_retraction_bound)

    ### get η via tCG.
    r = copy(df♯_x) # in case tCG modifies r.

    η_tCG, tCG_flag = tCG(r, x, radius, 𝐻, metricfunc, selfmetricfunc, df♯_x, config.max_iter_tCG, config.verbose_flag)

    x_tCG = R(x_manifold,η_tCG,one(T))
    f_x_tCG = f(x_tCG)

    if f_x < f_x_tCG
        tCG_flag = false
    end

    ### debug.
    # println("x_tCG = ", x_tCG)
    # println("f_x_tCG = ", f_x_tCG)
    # println("f_x = ", f_x)
    # println()

    ### Decide if we should apply this update.
    η = η_tCG
    ρ = zero(T)
    if tCG_flag
        ρ_tCG = evalTRmodel(x_manifold, x, η_tCG, 𝐻, f, df♯_x, metricfunc, f_x, f_x_tCG)

        if config.verbose_flag
            println("ρ_tCG is ", ρ_tCG)
        end

        ρ = ρ_tCG

        # update radius.
        if ρ < 0.25
            radius = max(0.25*radius, config.minimum_TR_radius)

        elseif ρ > 0.75
            radius = min(2.0*radius, config.maximum_TR_radius)
        end

        if config.ρ_lower_acceptance < ρ < config.ρ_upper_acceptance

            if config.verbose_flag
                println("tCG successful solve and yields acceptable model. ρ = ", ρ)
            end

            consecutive_fails_TR = 0

            return x_tCG, true, radius
        end

    else
        if config.verbose_flag
            println("Trust region subproblem isn't solved. Radius is ", radius)
        end

        radius = max(0.25*radius, config.minimum_TR_radius)
    end

    if config.verbose_flag
        println()
    end

    return x_tCG, false, radius
end
