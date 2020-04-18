

# this version of evaltrustregion() returns -Inf if ZERO_TOLERANCE.
# also ignores the sign of the computed ρ.
#   this means the calling function must check whether this is a descent update.
function evalTRmodel(  x_manifold,
                        x::Array{T,D},
                                η::Array{T,D},
                                𝐻::Function, # the function that computes 𝐻[η] efficiently.
                                f::Function,
                                df♯_x::Array{T,D},
                                metricfunc::Function,
                                f_x::T,
                                f_Rη::T)::T where {T,D}
    numerator = f_x - f_Rη

    # trust region model.
    g = ηη->(f_x + metricfunc(df♯_x, ηη, x_manifold) + 0.5*metricfunc(𝐻(ηη,x), ηη, x_manifold))

    z = zeros(T,size(η))

    denominator = g(z) - g(η)

    ### check output.
    ρ = numerator/denominator

    if !isfinite(ρ) || sign(numerator) < zero(T) # ascent direction.
        return zero(T)
    end

    return ρ
end



function TRstep( x_manifold::𝓜,
                f_x,
                df♯_x::Array{T,D},
                𝐻::Function,
                f::Function,
                R::Function,
                metricfunc::Function,
                selfmetricfunc::Function,
                radius,
                config)::Tuple{𝓜,Bool,T} where {𝓜,T,D}
    ### setup
    #x = convertMVNtoarray(x_manifold)
    x = x_manifold

    ### get η via tCG.
    r = copy(df♯_x) # in case tCG modifies r.

    η_tCG, tCG_flag = tCG(r, x, radius, 𝐻, metricfunc, selfmetricfunc, df♯_x, config.max_iter_tCG, config.verbose_flag)

    x_tCG = R(x_manifold,η_tCG,one(T))
    f_x_tCG = f(x_tCG)

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
            radius = max(0.25*radius, config.minimum_TR_radius)
        end
    end

    if config.verbose_flag
        println()
    end

    return x_tCG, false, radius
end
