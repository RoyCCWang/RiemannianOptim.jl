
# from ℝ to [-π, π].
function projectcircle(x::T)::T where T <: Real
    y = mod(x, 2*π)
    if y > π
        return -(2*π-y)
    end

    return y
end

function circleretraction(p::T, X::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    # b = one(T2)
    # ϵ = 1e-12
    # a::T2 = b+abs(p)
    # multiplier = sqrt(a)
    #
    # # limit to the values within 2*π from p.
    # limit = π-ϵ #2*π-ϵ
    #
    # t = t_in
    # if multiplier > limit
    #     t = clamp(t, 0.0, limit/multiplier)
    # end
    # q = t*X

    a::T2 = π^2
    multiplier = sqrt(a)
    q = t*X

    out = p + t*X*multiplier/sqrt(a+(t*X)^2)
    return out
end

function circleretractionwithproject(p::T, X::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    return projectcircle(circleretraction(p, X, t))
end
