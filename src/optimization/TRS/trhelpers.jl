


############## tCG-related.
function getrealrootfromquadratic(a::T, b::T, c::T,
                                     ZERO_TOLERANCE = 1e-12)::Tuple{T,Bool} where T


    if !isfinite(a) || !isfinite(b) || !isfinite(c)
        return convert(T,Inf), false
    end

    tmp = b*b - 4.0*a*c
    if tmp < ZERO_TOLERANCE
        #println("complex exit")
        return zero(T), false
    end

    tmp2 = sqrt(tmp)
    x1 = (-b + tmp2)/(2.0*a)
    x2 = (-b - tmp2)/(2.0*a)

    out = max(x1,x2)

    if out < ZERO_TOLERANCE
        #println("zero exit")
        return out, false
    end

    return out, true
end
## test: answer is 2/3
# a = 6.0
# b = 5.0
# c = -6.0
# r, status_flag = getrealrootfromquadratic(a,b,c)


function tCG(   r::Array{T,D},
                x,
                radius::T,
                𝐻::Function,
                genericmetricfunc::Function,
                selfmetricfunc::Function,
                df♯_x::Array{T,D},
                MAX_ITER::Int = 100,
                verbose_flag::Bool = true)::Tuple{Array{T,D},Bool} where {T,D}
    δ = -r
    η = zeros(T,size(r))
    α = zero(T)
    inner_prod_eval = one(T)

    # fall-back.
    default_η = -(1e-12) .* df♯_x

    iter = 1
    while iter <= MAX_ITER
        H_δ = 𝐻(δ,x) #evalHessfvector(H,δ)

        inner_prod_eval = genericmetricfunc(δ,𝐻(δ,x),x)
        if !isfinite(inner_prod_eval)
            return default_η, false
        end

        a = selfmetricfunc(δ,x)
        b = 2.0*genericmetricfunc(η,δ,x)
        c = -(radius^2 - selfmetricfunc(η,x))
        τ, τ_status_flag = getrealrootfromquadratic(a,b,c)

        # check condition 1.
        if inner_prod_eval <= zero(T) && τ_status_flag
            out = η + τ.*δ
            if !all(isfinite.(out))
                return default_η, false
            end

            if verbose_flag
                println("exit 1, τ is ", τ)

                if abs(inner_prod_eval) < 1e-6
                    println("warning, inner_prod_eval is ", inner_prod_eval)
                end
                if α > 1.0
                    println("warning, α is ", α)
                end
            end

            return out, true
        end

        # check condition 2/
        α = selfmetricfunc(r,x)/inner_prod_eval
        if !isfinite(α)
            return default_η, false
        end

        η_next = η + α.*δ
        if !all(isfinite.(η_next))
            return default_η, false
        end

        if sqrt(selfmetricfunc(η_next,x)) >= radius && τ_status_flag
            out = η + τ.*δ
            if !all(isfinite.(out))
                return default_η, false
            end

            if verbose_flag
                println("exit 2, τ is ", τ)

                if abs(inner_prod_eval) < 1e-6
                    println("warning, inner_prod_eval is ", inner_prod_eval)
                end
                if α > 1.0
                    println("warning, α is ", α)
                end
            end

            return out, true
        end

        # custom stopping criterion can go here...

        # update.
        r_next = r + α .* H_δ

        β = selfmetricfunc(r_next,x)/selfmetricfunc(r,x)
        if !isfinite(β)
            return default_η, false
        end

        δ = -r_next + β .* δ

        r = r_next

        iter += 1
    end

    if verbose_flag
        if abs(inner_prod_eval) < 1e-6
            println("warning, inner_prod_eval is ", inner_prod_eval)
        end
        if α > 1.0
            println("warning, α is ", α)
        end
    end

    return default_η, false
end


############## RW algorithm.


# function minimumeigenpair(D)
#     s,Q = eig(D)
#     λ_min, i = findmin(s)
#     return λ_min, Q[:,i]
# end
#
# function RWalgorithmfunctionk!(D::Matrix{T},s_sq::T,t::T)::Tuple{T,Vector{T}} where T
#     # update D.
#     D[1] = t
#
#     # get minimum eigen-pair.
#     λ_min, v_min = minimumeigenpair(D)
#     if typeof(λ_min) <: Complex
#         λ_min = real(λ_min)
#     end
#
#     # evaluate score
#     score = (s_sq + one(T))*λ_min - t
#     #@assert isfinite(score)
#     return score, v_min
# end
#
# function RWalgorithm(   A::Matrix{T},
#                         a,
#                         s_sq,
#                         t0,
#                         lower_bound = -1e22,
#                         upper_bound = 1e22,
#                         verbose_flag = true,
#                         ZERO_TOLERANCE = 1e-12)::Tuple{Vector{T},Bool}
#     D0 = [t0 -a'; -a A]
#
#     k = tt->(-(RWalgorithmfunctionk!(D0,s_sq,tt)[1])) # negative since we want to maximize.
#     result =  Optim.optimize(k, lower_bound, upper_bound, Optim.Brent())
#
#     status_flag = result.converged
#     if status_flag
#         t = result.minimizer[1]
#
#         # get minimum eigen-pair.
#         D0[1] = t
#         λ_min, v_min = minimumeigenpair(D0)
#         if λ_min < zero(T) && abs(v_min[1]) > ZERO_TOLERANCE
#             η_star = v_min[2:end]./v_min[1]
#
#             if verbose_flag
#                 println("RW algorithm successful: solution on boundary.")
#             end
#
#             return η_star, all(isfinite.(η_star))
#         else
#             η_star = A\a
#
#             if verbose_flag
#                 println("RW algorithm successful: solution within boundary.")
#             end
#
#             return η_star, all(isfinite.(η_star))
#         end
#     end
#
#     if verbose_flag
#         println("RW algorithm maximization of k(t) failed.")
#     end
#     return Vector{T}(undef,0), false
# end
