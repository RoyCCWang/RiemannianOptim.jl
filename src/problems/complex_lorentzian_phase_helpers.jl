
# least squares objective on F and S.
function evalcLβcostfunc(   βs::Vector{T2},
                        αs::Vector{T},
                        λ::T,
                        Ωs::Vector{T},
                        U,
                        S_U::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    #
    N = length(U)
    @assert length(S_U) == N

    score = zero(T2)
    for n = 1:N

        F_p = evalcomplexLorentzian(U[n], αs, βs, λ, Ωs)

        # corresponds to least-squares objective.
        score += abs2( S_U[n] - F_p )
    end

    return score
end

function evalcomplexLorentzian(u, α, β::T, λ, Ω)::Complex{T} where T <: Real
    return α*exp(im*β)/(λ + im*(2*π*u - Ω))
end

# methods related to the Fourier transform of FID.
function evalcomplexLorentzian(u, αs, βs::Vector{T}, λ, Ωs)::Complex{T} where T <: Real
    L = length(βs)
    @assert L == length(βs) == length(Ωs)

    out = sum( evalcomplexLorentzian(u, αs[l], βs[l], λ, Ωs[l]) for l = 1:L )

    return out
end



# set up intermediate storage for the cost function and its gradient.
function setupcLβ𝓛(L::Int, N::Int, dummy_val::T) where T <: Real
    #
    βs_persist = Vector{T}(undef, L)

    ∂𝓛_∂β_eval_persist = zeros(T, L)

    diff_persist = Vector{Complex{T}}(undef, N)

    #d𝓛_p = Vector{T}(undef, L + N_pairs)

    return βs_persist, ∂𝓛_∂β_eval_persist, diff_persist
end

# mutates βs, ∂𝓛_∂β_eval, diff.
# ω := 2*π*u.
function evalcLβcostfuncgradient!(   βs, ∂𝓛_∂β_eval, diff,
                        p::Vector{T},
                        αs::Vector{T},
                        λ::T,
                        Ωs::Vector{T},
                        U,
                        S_U::Vector{Complex{T}}) where T <: Real

    # parse
    L = length(Ωs)

    βs[:] = p
    #@assert L == length(βs)

    # set up.
    N = length(U)
    @assert length(S_U) == N

    ## pre-compute.
    #diff = collect( evalFIDFT(U[n], αs, βs, λ, Ωs) - S_U[n] for n = 1:N )

    resize!(diff, N)
    for n = 1:N
        diff[n] = evalcomplexLorentzian(U[n], αs, βs, λ, Ωs) - S_U[n]
    end

    # compute gradient.

    resize!(∂𝓛_∂β_eval, L)
    fill!(∂𝓛_∂β_eval, zero(T))

    for l = 1:L
        for n = 1:N

            ω = 2*π*U[n]

            # compute intermediates.
            B = λ^2 + (ω-Ωs[l])^2

            diff_r = real(diff[n])
            diff_i = imag(diff[n])

            ΩmUcβ = (Ωs[l]-ω)*cos(βs[l])
            UmΩsβ = (ω-Ωs[l])*sin(βs[l])
            λcβ = λ*cos(βs[l])
            λsβ = λ*sin(βs[l])

            # gradient wrt β[l].
            factor1 = -ΩmUcβ - λsβ
            term1 = diff_r*factor1

            factor2 = UmΩsβ + λcβ
            term2 = diff_i*factor2

            ∂𝓛_∂β_eval[l] += (term1 + term2)*αs[l]/B
        end

        ∂𝓛_∂β_eval[l] = 2*∂𝓛_∂β_eval[l]
    end

    return ∂𝓛_∂β_eval
end

function evalcLβcostfuncgradient(   p::Vector{T},
                        αs::Vector{T},
                        λ::T,
                        Ωs::Vector{T},
                        U,
                        S_U::Vector{Complex{T}}) where T <: Real

    # parse
    L = length(Ωs)
    # set up.
    βs, ∂𝓛_∂β_eval, diff = setupcLβ𝓛(L, length(S_U), one(T))

    ∂𝓛_∂β_eval_out = evalcLβcostfuncgradient!(βs, ∂𝓛_∂β_eval, diff,
                            p,
                            αs,
                            λ,
                            Ωs,
                            U,
                            S_U)

    return ∂𝓛_∂β_eval_out
end
