function FIDDTFTphasecostfunc(   p::Vector{T2},
                        Ω_array::Vector{T},
                        λ_array::Vector{T},
                        α_array::Vector{T},
                        𝓣,
                        𝓤,
                        DTFT_hs_𝓤::Vector{Complex{T}},
                        DTFT_h_𝓤::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    L = length(Ω_array)

    β_array = p

    @assert length(α_array) == L == length(β_array) == length(λ_array)

    # model.
    f = tt->sum( evalFIDcomponent(tt, α_array[l],
                    β_array[l],
                    λ_array[l],
                    Ω_array[l]) for l = 1:L )
    f_𝓣 = f.(𝓣)

    DTFT_f = vv->computeDTFTch3eq29AD(f_𝓣, vv, 𝓣)
    #DTFT_hf = vv->(DTFT_f(vv) * DTFT_h(vv))

    # data.
    #DTFT_hs = vv->(DTFT_s(vv) * DTFT_h(vv))

    score = zero(T2)
    for n = 1:length(𝓤)

        LHS = DTFT_hs_𝓤[n]

        #RHS = DTFT_hf(𝓤[n])
        RHS = DTFT_f(𝓤[n]) * DTFT_h_𝓤[n]

        # corresponds to least-squares objective.
        score += abs2( LHS - RHS )
    end

    return score
end

# least squares objective on F and S.
function evalFIDFTphasecostfunc(   βs::Vector{T2},
                        αs,
                        λs::Vector{T},
                        κs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    #
    N = length(𝓟)
    @assert length(S_𝓟) == N

    score = zero(T2)
    for n = 1:N

        F_p = evalFIDFT(𝓟[n], αs, βs, λs, κs, ξs)

        # corresponds to least-squares objective.
        score += abs2( S_𝓟[n] - F_p )
    end

    return score
end


# methods related to the Fourier transform of FID.
function evalFIDFT(p, αs, βs::Vector{T}, λs, κs, ξs)::Complex{T} where T <: Real
    L = length(αs)
    @assert L == length(βs) == length(λs) == length(κs) == length(ξs)

    out::Complex{T} = zero(Complex{T})
    for l = 1:L
        numerator = αs[l]*im*exp(im*βs[l])
        denominator = im*(ξs[l] + κs[l]*p) + λs[l]

        out += numerator/denominator
    end

    return out
end



##### standardized.

function evalFIDFT( p::T,
                    αs::Vector{T2},
                    βs::Vector{T2},
                    λs::Vector{T},
                    ξs::Vector{T})::Complex{T2} where {T <: Real, T2 <: Real}

    L = length(αs)
    @assert L == length(βs) == length(λs) == length(ξs)

    out::Complex{T2} = zero(Complex{T2})
    for l = 1:L
        numerator = αs[l]*exp(im*βs[l])
        denominator = im*( p - ξs[l] ) + λs[l]

        out += numerator/denominator
    end

    return out
end

# least squares objective on F and S.
function evalFIDFTαβcostfunc(   p::Vector{T2},
                        λs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    # parse
    L = length(ξs)

    α_values = p[1:end-L]
    αs = parseα(α_values, L)

    βs = p[end-L+1:end]

    # set up.
    N = length(𝓟)
    @assert length(S_𝓟) == N

    # compute cost.
    cost = zero(T2)
    for n = 1:N

        F_eval = evalFIDFT(𝓟[n], αs, βs, λs, ξs)

        # corresponds to least-squares objective.
        cost += abs2( S_𝓟[n] - F_eval )
    end

    return cost
end


# set up intermediate storage for the cost function and its gradient.
function setupFTFIDαβ𝓛(N_pairs::Int, L::Int, N::Int, dummy_val::T) where T <: Real
    #
    @assert L == 2*N_pairs || L == 2*N_pairs-1

    αs_persist = Vector{T}(undef, N_pairs)
    βs_persist = Vector{T}(undef, L)

    ∂𝓛_∂β_eval_persist = zeros(T, L)
    ∂𝓛_∂α_eval_persist = zeros(T, L)

    ∂𝓛_∂a_eval_persist = zeros(T, N_pairs)

    diff_persist = Vector{Complex{T}}(undef, N)

    #d𝓛_p = Vector{T}(undef, L + N_pairs)

    return αs_persist, βs_persist, ∂𝓛_∂β_eval_persist,
        ∂𝓛_∂α_eval_persist, ∂𝓛_∂a_eval_persist, diff_persist
end

# mutates αs, βs, ∂𝓛_∂β_eval, ∂𝓛_∂α_eval, ∂𝓛_∂a_eval, diff
function evalFIDFTαβcostfuncgradient!(   αs, βs, ∂𝓛_∂β_eval, ∂𝓛_∂α_eval, ∂𝓛_∂a_eval, diff,
                        p::Vector{T},
                        λs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}}) where T <: Real

    # parse
    L = length(ξs)

    α_values = p[1:end-L]
    parseα!(αs, α_values, L)

    βs[:] = p[end-L+1:end]
    #@assert L == length(βs)

    # set up.
    N = length(𝓟)
    @assert length(S_𝓟) == N

    N_pairs = length(α_values)

    ## pre-compute.
    #diff = collect( evalFIDFT(𝓟[n], αs, βs, λs, ξs) - S_𝓟[n] for n = 1:N )

    resize!(diff, N)
    for n = 1:N
        diff[n] = evalFIDFT(𝓟[n], αs, βs, λs, ξs) - S_𝓟[n]
    end

    # compute gradient.

    resize!(∂𝓛_∂β_eval, L)
    fill!(∂𝓛_∂β_eval, zero(T))

    resize!(∂𝓛_∂α_eval, L)
    fill!(∂𝓛_∂α_eval, zero(T))

    for l = 1:L
        for n = 1:N

            # compute intermediates.
            B = λs[l]^2 + (𝓟[n]-ξs[l])^2

            diff_r = real(diff[n])
            diff_i = imag(diff[n])

            ξm𝓟cβ = (ξs[l]-𝓟[n])*cos(βs[l])
            𝓟mξsβ = (𝓟[n]-ξs[l])*sin(βs[l])
            λcβ = λs[l]*cos(βs[l])
            λsβ = λs[l]*sin(βs[l])

            # gradient wrt α[l].
            factor1 = 𝓟mξsβ + λcβ
            term1 = diff_r*factor1

            factor2 = ξm𝓟cβ + λsβ
            term2 = diff_i*factor2

            ∂𝓛_∂α_eval[l] += (term1 + term2)/B

            # gradient wrt β[l].
            factor1 = -ξm𝓟cβ - λsβ
            term1 = diff_r*factor1

            factor2 = 𝓟mξsβ + λcβ
            term2 = diff_i*factor2

            ∂𝓛_∂β_eval[l] += (term1 + term2)*αs[l]/B
        end

        ∂𝓛_∂α_eval[l] = 2*∂𝓛_∂α_eval[l]
        ∂𝓛_∂β_eval[l] = 2*∂𝓛_∂β_eval[l]
    end

    # account for α := parseα(a).

    resize!(∂𝓛_∂a_eval, N_pairs)
    #fill!(∂𝓛_∂a_eval, zero(T))

    for i = 1:N_pairs
        ∂𝓛_∂a_eval[end-i+1] = ∂𝓛_∂α_eval[i] + ∂𝓛_∂α_eval[end-i+1]
        #α_array[i] = α_values[end-i+1]
        #α_array[end-i+1] = α_values[end-i+1]
    end

    if 2*N_pairs > L
        # case odd.
        ∂𝓛_∂a_eval[1] = ∂𝓛_∂α_eval[N_pairs]
    end

    return [∂𝓛_∂a_eval; ∂𝓛_∂β_eval]
end

function evalFIDFTαβcostfuncgradient(   p::Vector{T},
                        λs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}}) where T <: Real

    # parse
    L = length(ξs)
    N_pairs = length(p) - L

    # set up.
    αs, βs, ∂𝓛_∂β_eval, ∂𝓛_∂α_eval, ∂𝓛_∂a_eval = setupFTFIDαβ𝓛(N_pairs, L, length(S_𝓟), one(T))

    cost = evalFIDFTαβcostfuncgradient!(αs, βs, ∂𝓛_∂β_eval, ∂𝓛_∂α_eval, ∂𝓛_∂a_eval,
                            p,
                            λs,
                            ξs,
                            𝓟,
                            S_𝓟)

    return cost
end


function evalFIDFTαβcostfuncslow(   p::Vector{T},
                        λs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}}) where T <: Real

    # parse
    L = length(ξs)

    α_values = p[1:end-L]
    αs = parseα(α_values, L)

    βs = p[end-L+1:end]
    #@assert L == length(βs)

    # set up.
    N = length(𝓟)
    @assert length(S_𝓟) == N

    # pre-compute.
    F_𝓟 = collect( evalFIDFT(𝓟[n], αs, βs, λs, ξs) for n = 1:N )

    # compute gradient.
    cost = zero(T)
    for n = 1:N

        diff_r = real(F_𝓟[n]) - real(S_𝓟[n])
        diff_i = imag(F_𝓟[n]) - imag(S_𝓟[n])

        cost += diff_r^2 + diff_i^2
    end

    return cost
end

# https://www.wolframalpha.com/input/?i=real+part+of+a*exp%28i*b%29%2F%28c%2Bi*%28x-d%29%29
# https://www.wolframalpha.com/input/?i=imaginary+part+of+a*exp%28i*b%29%2F%28c%2Bi*%28x-d%29%29
function realFTFID(p, α, β::Vector{T}, λ, ξ) where T <: Real
    L = length(α)

    out = zero(T)
    for l = 1:L

        factor1 = (p-ξ[l])*sin(β[l]) + λ[l]*cos(β[l])
        denominator = λ[l]^2 + (p-ξ[l])^2

        out += factor1*α[l]/denominator
    end

    return out
end

function imagFTFID(p, α, β::Vector{T}, λ, ξ) where T <: Real
    L = length(α)

    out = zero(T)
    for l = 1:L

        factor1 = (ξ[l]-p)*cos(β[l]) + λ[l]*sin(β[l])
        denominator = λ[l]^2 + (p-ξ[l])^2

        out += factor1*α[l]/denominator
    end

    return out
end
# S = pp->evalFIDFT(pp, αs_oracle, β_oracle, λs, ξs)
# Sr_AN = pp->realFTFID(pp, αs_oracle, β_oracle, λs, ξs)
#
# discrepancy = norm(real(S(4.5)) - Sr_AN(4.5))
# println("real S: discrepancy = ", discrepancy)
# println()
#
# Si_AN = pp->imagFTFID(pp, αs_oracle, β_oracle, λs, ξs)
# discrepancy = norm(imag(S(4.5)) - Si_AN(4.5))
# println("imag S: discrepancy = ", discrepancy)
# println()
