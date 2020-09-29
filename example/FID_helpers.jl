function evalFIDcomponent(t::T, α, β, λ, Ω)::Complex{T} where T <: Real

    if t < zero(T)
        return zero(T)
    end

    term1 = im*Ω*t
    term2 = -λ*t

    return im*α*exp(term1+term2)*exp(im*β)
    #return α*exp(term1+term2)*exp(im*β)
end

function FIDcostfunc(   p::Vector{T2},
                        Ω_array::Vector{T},
                        λ_array::Vector{T},
                        N_pairs::Int,
                        𝓤,
                        DTFT_hs_𝓤::Vector{Complex{T}},
                        DTFT_h::Function)::T2 where {T <: Real, T2 <: Real}

    # parse.
    α_values = p[1:N_pairs]
    α_array = parseα(α_values)
    β_array = p[N_pairs+1:end]

    L = length(β_array)
    @assert length(α_array) == L

    # model.
    f = tt->sum( evalFIDcomponent(tt, α_array[l],
                    β_array[l],
                    λ_array[l],
                    Ω_array[l]) for l = 1:L )
    f_t = f.(t)

    DTFT_f = vv->computeDTFTch3eq29(f_t, vv, t)
    DTFT_hf = vv->(DTFT_f(vv) * DTFT_h(vv))

    # data.
    #DTFT_hs = vv->(DTFT_s(vv) * DTFT_h(vv))

    score = zero(T)
    for n = 1:length(𝓤)
        RHS = DTFT_hf(𝓤[n])
        LHS = DTFT_hs_𝓤[n]

        # score += abs( LHS - RHS )

        # corresponds to least-squares objective.
        score += abs2( LHS - RHS )
    end

    return score
end

function parseα( α_values::Vector{T}) where T <: Real

    # always even number of components.
    α_array = Vector{T}(undef, length(α_values)*2)
    for i = 1:length(α_values)
        α_array[i] = α_values[end-i+1]
        α_array[end-i+1] = α_values[end-i+1]
    end

    return α_array
end

function gettimerange(N::Int, fs::T) where T
   Ts::T = 1/fs

   return zero(T):Ts:(N-1)*Ts
end

# case 1D.
function computeDTFTch3eq29(h, u::T, Λ)::Complex{T} where T <: Real

    running_sum = zero(T)
    for i = 1:length(Λ)
        x = Λ[i]

        running_sum += h[i]*exp(-im*2*π*u*x)
    end

    return running_sum
end

function cosinetransitionbandpassimpulsefunc(x::T, a, b, δ)::Complex{T} where T <: Real

    term1 = evalDTFTintegralrising(x, a-δ, a+δ, a-δ, a+δ)
    term2 = evalDTFTintegralrect(x, a+δ, b-δ)
    term3 = evalDTFTintegralfalling(x, b-δ, b+δ, b-δ, b+δ)

    return (term1 + term2 + term3)/fs
end

function evalDTFTintegralrising(x::T,
                stop_freq,
                pass_freq,
                integration_lower_limit,
                integration_upper_limit) where T <: Real
    # rename.
    a = integration_lower_limit
    b = integration_upper_limit
    p = pass_freq
    s = stop_freq

    # singularity locations.
    x_0_negative = -1/(2*(p-s))
    x_0_positive = 1/(2*(p-s))

    # common intermediate value.
    k2 = 2*π*x*im

    ### term with negative exponent.
    A_negative::Complex{T} = zero(Complex{T})

    if x_0_negative - eps(T)*2 < x < x_0_negative + eps(T)*2
        # singularity case: when x is x_0_negative.
        A_negative = exp(-p*π*im/(p-s))*(b-a)

    else
        k1 = -π*im
        B = p*k1/(p-s)
        c = -k1/(p-s)
        q = c+k2

        A_negative = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### term with positive exponent.
    A_positive::Complex{T} = zero(Complex{T})

    if x_0_positive - eps(T)*2 < x < x_0_positive + eps(T)*2
        # singularity case: when x is x_0_positive.
        A_positive = exp(p*π*im/(p-s))*(b-a)

    else
        k1 = π*im
        B = p*k1/(p-s)
        c = -k1/(p-s)
        q = c+k2

        A_positive = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### third term.
    term3 = evalDTFTintegralrect(x, a, b)

    return A_positive/4 + A_negative/4 + term3/2
end


function evalDTFTintegralfalling(x::T,
                pass_freq,
                stop_freq,
                integration_lower_limit,
                integration_upper_limit) where T <: Real
    # rename.
    a = integration_lower_limit
    b = integration_upper_limit
    p = pass_freq
    s = stop_freq

    # singularity locations.
    x_0_negative = -1/(2*(s-p))
    x_0_positive = 1/(2*(s-p))

    # common intermediate value.
    k2 = 2*π*x*im

    ### term with negative exponent.
    A_negative::Complex{T} = zero(Complex{T})

    if x_0_negative - eps(T)*2 < x < x_0_negative + eps(T)*2
        # singularity case: when x is x_0_negative.
        A_negative = exp(-p*π*im/(s-p))*(b-a)

    else
        k1 = -π*im
        B = p*k1/(s-p)
        c = -k1/(s-p)
        q = c+k2
        A_negative = exp(B)/q * ( exp(b*q) - exp(a*q))
    end


    ### term with positive exponent.
    A_positive::Complex{T} = zero(Complex{T})

    if x_0_positive - eps(T)*2 < x < x_0_positive + eps(T)*2
        # singularity case: when x is x_0_positive.
        A_positive = exp(p*π*im/(s-p))*(b-a)

    else
        k1 = π*im
        B = p*k1/(s-p)
        c = -k1/(s-p)
        q = c+k2
        A_positive = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### third term.
    term3 = evalDTFTintegralrect(x, a, b)

    return A_positive/4 + A_negative/4 + term3/2
end

function evalDTFTintegralrect(x::T, a, b)::Complex{T} where T <: Real

    q = 2*π*x*im

    if zero(T) - eps(T)*2 < x < zero(T) + eps(T)*2
        # singularity case: when x is 0.
        return exp(q*b)*b - exp(q*a)*a
    end

    return (exp(b*q)-exp(a*q))/q
end

function gettimerangetunablefilter(N::Int, fs::T) where T
    @assert isodd(N)
    M = N - 1

    Ts::T = 1/fs

    A = fld(M,2)

    return (-A:1:A) * Ts
end
