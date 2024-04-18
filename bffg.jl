# Forward simulate (z ~ N(0,1))
forward(x, p, z) = p.ω + p.ψ * x + p.η * z

function forward(x0, S, p, Z)
  @assert length(Z)==S "length of innovations should equal S"
  x = Vector{typeof(x0)}(undef,S)# fill(x0, S)
  x[1] = forward(x0, p, Z[1])
  for i ∈ 2:S
    x[i] = forward(x[i-1], p, Z[i])
  end
  x
end

samplefromstationary(p) = rand(Normal(p.ω/(1.0 - p.ψ), sqrt(p.η^2 / (1 - p.ψ^2))))    

# Backward filtering
struct Message{Tc, TF, TH}
    c::Tc
    F::TF
    H::TH
end

fuse(m1::Message, m2::Message) = Message(m1.c+m2.c, m1.F+m2.F, m1.H+m2.H)

function pullback(m::Message, p)
    @unpack c, F, H = m
    @unpack ω, ψ, η = p
    C = η^2 + 1/H 
    Hnew = ψ^2 / C
    Fnew = ψ * (F/H - ω)/C
    cnew = c - logpdf(NormalCanon(F,H),0) + logpdf(Normal(F/H,C),ω)
    Message(cnew, Fnew, Hnew)
end   

function leaf_coefs(V, p) 
    @unpack μ, σ = p
    [Message(logpdf(Normal(μ, σ), v), (v-μ)/σ^2, σ^(-2)) for v in V]
end

function backwardfilter(V, p)
    leafmessages = leaf_coefs(V, p)
    m = normalize(leafmessages[end])
    ms = [m]
    S = length(V)
    for i in S-1:-1:1
        m = normalize(fuse(pullback(m, p), leafmessages[i]))
        pushfirst!(ms, m)
    end
    ms
end

function guide(x, m::Message, p, z) 
    @unpack F, H = m
    @unpack ω, ψ, η = p
    # pars for canonical normal distribution
    ν = F + (ω + ψ*x)/η^2
    P = H + η^(-2)
    ℱ = NormalCanon(ν, P)
    mean(ℱ) + std(ℱ) * z
end


g(x, m::Message) = exp(m.c + m.F * x - 0.5*x*m.H*x)

normalize(m::Message) = Message(logpdf(Normal(m.F, m.H),0), m.F, m.H) 

logpdf_logχ2(x) = x + logpdf(Chisq(1), exp(x))   
  
function logweights(x0, Xᵒ, V, p, bf, ϵ)
    m = bf[1]
    W = [log((g(x0,pullback(m, p)) + ϵ)/(g(Xᵒ[1], m)+ ϵ)) + logpdf_logχ2(V[1]-Xᵒ[1])]
    S = length(V)
    for i ∈ 2:S
        m = bf[i]
        w = log((g(Xᵒ[i-1],pullback(m, p)) + ϵ)/(g(Xᵒ[i], m)+ ϵ)) + logpdf_logχ2(V[i]-Xᵒ[i])
        push!(W, w)
    end
    W
end

sumlogweights(x0, bf, p, Z, V) = (ϵ) -> sum(forwardguide(x0, bf, p, Z, V, ϵ).lw)


function loglik(X, V, p)
    # only if ϵ=0
    @unpack μ, σ = p
    ll = 0.0
    for i ∈ eachindex(V)
        ll += logpdf_logχ2(V[i]-X[i]) - logpdf(Normal(X[i] + μ, σ), V[i])
    end
    ll
end
    
function forwardguide(x0, bf, p, Z, V, ϵ)
    #@assert ϵ>0 "ϵ should be strictly positive"
    S = length(bf)
    @assert S==length(Z) "length of innovations should equal S"
    x = x0
    xs = typeof(x0)[] #Vector{typeof(x0)}(undef,S) 
    λs = Float64[]
    guids = Bool[]
    for i in 1:S
         # Sampling from guided or unconditional?
        # m = bf[i]  # originally, but i think it is wrong
        m = pullback(bf[i],p)
        κg = g(x,m)
        λ = κg/(κg + ϵ) # prob to sample from guided
        z = Z[i]
        guid = rand()<λ
        x = guid ? guide(x, m, p, z) : forward(x, p, z)  
        push!(xs, x)
        push!(λs, λ)
        push!(guids, guid)
    end
    lw = logweights(x0, xs, V, p, bf, ϵ)
    (Xᵒ=xs, λs=λs, lw=lw, guids=guids)
end



function smc_ess(weights)
    normalized_weights = weights / sum(weights)
    1.0 / sum(normalized_weights.^2)
end