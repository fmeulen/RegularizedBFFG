wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack

### Parameters
p = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ=√((π^2)/2))
# with larger variance
mult_var = 3.0
p_mv = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ = √(mult_var*(π^2)/2))

# Forward simulate
forward(x, p) = p.ω + p.ψ *x + p.η * randn()

function xsim(x0, S, p)
  x = fill(x0, S+1)
  for i ∈ 2:S+1
    x[i] = forward(x[i-1], p)
  end
  x[2:end]
end

Random.seed!(10)
x0 = rand(Normal(p.ω/(1.0 - p.ψ), sqrt(p.η^2 / (1 - p.ψ^2))))    # Sample x0 from the stationary distribution 
S = 500 # Correct for time 0 (called tot_steps)  (Time steps)

X = xsim(x0, S, p)
R = exp.(X/2) .* randn(S)
V = log.(R.^2)
U = V - X

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


# Backward filtering
struct Message{Tc, TF, TH}
    c::Tc
    F::TF
    H::TH
end

fuse(m1::Message, m2::Message) = Message(m1.c+m2.c, m1.F+m2.F, m1.H+m2.H)

function pullback(m::Message, p)
    @unpack c, F, H = m
    @unpack ω, ψ, η, μ, σ = p
    C = η^2 + 1/H 
    Hnew = ψ^2 / C
    Fnew = ψ*(F/H - ω)/C
    cnew = c - logpdf(NormalCanon(F,H),0) + logpdf(Normal(F/H,C),ω)
    Message(cnew, Fnew, Hnew)
end   

leaf_coefs(V, p) = [Message(logpdf(Normal(p.μ, p.σ), v), 
                                (v-p.μ)/p.σ^2, p.σ^(-2)) for v in V]

function backwardfilter(V, p)
    leafmessages = leaf_coefs(V, p)
    m = leafmessages[end]
    ms = [m]
    S = length(V)
    for i in S-1:-1:1
        m = fuse(pullback(m, p), leafmessages[i])
        pushfirst!(ms, m)
    end
    ms
end


bf = backwardfilter(V,p)
bf_mv = backwardfilter(V,p_mv)


function guide(x, m::Message, p) 
    @unpack c, F, H = m
    @unpack ω, ψ, η, μ, σ = p
    μg = F/H
    σg2 = H^(-1)
    η2 = η^2
    μx = ω + ψ * x
    μ̃ = (μx*σg2 + μg*η2)/(η2 + σg2)
    σ̃ = sqrt((η2*σg2)/(η2+σg2))
    rand(Normal(μ̃, σ̃))
end

logweight(x, m::Message) = m.c + m.F * x + 0.5*x*m.H*x

robustify(m::Message, ϵ) = Message(m.c + log(ϵ),m.F, m.H)

function forwardguide(x0, bf, p; ϵ=0.1)
    @assert ϵ>0 "ϵ should be strictly positive"
    x = x0
    xs = [x]
    S = length(bf)
    for i in 1:S
         # Sampling from guided or unconditional?
        kg = exp(logweight(x,bf[i]))
        λ = kg/(kg + ϵ)
        m = robustify(bf[i], ϵ)
        x = rand()<λ ? guide(x, m, p) : forward(x, p)  
        push!(xs, x)
        # also compute logweight and save
    end
    xs
end

xᵒ = forwardguide(x0, bf, p)
plot!(p1, xᵒ, color="red")