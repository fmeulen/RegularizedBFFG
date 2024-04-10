wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack

### Parameters
mult_var = 3.0
p = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27,
         σ=√((π^2)/2), modσ = √(mult_var*(π^2)/2))

function xsim(x0, S, p)
  x = fill(x0, S+1)
  for i ∈ 2:S+1
    x[i] = p.ω + p.ψ *x[i - 1] + p.η * randn()
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

struct Message{Tc, TF, TH}
    c::Tc
    F::TF
    H::TH
end

fuse(m1::Message, m2::Message) = Message(m1.c+m2.c, m1.F+m2.F, m1.H+m2.H)

function pullback(m::Message, p)
    @unpack c, F, H = m
    C = p.η^2 + 1/H 
    Hnew = p.ψ^2 / C
    Fnew = p.ψ*(F/H - p.ω)/C
    cnew = c - logpdf(NormalCanon(F,H),0) + logpdf(Normal(F/H,C),p.ω)
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


