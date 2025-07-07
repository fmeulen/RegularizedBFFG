using Plots
function KLGaussian(Q1,Q2)
    # KL divergence between two Gaussian distributions
    # Q1 = (mu1, sigma1)
    # Q2 = (mu2, sigma2)
    mu1, sigma1 = Q1
    mu2, sigma2 = Q2
    return log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2)/(2*sigma2^2) - 0.5
end

fF(x,p) = x * p.v/p.σ2 
fH(x,p) = x^2/p.σ2
μ(x) = 2.0

p = (v=-2.2, x0=0.0, Q = 0.01, σ2 =0.1)
p = (v=0.0, x0=0.0, Q = 0.01, σ2 =0.1)


function KL(a, ã, p)
   α = fF(a,p) + μ(p.x0)/p.Q
   α̃ = fF(ã,p) + μ(p.x0)/p.Q
   η2 = fH(a,p) + p.Q^(-1)
   η̃2 = fH(ã,p) + p.Q^(-1)
   return KLGaussian((α/η2, sqrt(1/η2)), (α̃/η̃2, sqrt(1/η̃2)))
end

KL(a,p) = (ã) -> KL(a, ã, p)


a = 0.05

# additive perturbation
ϵseq = -1.7:0.01:0.3
kl = KL(a,p).(a .+ ϵseq)     

# multiplicative perturbation
ϵseq = 0.5:.01:1.5
kl = KL(a,p).(a .* ϵseq)  


pl = plot(ϵseq, kl, label="KL divergence", xlabel="ϵ", ylabel="KL divergence", title="KL divergence vs ϵ")
pl_log = plot(ϵseq, log.(kl), label="KL divergence", xlabel="ϵ", ylabel="KL divergence", title="KL divergence vs ϵ")
plot(pl, pl_log, layout=(2,1), size=(800, 600))

# find derivative with respect to ϵ
using ForwardDiff

der = map(x-> ForwardDiff.derivative(KL(a,p),x), a .+ ϵseq)
plot( a .+ ϵseq, der, label="derivative", xlabel="ϵ", ylabel="derivative", title="Derivative of KL divergence vs ϵ")