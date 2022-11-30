using Distributions
using LinearAlgebra
using QuadGK
using PDMats
include("expit.jl");

# -----------------------------------------------------------------------------
# Function to evaluate the integral for continuous and binary Cauchy
# -----------------------------------------------------------------------------
function integrand( u, par, y, f, model)

	if(model == "continuous")
        # u =  (beta - y/f) / (sigma / |f| )

		delta = par[1]
		gamma = exp(par[2])
		sigma = exp(par[3])

		val = pdf( Normal(0,1), u)
		val *= pdf( Cauchy( abs(f)*(delta - y/f)/sigma, gamma*abs(f)/sigma), u)
	elseif(model == "discrete")
        # u = (beta - delta) / gamma

		delta = par[1]
		gamma = exp(par[2])

		val = pdf( Bernoulli( expit( (u * gamma + delta) * f) ), y )
		val *= pdf( Cauchy(0, 1), u )
	end

	return(val)
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Function to compute the log-posterior of target model parameters
# -----------------------------------------------------------------------------
function log_post( par, y, f, model)
	delta = par[1]
	log_gamma = par[2]


	if(model == "continuous")
		log_sigma = par[3]

		# Continuous Cauchy model
		# s2 = 39
		# k = exp(-.5 * s2^2)/sqrt(2*pi) = 0.0
		integral = quadgk(u -> integrand.(u, Ref(par), y, f, model), -39, 39)
		val = log(integral[1]) - log_sigma

		# prior on log(sigma)
		val += logpdf( Normal(0,3), log_sigma) / n_T

	elseif(model == "discrete")
        # these bounds give us error on order of 10^-18
        # qcauchy(10^(-18), 0, 1)
		# integral = quadgk(u -> integrand.(u, Ref(par), y, f, model),  -32 * 10^(16), 32 * 10^(16) )[1]
        # integral = quadgk(u -> integrand.(u, Ref(par), y, f, model),  -Inf, 0 )[1]
        # integral += quadgk(u -> integrand.(u, Ref(par), y, f, model),  0, Inf )[1]
        integral = quadgk(u -> integrand.(u, Ref(par), y, f, model),  -Inf, Inf )[1]


		val = log(integral)
	end

	# priors on delta and log(gamma)
	val += logpdf( Normal(1, 20), delta) / n_T
	val += logpdf( Normal(0, 3), log_gamma) / n_T

	return(val)

end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# The mcmc routine
# -----------------------------------------------------------------------------
function mcmc_routine( y, f, model, init_par, steps, burnin)

	par = init_par
	n_par = length(par)
	chain = Array{Float64}(undef, (steps, n_par))

	pcov = Diagonal(ones(n_par))
	pscale = .000001
	accept = 0

    global n_T = length(y)

	# Evaluate the log_post of the initial par
	log_dens_prev = sum(log_post.( Ref(par), y, f, model))
	if isinf(log_dens_prev)
		println("Infinite log-posterior; choose better initial parameters")
	else
		# Begin the MCMC algorithm ------------------------------------------------
		chain[1,:] = par
		for ttt in 2:steps

			# Propose an update
			proposal = rand(MvNormal( par, pcov*pscale))

			# Compute the log density for the proposal
			log_dens = sum(log_post.( Ref(proposal), y, f, model))

			# Only propose valid parameters during the burnin period
			if ttt < burnin
				while isinf(log_dens)
					println("bad proposal")
					proposal = rand(MvNormal( par, pcov*pscale))
				  log_dens = sum(log_post.( Ref(proposal), y, f, model))
				end
			end

			# Evaluate the Metropolis-Hastings ratio
			if( log_dens - log_dens_prev > log(rand()) )
				log_dens_prev = log_dens
				par = proposal
				accept = accept +1
			end
			chain[ttt,:] = par

			# Proposal tuning scheme ------------------------------------------------
			if ttt < burnin
				# During the burnin period, update the proposal covariance in each step
				# to capture the relationships within the parameters vectors for each
				# transition.  This helps with mixing.
				if ttt == 100  pscale = 1  end

				if 100 <= ttt & ttt <= 2000
					temp_chain = chain[1:ttt,:]
					pcov = cov( unique( temp_chain, dims=1), dims=1)
				elseif 2000 < ttt
					temp_chain = chain[(ttt-2000):ttt,:]
					pcov = cov( unique( temp_chain, dims=1), dims=1)
				end
				if any( isnan, pcov)  pcov = Diagonal(ones(n_par))  end

				# Tune the proposal covariance for each transition to achieve
				# reasonable acceptance ratios.
				if ttt % 30 == 0
					if ttt % 480 == 0
						accept = 0
					elseif accept / (ttt % 480) < .4
						pscale = (.75^2)*pscale
					elseif accept / (ttt % 480) > .5
						pscale = (1.25^2)*pscale
					end
				end
			end
			# -----------------------------------------------------------------------

			# Restart the acceptance ratio at burnin.
			if ttt == burnin  accept = 0  end
			if ttt%100 == 0  println("---> ",ttt)  end
            # println("----> ", ttt)
		end
		# -------------------------------------------------------------------------
	end

	println(accept/(steps-burnin))
	return( [chain[burnin:steps,:], accept/(steps-burnin), pscale])
end
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
