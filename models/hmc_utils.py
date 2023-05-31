"""Implementation of Hamiltonian Monte Carlo in PyTorch."""


import torch
import numpy as np
import tqdm


def get_kinetic_energy_diff(momentum1, momentum2, diagonal_mass_inv):
  return sum([(0.5 * (m1**2 - m2**2) * mass).sum() for m1, m2, mass in
              zip(momentum1, momentum2, diagonal_mass_inv)])


def get_accept_prob(
   log_prior_diff_fn, log_likelihood1, params1, momentum1,
   log_likelihood2, params2, momentum2, diagonal_mass_inv
):
  with torch.no_grad():
    energy_diff = get_kinetic_energy_diff(momentum1, momentum2, diagonal_mass_inv)
    energy_diff -= log_likelihood1 - log_likelihood2
    energy_diff -= log_prior_diff_fn(params1, params2)
    if energy_diff.item() > 1:
      accept_prob = 1.
    else:
      accept_prob = np.minimum(1., np.exp(energy_diff.item()))

  return accept_prob


def leapfrog(
   model, log_prob_fn, init_momentum, init_grad, step_size, n_leapfrog_steps, diagonal_mass_inv
):
  grad = init_grad
  momentum = init_momentum
  log_likelihood = None
  for i in range(n_leapfrog_steps):
    momentum = [m + 0.5 * step_size * g for (m, g) in zip(momentum, grad)]
    for p, m, mass in zip(model.parameters(), momentum, diagonal_mass_inv):
      p.data += mass * m * step_size

    model.zero_grad()
    log_prob, log_likelihood = log_prob_fn(model)
    log_prob.backward()
    grad = [p.grad.clone() for p in model.parameters()]

    momentum = [m + 0.5 * step_size * g for (m, g) in zip(momentum, grad)]
  return momentum, grad, log_likelihood


def hmc_update(
  model, log_prob_fn, log_prior_diff_fn, log_likelihood, grad, step_size,
  n_leapfrog_steps, diagonal_mass_inv, do_mh_correction=True
):
  momentum = [torch.randn_like(p) for p in model.parameters()]
  init_params = [p.clone() for p in model.parameters()]

  new_momentum, new_grad, new_log_likelihood = leapfrog(
      model, log_prob_fn, momentum, grad, step_size, n_leapfrog_steps, diagonal_mass_inv)

  accept_prob = get_accept_prob(
      log_prior_diff_fn, log_likelihood, init_params, momentum,
      new_log_likelihood, list(model.parameters()), new_momentum, diagonal_mass_inv)
  if do_mh_correction:
    accepted = np.random.uniform() < accept_prob
  else:
    accepted = True

  if accepted:
    log_likelihood = new_log_likelihood
    grad = new_grad
  else:
    for p, p_init in zip(model.parameters(), init_params):
      p.data = p_init.data

  return log_likelihood, grad, accept_prob, accepted


def run_hmc(n_samples_per_chain, net, log_density_fn, log_prior_diff_fn, step_size, path_length, adapt_step_size, n_burn_in, do_mh_correction, diagonal_mass_inv=None):
  llhs = []

  net.zero_grad()
  log_prob, log_likelihood = log_density_fn(net)
  log_prob.backward()
  grad = [p.grad for p in net.parameters()]

  # for adaptation
  gamma = 0.05
  t = 10.0
  kappa = 0.75
  mu = np.log(10.0 * step_size)
  log_best_step_size = 0.0
  closeness = 0.0

  n_accepted = 0.0
  
  param_size = len(torch.nn.utils.parameters_to_vector(net.state_dict().values()).detach())
  params = torch.zeros([n_samples_per_chain, param_size]).to(log_prob)

  if diagonal_mass_inv is None:
      diagonal_mass_inv = [torch.ones_like(p) for p in net.parameters()]

  print("HMC leapfrog")
  for iteration in tqdm.tqdm(range(n_samples_per_chain + n_burn_in), disable=True):
      n_leapfrog_steps = min(200, int(np.ceil(path_length / step_size)))
      # print(iteration, "N_LEAPFROG", n_leapfrog_steps)
      if iteration % 200 == 0:
        print("\titer %d\tstep size %.6f\t leapfrog steps %d" % (iteration, step_size, n_leapfrog_steps))
      log_likelihood, grad, accept_prob, accepted = hmc_update(
              net, log_density_fn, log_prior_diff_fn, 
              log_likelihood, grad, step_size, n_leapfrog_steps, diagonal_mass_inv, do_mh_correction)

      if adapt_step_size and iteration < n_burn_in:
        if np.isnan(accept_prob):
          accept_prob = 0.0

        iter = float(iteration + 1)
        closeness_frac = 1.0 / (iter + t)
        closeness = (1 - closeness_frac) * closeness + (closeness_frac * (0.8 - accept_prob))
        log_step_size = mu - np.sqrt(iter) / gamma * closeness

        step_frac = np.power(iter, -kappa)
        log_best_step_size = (step_frac * log_step_size) + (1 - step_frac) * log_best_step_size
        step_size = np.exp(log_step_size)
        
        if (path_length / step_size) > 200:
               path_length = path_length / 2
               print("new path length", path_length, "step_size", step_size)

        if iteration + 1 == n_burn_in:
          step_size = np.exp(log_best_step_size)
          print("\tfinal step size %.6f\t" % step_size, "leapfrog steps", min(1000, int(np.ceil(path_length / step_size))))

      net.zero_grad()

      # save samples after burn-in
      if iteration >= n_burn_in:
        # Evaluation
        if accepted:
            n_accepted += 1
        params_hmc = torch.nn.utils.parameters_to_vector(net.state_dict().values()).detach().unsqueeze(0)
        params_hmc = params_hmc.detach()
        params[iteration - n_burn_in] = params_hmc
        llhs.append(log_likelihood.sum().cpu().detach())
  
  print("\taccept prob", n_accepted / n_samples_per_chain)

  return params, llhs
