"""Historical one-instance quadratic RCC sample.

Only module imports and filesystem wiring have been reorganized.
"""

from conformal_cl.quadratic import problem as quadratic_sample
from conformal_cl.quadratic.group_conformal import run_rcc
import logging
import os
from datetime import datetime

if __name__ == "__main__":
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join(
        "logs", f"RCC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Also print to console
        ],
    )

    logging.info(f"Starting RCC execution. Log file: {log_filename}")

    seeds = range(1)
    noises = [0.1]
    dims = [2]
    alpha = 0.1
    delta = 0.1

    for seed in seeds:
        for noise_std in noises:
            for dim in dims:
                logging.info(
                    f"Running with seed={seed}, noise_std={noise_std}, dim={dim}"
                )
                samples = quadratic_sample.generate_samples(
                    seed=seed, noise_std=noise_std, dim=dim
                )
                opt, is_f, t, true_f, min_constr, *_ = run_rcc(
                    samples, alpha, delta
                )
                logging.info(
                    f"Results: opt={opt}, is_f={is_f}, time={t}, true_f={true_f}, min_constr={min_constr}"
                )

    logging.info("RCC execution completed")
