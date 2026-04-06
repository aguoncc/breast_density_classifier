# 1. confidence interval definition
from sklearn.metrics import cohen_kappa_score
import numpy as np

rate2 = ["C", "B", "D"]
rate1 = ["C", "B", "C"]

# method from google
def bootstrap_kappa(r1, r2, n_bootstraps=1000, alpha=0.95):
    bootstrapped_kappas = []
    n = len(r1)
    
    for _ in range(n_bootstraps):

        sample_indices = np.random.randint(0, n, n)
        r1_boot = np.array(r1)[sample_indices]
        r2_boot = np.array(r2)[sample_indices]
        
        k = cohen_kappa_score(r1_boot, r2_boot, weights='quadratic')
        bootstrapped_kappas.append(k)
        
    lower = np.percentile(bootstrapped_kappas, (1 - alpha) / 2 * 100)
    upper = np.percentile(bootstrapped_kappas, (1 + alpha) / 2 * 100)
    return lower, upper

# 2. kappa coefficients
kappa_unweight = cohen_kappa_score(rate2, rate1) # counts disagreement but doesn't care how big or small it was
kappa_linear = cohen_kappa_score(rate2, rate1, weights="linear") # disagreements' penalty grows evenly
kappa_quad = cohen_kappa_score(rate2, rate1, weights="quadratic") # bigger disagreements are penalized heavily compared to smaller ones

# 3. display
print(f"Unweighted: ", {kappa_unweight})
print(f"Linear: ", {kappa_linear})
print(f"Quadratic: ", {kappa_quad})

# confidence interval
ci_lower, ci_upper = bootstrap_kappa(rate1, rate2)
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")