# Can We Trust Food Delivery ETAs?
## Investigating Prediction Reliability and Calibration

### 1. Research Question

How reliable are machine-learning food delivery
time predictions across different weather and
traffic conditions? Can conformal calibration
improve prediction interval coverage?

### 2. Background

A single estimated arrival time does not communicate
how uncertain a delivery prediction is.

This project investigates prediction intervals
using an existing LightGBM quantile regression
pipeline, then evaluates ways to improve their
reliability.

### 3. Methodology

This work extends the original Food Delivery ETA
& Late-Risk Model by SilaShiva.

The experiments include:

1. Analyzing prediction error and interval coverage
   across weather and traffic conditions.
2. Applying conformal calibration to P10–P90
   prediction intervals.
3. Comparing performance before and after
   removing two potentially problematic features.

### 4. Results

#### Experiment 1: Weather and Traffic

The original model achieved 70.36% prediction
interval coverage on the development/test split,
below the intended 80%.

Stormy-weather orders showed 68.42% coverage.
Low-traffic orders showed 67.57% coverage,
despite having relatively low prediction error.

#### Experiment 2: Conformal Calibration

On a separate evaluation split of 6,839 orders:

| Metric | Original | Calibrated |
|---|---:|---:|
| Coverage | 70.70% | 80.39% |
| Average interval width | 8.79 min | 10.04 min |

A conformal adjustment of approximately
0.62 minutes per side increased observed coverage
by 9.69 percentage points.

#### Experiment 3: Feature Ablation

| Model | MAE | Coverage |
|---|---:|---:|
| Original | 3.18 min | 70.36% |
| Without pickup delay | 3.19 min | 70.11% |
| Without pickup delay and rider history | 3.16 min | 70.67% |

Removing these features produced only small
performance differences on the tested data.

### 5. Limitations

- These findings are based on a single dataset.
- Improved coverage comes with wider intervals.
- Conformal coverage guarantees require assumptions
  that may not hold for time-dependent data.
- The ablation experiments are exploratory.
- The prediction timing and remaining input
  features require further auditing.
- Small performance differences do not establish
  statistically significant improvements.
- The source and definition of the delivery-time
  target require verification before claiming
  an order-placement ETA system.

### 6. Conclusion

Conformal calibration improved observed interval
coverage from 70.70% to 80.39%, while increasing
average interval width by 1.25 minutes.

The feature ablation results suggest that removing
pickup-delay and rider-history features may be
possible without substantially reducing prediction
accuracy.

These results emphasize the importance of
evaluating both accuracy and uncertainty in
machine-learning systems.

### 7. Original Work and Attribution

Base implementation:
https://github.com/SilaShiva/delivery-eta-quantile

Extensions developed in this fork:
- Weather- and traffic-specific interval analysis
- Conformal calibration experiment
- Feature ablation comparisons
- Research visualizations
- Research summary and interpretation

The base code is licensed under MIT. This
extension retains attribution to the original author.
