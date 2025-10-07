# Fourier Som
Self Organizing Map that processes input feature vectors using Discrete Fourier Transform.
 It treats each data row as a short wave, computes its DFT and trains the SOM using a distance metric that handles magnitudes and phases separately. Euclidean for magnitudes and circular chordal distance for phases.
 The code supports CSV input, with optional labels(it only supoprts one non numeric column) and JSON output for easy result interpretation.
 # Usage 
 The program supports the following arguments
```
 ./fourier_som <csv_file> <grid_size> <iterations> <initial_eta> <initial_sigma> <is_toroidal> <calculate_labels>
```
- <csv_file>: Path to input CSV. Assumes first row is header, detects labels if a column is non-numeric.
- <grid_size>: Integer for square grid (e.g. 5 for 5x5).
-  <iterations: Training epochs.
- <initial_eta>: Starting learning rate.
- <initial_sigma>: Starting neighborhood radius..
- <is_toroidal>: 0 (rectangular) or 1 (toroidal/wrap-around).
- <calculate_labels>: 0 (skip) or 1 (compute label grid if labels present).
