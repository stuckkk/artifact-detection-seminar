In dieser Datei werden die Features von mir eingestuft, nämlich in Bezug darauf, wie sehr sich positive von negativen Segmenten unterscheiden.

- Gut
    - `line_length`
    - `ptp_amp`
    - `quantile`
    - `rms`
    - `std`
    - `variance`

- Mittel
    - `higuchi_fd`
    - `hurst_exp`
    - `kurtosis`
    - `mean`
    - `pow_freq_bands`
    - `spect_entropy`
    - `svd_entropy`
    - `teager_kaiser_energy` (ggf. ohne `range` ähnliche Spikes wie `wavelet_coef_energy`?)
    - `wavelet_coef_energy`
    - `zero_crossings`

- Schlecht
    - `app_entropy`
    - `decorr_time`
    - `skewness`