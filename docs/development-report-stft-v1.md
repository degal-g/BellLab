# Development Report — STFT v1

Initial suite: 63 passed. Final suite: 66 passed. Corrections: peak report count updated; default peak
prominence is now `None`; candidate count now counts raw local maxima before
filters; Spectrum documentation reflects implemented FFT.

STFT adds `STFTSettings`, `TimeFrequencySpectrum`, `STFTResults`, and
`analyze_stft`. Matrix orientation is `values[frequency_index, time_index]`;
times are window centers. It uses the stationary FFT's unilateral coherent-gain
amplitude convention, supports linear amplitude/dBFS, explicit channel select
or mean, rectangular/Hann windows, optional end zero padding, frequency limits,
and hop length.

Padding is explicit (`pad_end`), with added samples reported in parameters and
diagnostics. No tracking, modal analysis, GUI, PSD, or STFT peak association was
implemented. Tests cover centered sine amplitude, chirp evolution, padding, and
dBFS silence. Real recordings remain required to validate window/hop choices,
noise behavior, and edge effects.
