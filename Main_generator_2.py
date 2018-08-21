from __future__ import division, print_function

import numpy as np
import tupak
import gwpy
from gwpy.plotter import TimeSeriesPlot
from tupak.core.utils import logger
from tupak.core import utils
import matplotlib.pyplot as plt

duration = 4.
sampling_frequency = 2048.
start_time = 0
dampener = 0.4
delta_t_echo = 0.2

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0, tilt_2=0, phi_12=1.7,
    phi_jl=0.3, luminosity_distance=250., iota=0.4, psi=2.659, phase=1.3,
    geocent_time=1126259642.413, ra=1.375, dec=-1.2108, t_echo= 0.4, delta_t_echo = 0.2, dampener = 0.4)

def lal_binary_black_hole_echo(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, iota, phase, ra, dec, geocent_time, psi,
        t_echo, **kwargs):

    signal = tupak.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, a_1=a_1, tilt_1=tilt_1,
        phi_12=phi_12, a_2=a_2, tilt_2=tilt_2, phi_jl=phi_jl, iota=iota,
        phase=phase, ra=ra, dec=dec, geocent_time=geocent_time, psi=psi,
        **kwargs)

    if signal is None:
        return None

    else:

        for i in range(10):

            dt = t_echo + i*delta_t_echo

            signal['plus'] += (dampener**i) * (signal['plus'] * np.exp(-1j * 2 * np.pi * dt * frequency_array))
            signal['cross'] += (dampener**i) * (signal['cross'] * np.exp(-1j * 2 * np.pi * dt * frequency_array))

        return signal

label = 'echo'
outdir = 'outdir'

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50.)

waveform_generator = tupak.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole_echo,
    parameters=injection_parameters, waveform_arguments=waveform_arguments)

hf_signal = waveform_generator.frequency_domain_strain()

'''
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, outdir=outdir)
    for name in ['H1', 'L1']]
'''

H1 = tupak.gw.detector.get_empty_interferometer('H1')
H1.set_strain_data_from_power_spectral_density(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)
H1.inject_signal(waveform_generator=waveform_generator,
                 parameters=injection_parameters)

time_domain_data = H1.strain_data.time_domain_strain
timeseries = gwpy.timeseries.TimeSeries(time_domain_data, t0=start_time,
                                        sample_rate=sampling_frequency)


hdata = gwpy.timeseries.TimeSeries(time_domain_data, t0=start_time,
                                        sample_rate=sampling_frequency)

from gwpy.signal import filter_design
bp = filter_design.bandpass(50, 250, 2048)

notches = [filter_design.notch(line, 2048) for
           line in (60, 120, 180)]

zpk = filter_design.concatenate_zpks(bp, *notches)

hfilt = hdata.filter(zpk, filtfilt=True)

hdata = hdata.crop(*hdata.span.contract(1))
hfilt = hfilt.crop(*hfilt.span.contract(1))

from gwpy.plotter import TimeSeriesPlot
plot = TimeSeriesPlot(hdata, hfilt, figsize=[12, 8], sep=True, sharex=True,
                      color='gwpy:ligo-hanford')
ax1, ax2 = plot.axes
ax1.set_title('Strain Data')
ax1.text(1.0, 1.0, 'Unfiltered data', transform=ax1.transAxes, ha='right')
ax1.set_ylabel('Amplitude [strain]', y=-0.2)
ax2.set_ylabel('')
ax2.text(1.0, 1.0, '50-250\,Hz bandpass, notches at 60, 120, 180 Hz',
         transform=ax2.transAxes, ha='right')
plot.show()


priors = tupak.gw.prior.BBHPriorSet()

priors['luminosity_distance'] = tupak.core.prior.Uniform(
    minimum=injection_parameters['luminosity_distance'] - 100,
    maximum=injection_parameters['luminosity_distance'] + 100,
    name='luminosity_distance', latex_label='$t_c$')

priors['t_echo'] = tupak.core.prior.Uniform(
    minimum=injection_parameters['t_echo'] - 2,
    maximum=injection_parameters['t_echo'] + 2,
    name='t_echo', latex_label='$t_e$')

priors['delta_t_echo'] = tupak.core.prior.Uniform(
    minimum=injection_parameters['delta_t_echo'] - 2,
    maximum=injection_parameters['delta_t_echo'] + 2,
    name='delta_t_echo', latex_label='$t_e$')

priors['dampener'] = tupak.core.prior.Uniform(
    minimum=injection_parameters['dampener'] - 0.5,
    maximum=injection_parameters['dampener'] + 0.5,
    name='dampener', latex_label='$t_e$')

for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
            'dec', 'geocent_time', 'phase', 'mass_1', 'mass_2', 'iota', 'luminosity_distance', 't_echo',
            'delta_t_echo']:
    priors[key] = injection_parameters[key]

likelihood = tupak.GravitationalWaveTransient(
    interferometers=[H1], waveform_generator=waveform_generator, prior=priors)

result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=250,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

result.plot_corner()
