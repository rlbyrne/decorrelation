import numpy as np
import coordinate_transforms
import scipy


def freq_decorrelation(
    freq_resolution_hz,
    bl_length_m,
    source_za=np.pi / 2,
    c=3e8,
):
    return 1 - np.abs(
        np.sinc(freq_resolution_hz * bl_length_m / c * np.sin(source_za))
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)


def time_decorrelation(
    time_resolution_s,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)
    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c
    sinc_arg = (
        time_resolution_s
        * omega
        * np.cos(source_dec_rad)
        * (
            u * np.cos(source_ra_offset_rad)
            - v * np.sin(zenith_dec_rad) * np.sin(source_ra_offset_rad)
        )
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)
    return 1 - np.abs(np.sinc(sinc_arg))


def time_decorrelation_with_continuous_phase_tracking(
    time_resolution_s,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)
    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c
    sinc_arg = (
        time_resolution_s
        * omega
        * (
            u * np.cos(source_dec_rad) * np.cos(source_ra_offset_rad)
            - u * np.cos(zenith_dec_rad)
            - v
            * np.cos(source_dec_rad)
            * np.sin(zenith_dec_rad)
            * np.sin(source_ra_offset_rad)
        )
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)
    return 1 - np.abs(np.sinc(sinc_arg))


def time_decorrelation_with_discrete_phase_tracking_to_zenith(
    total_time_interval_s,
    n_time_steps,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)
    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c

    phase_tracking_interval_s = total_time_interval_s / n_time_steps
    sinc_arg = (
        phase_tracking_interval_s
        * omega
        * np.cos(source_dec_rad)
        * (
            np.cos(source_ra_offset_rad) * u
            - np.sin(zenith_dec_rad) * np.sin(source_ra_offset_rad) * v
        )
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)

    n_array = np.arange(
        -(float(n_time_steps) - 1) / 2, (float(n_time_steps) - 1) / 2 + 1, 1
    )

    exp_term = 0
    for n in n_array:
        exp_arg = (
            n
            * phase_tracking_interval_s
            * omega
            * (
                np.cos(source_dec_rad) * np.cos(source_ra_offset_rad) * u
                - np.cos(zenith_dec_rad) * u
                - np.cos(source_dec_rad)
                * np.sin(zenith_dec_rad)
                * np.sin(source_ra_offset_rad)
                * v
            )
        )
        exp_term += np.real(np.exp(2 * np.pi * 1j * exp_arg))
    decorr = 1 - np.abs((1 / float(n_time_steps)) * exp_term * np.sinc(sinc_arg))

    return decorr


def time_decorrelation_with_discrete_phase_tracking(
    total_time_interval_s,
    n_time_steps,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    phase_center_ra_offset_hr=0,
    phase_center_dec_deg=39.25,
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    phase_center_dec_rad = np.deg2rad(phase_center_dec_deg)
    phase_center_ra_offset_rad = phase_center_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)
    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c

    phase_tracking_interval_s = total_time_interval_s / n_time_steps
    sinc_arg = (
        phase_tracking_interval_s
        * omega
        * np.cos(source_dec_rad)
        * (
            np.cos(source_ra_offset_rad) * u
            - np.sin(zenith_dec_rad) * np.sin(source_ra_offset_rad) * v
        )
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)

    n_array = np.arange(
        -(float(n_time_steps) - 1) / 2, (float(n_time_steps) - 1) / 2 + 1, 1
    )

    exp_term = 0
    for n in n_array:
        exp_arg = (
            n
            * phase_tracking_interval_s
            * omega
            * (
                -np.cos(phase_center_dec_rad) * np.cos(phase_center_ra_offset_rad) * u
                + np.cos(phase_center_dec_rad)
                * np.sin(zenith_dec_rad)
                * np.sin(phase_center_ra_offset_rad)
                * v
                + np.cos(source_dec_rad) * np.cos(source_ra_offset_rad) * u
                - np.cos(source_dec_rad)
                * np.sin(zenith_dec_rad)
                * np.sin(source_ra_offset_rad)
                * v
            )
        )
        exp_term += np.real(np.exp(2 * np.pi * 1j * exp_arg))
    decorr = 1 - np.abs((1 / float(n_time_steps)) * exp_term * np.sinc(sinc_arg))

    return decorr


def time_and_freq_decorrelation(
    time_resolution_s,
    freq_resolution_hz,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)

    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c

    l_start, m_start = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=source_dec_rad,
        source_ra_offset_rad=source_ra_offset_rad,
        zenith_dec_rad=zenith_dec_rad,
        time_offset_s=-time_resolution_s / 2,
    )
    l_end, m_end = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=source_dec_rad,
        source_ra_offset_rad=source_ra_offset_rad,
        zenith_dec_rad=zenith_dec_rad,
        time_offset_s=time_resolution_s / 2,
    )

    start_prod = l_start * u + m_start * v
    end_prod = l_end * u + m_end * v

    sin_func_1, cos_func_1 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * start_prod + 1j * 0
    )
    sin_func_2, cos_func_2 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * end_prod + 1j * 0
    )
    sin_func_3, cos_func_3 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * start_prod + 1j * 0
    )
    sin_func_4, cos_func_4 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * end_prod + 1j * 0
    )

    special_funcs = (
        -sin_func_1
        + sin_func_2
        - sin_func_3
        + sin_func_4
        + 1j * (cos_func_1 - cos_func_2 - cos_func_3 + cos_func_4)
    )

    return 1 - np.abs(
        special_funcs
        / (2 * np.pi * freq_resolution_hz / freq_hz * (end_prod - start_prod))
    )


def time_and_freq_decorrelation_with_discrete_phase_tracking(
    total_time_interval_s,
    n_time_steps,
    freq_resolution_hz,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    omega=7.27e-5,
    telescope_lat_deg=39.25,
    c=3e8,
):
    source_dec_rad = np.deg2rad(source_dec_deg)
    source_ra_offset_rad = source_ra_offset_hr / 12 * np.pi
    zenith_dec_rad = np.deg2rad(telescope_lat_deg)

    u = bl_ew_extent_m * freq_hz / c
    v = bl_ns_extent_m * freq_hz / c

    phase_tracking_interval_s = total_time_interval_s / n_time_steps

    l_t0, m_t0 = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=source_dec_rad,
        source_ra_offset_rad=source_ra_offset_rad,
        zenith_dec_rad=zenith_dec_rad,
    )

    b = (
        omega * np.cos(source_dec_rad) * np.cos(source_ra_offset_rad) * u
        + omega
        * np.sin(zenith_dec_rad)
        * np.cos(source_dec_rad)
        * np.sin(source_ra_offset_rad)
        * v
    )

    n_array = np.arange(
        -(float(n_time_steps) - 1) / 2, (float(n_time_steps) - 1) / 2 + 1, 1
    )
    for n in n_array:
        a = (
            l_t0 * u
            + n
            * phase_tracking_interval_s
            * omega
            * np.cos(source_dec_rad)
            * np.cos(source_ra_offset_rad)
            * u
            - n * phase_tracking_interval_s * omega * np.cos()
        )

    start_prod = l_start * u + m_start * v
    end_prod = l_end * u + m_end * v

    sin_func_1, cos_func_1 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * start_prod + 1j * 0
    )
    sin_func_2, cos_func_2 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * end_prod + 1j * 0
    )
    sin_func_3, cos_func_3 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * start_prod + 1j * 0
    )
    sin_func_4, cos_func_4 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * end_prod + 1j * 0
    )

    special_funcs = (
        -sin_func_1
        + sin_func_2
        - sin_func_3
        + sin_func_4
        + 1j * (cos_func_1 - cos_func_2 - cos_func_3 + cos_func_4)
    )

    return 1 - np.abs(
        special_funcs
        / (2 * np.pi * freq_resolution_hz / freq_hz * (end_prod - start_prod))
    )
