import numpy as np
import coordinate_transforms
import scipy


def max_fractional_freq_decorr(
    freq_resolution_hz,
    bl_length_m,
    zenith_angle_deg,
    c=3e8,
):

    l = np.sin(np.deg2rad(zenith_angle_deg))
    return 1 - np.abs(
        np.sinc(freq_resolution_hz / c * l * bl_length_m)
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)


def freq_decorr(
    freq_resolution_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    c=3e8,
):

    l, m = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=np.deg2rad(source_dec_deg),
        source_ra_offset_rad=source_ra_offset_hr / 12 * np.pi,
        zenith_dec_rad=np.deg2rad(telescope_lat_deg),
    )
    return np.sinc(
        freq_resolution_hz / c * (l * bl_ew_extent_m + m * bl_ns_extent_m)
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)


def fractional_freq_decorr(
    freq_resolution_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    telescope_lat_deg=39.25,
    c=3e8,
):
    return 1 - np.abs(
        freq_decorr(
            freq_resolution_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,
            source_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            c=c,
        )
    )


def time_decorr(
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
    return np.sinc(sinc_arg)


def fractional_time_decorr(
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

    return 1 - np.abs(
        time_decorr(
            time_resolution_s,
            freq_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,
            source_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            omega=omega,
            c=c,
        )
    )


def time_decorr_with_continuous_phase_tracking_to_zenith(
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
    return np.sinc(sinc_arg)


def fractional_time_decorr_with_continuous_phase_tracking_to_zenith(
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
    decorr_factor = time_decorr_with_continuous_phase_tracking_to_zenith(
        time_resolution_s,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,  # Difference between zenith and source RAs
        source_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_decorr_with_continuous_phase_tracking(
    time_resolution_s,
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

    sinc_arg = (
        time_resolution_s
        * omega
        * (
            u * np.cos(source_dec_rad) * np.cos(source_ra_offset_rad)
            - u * np.cos(phase_center_dec_rad) * np.cos(phase_center_ra_offset_rad)
            - v
            * np.sin(zenith_dec_rad)
            * np.cos(source_dec_rad)
            * np.sin(source_ra_offset_rad)
            + v
            * np.sin(zenith_dec_rad)
            * np.cos(phase_center_dec_rad)
            * np.sin(source_ra_offset_rad)
        )
    )  # Note that the numpy sinc function is \sin(\pi x)/(\pi x)
    return np.sinc(sinc_arg)


def fractional_time_decorr_with_continuous_phase_tracking(
    time_resolution_s,
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

    decorr_factor = time_decorr_with_continuous_phase_tracking(
        time_resolution_s,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,  # Difference between zenith and source RAs
        source_dec_deg,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_decorr_with_discrete_phase_tracking_to_zenith(
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
    decorr = (1 / float(n_time_steps)) * exp_term * np.sinc(sinc_arg)

    return decorr


def fractional_time_decorr_with_discrete_phase_tracking_to_zenith(
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
    decorr_factor = time_decorr_with_discrete_phase_tracking_to_zenith(
        total_time_interval_s,
        n_time_steps,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,
        source_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_decorr_with_discrete_phase_tracking(
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
    decorr = (1 / float(n_time_steps)) * exp_term * np.sinc(sinc_arg)

    return decorr


def fractional_time_decorr_with_discrete_phase_tracking(
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
    decorr_factor = time_decorr_with_discrete_phase_tracking(
        total_time_interval_s,
        n_time_steps,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,
        source_dec_deg,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_and_freq_decorr(
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
    l_t0, m_t0 = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=source_dec_rad,
        source_ra_offset_rad=source_ra_offset_rad,
        zenith_dec_rad=zenith_dec_rad,
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
    decorr_visibility_normalized = special_funcs / (
        2 * np.pi * freq_resolution_hz / freq_hz * (end_prod - start_prod)
    )
    decorr_factor = (
        np.exp(-2 * np.pi * 1j * (l_t0 * u + m_t0 * v)) * decorr_visibility_normalized
    )  # divide out the instantaneous visibility value
    return decorr_factor


def fractional_time_and_freq_decorr(
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
    decorr_factor = time_and_freq_decorr(
        time_resolution_s,
        freq_resolution_hz,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,
        source_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_and_freq_decorr_with_continuous_phase_tracking(
    time_resolution_s,
    freq_resolution_hz,
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

    l_t0, m_t0 = coordinate_transforms.ra_dec_to_l_m(
        source_dec_rad=source_dec_rad,
        source_ra_offset_rad=source_ra_offset_rad,
        zenith_dec_rad=zenith_dec_rad,
    )

    a = l_t0 * u + m_t0 * v

    b = (
        omega * np.cos(source_dec_rad) * np.cos(source_ra_offset_rad) * u
        - omega * np.cos(phase_center_dec_rad) * np.cos(phase_center_ra_offset_rad) * u
        - omega
        * np.sin(zenith_dec_rad)
        * np.cos(source_dec_rad)
        * np.sin(source_ra_offset_rad)
        * v
        + omega
        * np.sin(zenith_dec_rad)
        * np.cos(phase_center_dec_rad)
        * np.sin(phase_center_ra_offset_rad)
        * v
    )

    sin_func_1, cos_func_1 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * (a + time_resolution_s * b / 2)
        + 1j * 0
    )
    sin_func_2, cos_func_2 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz + 2) * (a - time_resolution_s * b / 2)
        + 1j * 0
    )
    sin_func_3, cos_func_3 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * (a + time_resolution_s * b / 2)
        + 1j * 0
    )
    sin_func_4, cos_func_4 = scipy.special.sici(
        np.pi * (freq_resolution_hz / freq_hz - 2) * (a - time_resolution_s * b / 2)
        + 1j * 0
    )

    special_funcs = (
        +sin_func_1
        - sin_func_2
        + sin_func_3
        - sin_func_4
        + 1j * (-cos_func_1 + cos_func_2 + cos_func_3 - cos_func_4)
    )

    decorr_visibility_normalized = special_funcs / (
        2 * np.pi * time_resolution_s * freq_resolution_hz / freq_hz * b
    )
    decorr_factor = (
        np.exp(-2 * np.pi * 1j * (l_t0 * u + m_t0 * v)) * decorr_visibility_normalized
    )  # divide out the instantaneous visibility value

    return decorr_factor


def fractional_time_and_freq_decorr_with_continuous_phase_tracking(
    time_resolution_s,
    freq_resolution_hz,
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
    decorr_factor = time_and_freq_decorr_with_continuous_phase_tracking(
        time_resolution_s,
        freq_resolution_hz,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,  # Difference between zenith and source RAs
        source_dec_deg,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def time_and_freq_decorr_with_discrete_phase_tracking(
    total_time_interval_s,
    n_time_steps,
    freq_resolution_hz,
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
    special_funcs = 0 + 0 * 1j
    for n in n_array:
        a = (
            l_t0 * u
            + n
            * phase_tracking_interval_s
            * omega
            * np.cos(source_dec_rad)
            * np.cos(source_ra_offset_rad)
            * u
            - n
            * phase_tracking_interval_s
            * omega
            * np.cos(phase_center_dec_rad)
            * np.cos(phase_center_ra_offset_rad)
            * u
            + m_t0 * v
            - n
            * phase_tracking_interval_s
            * omega
            * np.sin(zenith_dec_rad)
            * np.cos(source_dec_rad)
            * np.sin(source_ra_offset_rad)
            * v
            + n
            * phase_tracking_interval_s
            * omega
            * np.sin(zenith_dec_rad)
            * np.cos(phase_center_dec_rad)
            * np.sin(phase_center_ra_offset_rad)
            * v
        )

        sin_func_1, cos_func_1 = scipy.special.sici(
            np.pi
            * (freq_resolution_hz / freq_hz + 2)
            * (a + phase_tracking_interval_s * b / 2)
            + 1j * 0
        )
        sin_func_2, cos_func_2 = scipy.special.sici(
            np.pi
            * (freq_resolution_hz / freq_hz + 2)
            * (a - phase_tracking_interval_s * b / 2)
            + 1j * 0
        )
        sin_func_3, cos_func_3 = scipy.special.sici(
            np.pi
            * (freq_resolution_hz / freq_hz - 2)
            * (a + phase_tracking_interval_s * b / 2)
            + 1j * 0
        )
        sin_func_4, cos_func_4 = scipy.special.sici(
            np.pi
            * (freq_resolution_hz / freq_hz - 2)
            * (a - phase_tracking_interval_s * b / 2)
            + 1j * 0
        )

        special_funcs += (
            +sin_func_1
            - sin_func_2
            + sin_func_3
            - sin_func_4
            + 1j * (-cos_func_1 + cos_func_2 + cos_func_3 - cos_func_4)
        )
    decorr_visibility_normalized = special_funcs / (
        2 * np.pi * total_time_interval_s * freq_resolution_hz / freq_hz * b
    )
    decorr_factor = (
        np.exp(-2 * np.pi * 1j * (l_t0 * u + m_t0 * v)) * decorr_visibility_normalized
    )  # divide out the instantaneous visibility value

    return decorr_factor


def fractional_time_and_freq_decorr_with_discrete_phase_tracking(
    total_time_interval_s,
    n_time_steps,
    freq_resolution_hz,
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
    decorr_factor = time_and_freq_decorr_with_discrete_phase_tracking(
        total_time_interval_s,
        n_time_steps,
        freq_resolution_hz,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,  # Difference between zenith and source RAs
        source_dec_deg,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)


def decorr_general(
    total_time_interval_s,
    n_time_steps,  # Integer. If 1, assumes no phase tracking; if infinity assumes continuous phase tracking.
    freq_resolution_hz,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    phase_center_ra_offset_hr=0,  # Used only if n_time_steps > 1
    phase_center_dec_deg=39.25,  # Used only if n_time_steps > 1
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):

    if total_time_interval_s == 0:  # Frequency decorrelation only
        decorr_factor = freq_decorr(
            freq_resolution_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,  # Difference between zenith and source RAs
            source_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            c=c,
        )
        return decorr_factor

    if freq_resolution_hz == 0:  # Time decorrelation only
        if n_time_steps == 1:
            decorr_factor = time_decorr(
                total_time_interval_s,
                freq_hz,
                bl_ew_extent_m,
                bl_ns_extent_m,
                source_ra_offset_hr,
                source_dec_deg,
                telescope_lat_deg=telescope_lat_deg,
                omega=omega,
                c=c,
            )
            return decorr_factor
        elif n_time_steps == np.inf:
            decorr_factor = time_decorr_with_continuous_phase_tracking(
                total_time_interval_s,
                freq_hz,
                bl_ew_extent_m,
                bl_ns_extent_m,
                source_ra_offset_hr,
                source_dec_deg,
                phase_center_ra_offset_hr=phase_center_ra_offset_hr,
                phase_center_dec_deg=phase_center_dec_deg,
                telescope_lat_deg=telescope_lat_deg,
                omega=omega,
                c=c,
            )
            return decorr_factor
        else:
            decorr_factor = time_decorr_with_discrete_phase_tracking(
                total_time_interval_s,
                n_time_steps,
                freq_hz,
                bl_ew_extent_m,
                bl_ns_extent_m,
                source_ra_offset_hr,
                source_dec_deg,
                phase_center_ra_offset_hr=phase_center_ra_offset_hr,
                phase_center_dec_deg=phase_center_dec_deg,
                telescope_lat_deg=telescope_lat_deg,
                omega=omega,
                c=c,
            )
            return decorr_factor

    # Both time and frequency decorrelation
    if n_time_steps == 1:
        decorr_factor = time_and_freq_decorr(
            total_time_interval_s,
            freq_resolution_hz,
            freq_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,
            source_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            c=c,
        )
        return decorr_factor
    elif n_time_steps == np.inf:
        decorr_factor = time_and_freq_decorr_with_continuous_phase_tracking(
            total_time_interval_s,
            freq_resolution_hz,
            freq_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,
            source_dec_deg,
            phase_center_ra_offset_hr=phase_center_ra_offset_hr,
            phase_center_dec_deg=phase_center_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            omega=omega,
            c=c,
        )
        return decorr_factor
    else:
        decorr_factor = time_and_freq_decorr_with_discrete_phase_tracking(
            total_time_interval_s,
            n_time_steps,
            freq_resolution_hz,
            freq_hz,
            bl_ew_extent_m,
            bl_ns_extent_m,
            source_ra_offset_hr,
            source_dec_deg,
            phase_center_ra_offset_hr=phase_center_ra_offset_hr,
            phase_center_dec_deg=phase_center_dec_deg,
            telescope_lat_deg=telescope_lat_deg,
            omega=omega,
            c=c,
        )
        return decorr_factor


def fractional_decorr_general(
    total_time_interval_s,
    n_time_steps,  # Integer. If 1, assumes no phase tracking; if infinity assumes continuous phase tracking.
    freq_resolution_hz,
    freq_hz,
    bl_ew_extent_m,
    bl_ns_extent_m,
    source_ra_offset_hr,  # Difference between zenith and source RAs
    source_dec_deg,
    phase_center_ra_offset_hr=0,  # Used only if n_time_steps > 1
    phase_center_dec_deg=39.25,  # Used only if n_time_steps > 1
    telescope_lat_deg=39.25,
    omega=7.27e-5,
    c=3e8,
):
    decorr_factor = decorr_general(
        total_time_interval_s,
        n_time_steps,
        freq_resolution_hz,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        source_ra_offset_hr,
        source_dec_deg,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat_deg,
        omega=omega,
        c=c,
    )
    return 1 - np.abs(decorr_factor)
