import numpy as np


def az_za_to_ra_dec(
    az_vals,  # Units degrees
    za_vals,  # Units degrees
    ra_zenith,  # Units hours
    dec_zenith,  # Units degrees
):
    ra_vals = (
        ra_zenith
        - np.arctan2(
            np.sin(np.deg2rad(dec_zenith))
            * np.sin(np.deg2rad(za_vals))
            * np.sin(np.deg2rad(az_vals))
            - np.cos(np.deg2rad(dec_zenith)) * np.cos(np.deg2rad(za_vals)),
            np.sin(np.deg2rad(za_vals) * np.cos(np.deg2rad(az_vals))),
        )
        / np.pi
        * 12
        - 6
    )  # Units hours
    dec_vals = 90 - np.rad2deg(
        np.arccos(
            np.cos(np.deg2rad(dec_zenith))
            * np.sin(np.deg2rad(za_vals))
            * np.sin(np.deg2rad(az_vals))
            + np.sin(np.deg2rad(dec_zenith)) * np.cos(np.deg2rad(za_vals))
        )
    )  # Units degrees
    return ra_vals, dec_vals


def ra_dec_to_l_m(
    source_dec_rad=None,
    source_ra_offset_rad=None,  # Difference between the source RA and zenith RA
    zenith_dec_rad=None,
    time_offset_s=0,
    omega=7.27e-5,
):
    l = -np.cos(source_dec_rad) * np.sin(source_ra_offset_rad - time_offset_s * omega)
    m = np.cos(zenith_dec_rad) * np.sin(source_dec_rad) - np.sin(
        zenith_dec_rad
    ) * np.cos(source_dec_rad) * np.cos(source_ra_offset_rad - time_offset_s * omega)
    return l, m


def az_za_to_l_m(
    az_vals,  # Units degrees
    za_vals,  # Units degrees
):
    za_sin = np.sin(np.deg2rad(za_vals))
    l = za_sin * np.cos(np.deg2rad(az_vals))
    m = za_sin * np.sin(np.deg2rad(az_vals))
    return l, m


def l_m_to_az_za(
    l_vals,
    m_vals,
):
    za = np.arccos(np.sqrt(1 - l_vals**2 - m_vals**2))
    az = np.arctan2(m_vals, l_vals)
    return az, za
