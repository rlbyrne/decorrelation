import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display, HTML
import coordinate_transforms
import decorr_calculations


def polar_contour_plot(
    plot_vals,
    az_vals,  # Units radians
    za_vals,  # Units degrees
    vmin=-1,
    vmax=1,
    ncontours=100,
    title="",
    show=True,
    mark_north_pole=True,  # Create a label for the North Pole
    telescope_lat_deg=39.25,  # Used if mark_north_pole is True
):
    # use_cmap = matplotlib.cm.get_cmap("inferno").copy()
    use_cmap = matplotlib.colormaps["inferno"]
    use_cmap.set_bad(color="whitesmoke")

    # Set contour levels
    levels = np.linspace(vmin, vmax, num=ncontours)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw=dict(projection="polar"),
        figsize=(9, 6),
    )
    contourplot = ax.contourf(
        az_vals,
        za_vals,
        plot_vals,
        levels,
        vmin=vmin,
        vmax=vmax,
        cmap=use_cmap,
    )
    if mark_north_pole:
        ax.plot([np.pi / 2], [90 - telescope_lat_deg], "x", color="white", linewidth=0)
        ax.annotate(
            "North Pole",
            xy=(np.pi / 2, 90 - telescope_lat_deg),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="white",
            # bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3),
            # arrowprops=dict(
            #    arrowstyle="->", connectionstyle="arc3,rad=0.5", color="red"
            # ),
        )
    contourplot.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(contourplot, ax=ax, label="Decorrelation %")
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()


def plot_decorrelation(
    time_resolution_s,
    n_time_steps,
    freq_hz,
    freq_resolution_kHz,
    bl_length,
    bl_orientation,
    title="",
    show=False,
    phase_center_ra_offset_hr=0,
    phase_center_dec_deg=39.25,
    telescope_lat=39.25,
    mark_north_pole=True,
):
    bl_ew_extent_m = bl_length * np.cos(np.radians(bl_orientation))
    bl_ns_extent_m = bl_length * np.sin(np.radians(bl_orientation))

    za_vals = np.arange(0, 90, 1) + 0.5
    az_vals = np.arange(0, 370, 1) + 0.5
    az_array, za_array = np.meshgrid(az_vals, za_vals)
    ra_vals, dec_vals = coordinate_transforms.az_za_to_ra_dec(
        az_array,  # Units degrees
        za_array,  # Units degrees
        0,  # Units hours (RA of zenith)
        telescope_lat,  # Units degrees
    )

    decorr_values = decorr_calculations.fractional_decorr_general(
        time_resolution_s,
        n_time_steps,
        freq_resolution_kHz * 1e3,
        freq_hz,
        bl_ew_extent_m,
        bl_ns_extent_m,
        ra_vals,
        dec_vals,
        phase_center_ra_offset_hr=phase_center_ra_offset_hr,
        phase_center_dec_deg=phase_center_dec_deg,
        telescope_lat_deg=telescope_lat,
    )

    polar_contour_plot(
        decorr_values * 100,  # Convert to percentage
        np.deg2rad(az_vals),  # Units radians
        za_vals,  # Units degrees
        vmin=0,
        vmax=100,
        ncontours=100,
        title=title,
        show=show,
        mark_north_pole=mark_north_pole,
    )


def plot_decorrelation_interactive(
    time_resolution_s,
    n_time_steps,
    freq_resolution_kHz,
    bl_length,
    bl_orientation,
):

    return plot_decorrelation(
        time_resolution_s,
        n_time_steps,
        250e6,
        freq_resolution_kHz,
        bl_length,
        bl_orientation,
        title="",
        show=False,
        phase_center_ra_offset_hr=0,
        phase_center_dec_deg=39.25,
        telescope_lat=39.25,
        mark_north_pole=True,
    )


def create_interactive_plot():

    # Create sliders for interactive control
    int_time_slider = widgets.FloatSlider(
        value=1.5,  # Initial value
        min=0,  # Minimum value
        max=10,  # Maximum value
        step=0.01,  # Step size
        description="Int. Time (s):",
        continuous_update=False,
        layout=widgets.Layout(width="auto", min_width="10px"),
    )

    time_steps_slider = widgets.SelectionSlider(
        options=np.append(np.arange(1, 20), np.inf),
        value=1,  # Initial value
        description="Phase Tracking Steps",
        continuous_update=False,
        layout=widgets.Layout(width="auto", min_width="10px"),
    )

    freq_resolution_kHz_slider = widgets.IntSlider(
        value=130,  # Initial value
        min=0,  # Minimum value
        max=200,  # Maximum value
        step=10,  # Step size
        description="Freq. Res. (kHz)",
        continuous_update=False,
        layout=widgets.Layout(width="auto", min_width="10px"),
    )

    bl_length_slider = widgets.IntSlider(
        value=15000,  # Initial value
        min=0,  # Minimum value
        max=20000,  # Maximum value
        step=10,  # Step size
        description="BL length (m):",
        continuous_update=False,
        layout=widgets.Layout(width="auto", min_width="10px"),
    )

    bl_orientation_slider = widgets.IntSlider(
        value=0,  # Initial value
        min=0,  # Minimum value
        max=180,  # Maximum value
        step=1,  # Step size
        description="BL orientation (deg.):",
        continuous_update=False,
        layout=widgets.Layout(width="auto", min_width="10px"),
    )

    # Inject CSS to ensure label has minimum width
    display(
        HTML(
            """
    <style>
    .widget-label {
        min-width: 200px !important; /* Adjust this value as needed */
        display: flex; /* Ensure label takes full available width */
        align-items: center; /* Vertically align label text */
    }
    .widget-box.hbox > .widget-label {
        flex: 0 0 auto; /* Prevent label from stretching in hbox */
        padding-right: 10px; /* Add some spacing between label and slider */
    }
    </style>
    """
        )
    )

    interactive_plot_func = interactive(
        plot_decorrelation_interactive,
        time_resolution_s=int_time_slider,
        n_time_steps=time_steps_slider,
        freq_resolution_kHz=freq_resolution_kHz_slider,
        bl_length=bl_length_slider,
        bl_orientation=bl_orientation_slider,
    )

    display(interactive_plot_func)
