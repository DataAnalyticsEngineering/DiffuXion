import marimo

__generated_with = "0.16.5"
app = marimo.App(
    width="full",
    app_title="PolycrystalDiffusion",
    layout_file="layouts/run.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import h5py
    import json
    import subprocess
    from scipy.ndimage import map_coordinates
    from skimage import measure
    import trimesh
    import os

    return (
        h5py,
        json,
        map_coordinates,
        measure,
        mo,
        np,
        os,
        subprocess,
        trimesh,
    )


@app.cell(hide_code=True)
def _(os):
    # Make sure that data directory exists
    os.makedirs("./data", exist_ok=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Geometry""")
    return


@app.cell(hide_code=True)
def _(mo):
    n_grains_input = mo.ui.text(label="Number of grains:", value="27")
    n_grains_input
    return (n_grains_input,)


@app.cell(hide_code=True)
def _(mo, n_grains_input):
    _option_list = [
        "random",
        "llhs-lloyd",
        "halton",
        "sobol",
        "rubiks-cube",
        "honeycomb",
    ]

    n_grains = int(n_grains_input.value)

    if n_grains == 2:
        _option_list.append("diamond")

    tess_type_input = mo.ui.dropdown(
        label="Tessellation type:", options=_option_list, value="sobol"
    )
    tess_type_input
    return n_grains, tess_type_input


@app.cell(hide_code=True)
def _(mo, tess_type_input):
    seed_input = None
    if tess_type_input.value not in ["diamond", "rubiks-cube"]:
        seed_input = mo.ui.text(label="generator seed for tessellation", value="42")
        mo.output.replace(seed_input)
    return (seed_input,)


@app.cell(hide_code=True)
def _(mo):
    def_domain = mo.ui.dropdown(
        options=["...size of the simulation domain", "...average grain size"],
        label="Define...",
        value="...size of the simulation domain",
    )
    def_domain
    return (def_domain,)


@app.cell(hide_code=True)
def _(def_domain, mo):
    match def_domain.value:
        case "...size of the simulation domain":
            domain_size_input = mo.ui.text(label="Size of domain [nm]:", value="1,1,1")
        case "...average grain size":
            domain_size_input = mo.ui.text(
                label="Average grain size (assuming cube geometry) [nm]:", value="50"
            )

    domain_size_input
    return (domain_size_input,)


@app.cell(hide_code=True)
def _(def_domain, domain_size_input, n_grains, np):
    match def_domain.value:
        case "...size of the simulation domain":
            domain_size = [float(n) for n in domain_size_input.value.split(",")]
        case "...average grain size":
            domain_size = [
                float(domain_size_input.value) * (n_grains * np.pi / 6) ** (1 / 3)
                for n in range(3)
            ]
    return (domain_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Discretization""")
    return


@app.cell(hide_code=True)
def _(mo):
    resolution_input = mo.ui.text(label="Resolution in x,y,z:", value="64,64,64")
    resolution_input
    return (resolution_input,)


@app.cell(hide_code=True)
def _(domain_size, resolution_input):
    resolution = [int(n) for n in resolution_input.value.split(",")]
    voxel_length = domain_size[0] / resolution[0]
    return resolution, voxel_length


@app.cell(hide_code=True)
def _(mo):
    def_thickness = mo.ui.dropdown(
        options=["...directly", "...as multiple of voxel length"],
        label="Define GB thickness...",
        value="...as multiple of voxel length",
    )
    def_thickness
    return (def_thickness,)


@app.cell(hide_code=True)
def _(def_thickness, mo):
    match def_thickness.value:
        case "...directly":
            thickness_input = mo.ui.text(label="GB thickness [nm]", value="2.5")
        case "...as multiple of voxel length":
            thickness_input = mo.ui.text(label="Number of voxel layers:", value="4")
    thickness_input
    return (thickness_input,)


@app.cell(hide_code=True)
def _(def_thickness, thickness_input, voxel_length):
    match def_thickness.value:
        case "...directly":
            thickness = float(thickness_input.value)
        case "...as multiple of voxel length":
            thickness = int(thickness_input.value) * voxel_length
    return (thickness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate voxel representation of microstructure""")
    return


@app.cell(hide_code=True)
def _():
    from MSUtils.general.h52xdmf import write_xdmf
    from MSUtils.general.MicrostructureImage import MicrostructureImage
    from MSUtils.voronoi.VoronoiGBErosion import PeriodicVoronoiImageErosion
    from MSUtils.voronoi.VoronoiImage import PeriodicVoronoiImage
    from MSUtils.voronoi.VoronoiSeeds import VoronoiSeeds
    from MSUtils.voronoi.VoronoiTessellation import PeriodicVoronoiTessellation

    return (
        MicrostructureImage,
        PeriodicVoronoiImage,
        PeriodicVoronoiImageErosion,
        PeriodicVoronoiTessellation,
        VoronoiSeeds,
        write_xdmf,
    )


@app.cell(hide_code=True)
def _(mo):
    path_geo_input = mo.ui.text(label="Path for export:", value="data/voroImg_eroded")
    path_geo_input
    return (path_geo_input,)


@app.cell(hide_code=True)
def _(mo):
    dataset_name_input = mo.ui.text(label="Dataset name:", value="dset_0")
    dataset_name_input
    return (dataset_name_input,)


@app.cell(hide_code=True)
def _(dataset_name_input, path_geo_input, seed_input, tess_type_input):
    tess_type = tess_type_input.value
    if seed_input is not None:
        seed = int(seed_input.value)
    else:
        seed = 0

    path_geo = path_geo_input.value
    dataset_name = dataset_name_input.value
    return dataset_name, path_geo, seed, tess_type


@app.cell(hide_code=True)
def _(domain_size, thickness):
    # Rescale size of the simulation domain
    scale_length_x = domain_size[0]
    domain_size_scaled_xyz = [1 / scale_length_x * l for l in domain_size]
    domain_size_zyx = [domain_size[2], domain_size[1], domain_size[0]]
    domain_size_scaled_zyx = [
        domain_size_scaled_xyz[2],
        domain_size_scaled_xyz[1],
        domain_size_scaled_xyz[0],
    ]

    thickness_scaled = thickness / scale_length_x
    return domain_size_scaled_xyz, domain_size_zyx, thickness_scaled


@app.cell(hide_code=True)
def _(mo):
    run_msutils = mo.ui.run_button(label="Generate microstructure")
    run_msutils
    return (run_msutils,)


@app.cell(hide_code=True)
def _(
    PeriodicVoronoiImage,
    PeriodicVoronoiImageErosion,
    PeriodicVoronoiTessellation,
    VoronoiSeeds,
    dataset_name,
    domain_size_scaled_xyz,
    domain_size_zyx,
    mo,
    n_grains,
    path_geo,
    resolution,
    run_msutils,
    seed,
    tess_type,
    thickness_scaled,
    write_xdmf,
):
    ms_fn = f"{path_geo}.h5"
    ms_dsn = f"/{dataset_name}"
    if run_msutils.value:
        mo.output.replace(mo.md("Generating microstructure..."))
        permute_order = "zyx"

        with mo.capture_stdout() as _buffer:
            # Generate Voronoi seeds and tessellation
            SeedInfo = VoronoiSeeds(
                n_grains, domain_size_scaled_xyz, tess_type, BitGeneratorSeed=seed
            )
            voroTess = PeriodicVoronoiTessellation(
                domain_size_scaled_xyz, SeedInfo.seeds
            )

            # Generate Voronoi image
            voroImg = PeriodicVoronoiImage(
                resolution, SeedInfo.seeds, domain_size_scaled_xyz
            )

            # Generate Voronoi image with grain boundaries of a specific thickness
            voroErodedImg = PeriodicVoronoiImageErosion(
                voroImg, voroTess, interface_thickness=thickness_scaled
            )
            voroErodedImg.write_h5(
                ms_fn, ms_dsn, order=permute_order, save_normals=True
            )
            write_xdmf(
                f"{path_geo}.h5",
                f"{path_geo}.xdmf",
                microstructure_length=domain_size_zyx,
            )

        mo.output.append(mo.md("Done!"))
        mo.output.append(_buffer.getvalue())
    return ms_dsn, ms_fn, voroErodedImg


@app.cell(hide_code=True)
def _(
    MicrostructureImage,
    domain_size,
    domain_size_scaled_xyz,
    mo,
    n_grains,
    np,
    run_msutils,
    thickness,
    voroErodedImg,
):
    mo.output.replace(mo.md("__Microstructure properties:__"))
    mo.output.append(
        mo.md("_Click the button above first to generate a microstructure_")
    )
    if run_msutils.value:
        # Calculate and print volume fraction of all grain boundary (all tags >= n_grains)
        msimage = MicrostructureImage(
            image=voroErodedImg.eroded_image, L=domain_size_scaled_xyz
        )
        gb_volume_fraction = 0
        for phase, fraction in msimage.volume_fractions.items():
            if phase >= n_grains:
                gb_volume_fraction += fraction

        gb_volume_fraction_percent = gb_volume_fraction * 100

        avg_grain_size = (
            domain_size[0] * domain_size[1] * domain_size[2] / n_grains * 6 / np.pi
        ) ** (1 / 3)

        mo.output.replace_at_index(mo.md(f"Interface thickness: {thickness:.2f} nm"), 1)
        mo.output.append(
            mo.md(
                f"Size of simulation domain: [{domain_size[0]:.2f}, {domain_size[1]:.2f}, {domain_size[2]:.2f}] nm"
            )
        )
        mo.output.append(mo.md(f"Average grain size [nm]: {avg_grain_size:.2f}nm"))
        mo.output.append(
            mo.md(
                f"Volume fraction of all grain boundaries: {gb_volume_fraction_percent:.2f}%"
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Material parameters""")
    return


@app.cell(hide_code=True)
def _(mo):
    def_diff = mo.ui.dropdown(
        options=["...directly", "...via Arrhenius relation"],
        label=r"Define diffusion coefficients...",
        value="...directly",
    )
    def_diff
    return (def_diff,)


@app.cell(hide_code=True)
def _(def_diff, mo):
    apply_arrhenius = False
    match def_diff.value:
        case "...directly":
            diff_bulk_input = mo.ui.text(
                label=r"$D^{\mathrm{bulk}}~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.0",
            )
            diff_para_input = mo.ui.text(
                label=r"$D^{\mathrm{GB}}_{\Vert}~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.0",
            )
            diff_perp_input = mo.ui.text(
                label=r"$D^{\mathrm{GB}}_{\perp}~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.0",
            )
            _options = [diff_bulk_input, diff_para_input, diff_perp_input]
        case "...via Arrhenius relation":
            apply_arrhenius = True
            temperature = mo.ui.text(label=r"$T[K]$", value="300")
            E_a_bulk = mo.ui.text(
                label=r"$E^{\mathrm{bulk}}~ [\mathrm{meV}]$", value="500"
            )
            D_bulk = mo.ui.text(
                label=r"$D^{\mathrm{bulk}}_0~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.e5",
            )
            E_a_para = mo.ui.text(
                label=r"$E^{\mathrm{GB}}_\Vert~ [\mathrm{meV}]$", value="500"
            )
            D_para = mo.ui.text(
                label=r"$D^{\mathrm{GB}}_{0,\Vert}~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.e5",
            )
            E_a_perp = mo.ui.text(
                label=r"$E^{\mathrm{GB}}_\perp [\mathrm{meV}]$", value="500"
            )
            D_perp = mo.ui.text(
                label=r"$D^{\mathrm{GB}}_{0,\perp}~ [1.\mathrm{e-}7\mathrm{cm}^2/\mathrm{s}]$",
                value="1.e5",
            )
            _options = [
                mo.vstack([temperature, ""]),
                mo.vstack([E_a_bulk, D_bulk]),
                mo.vstack([E_a_para, D_para]),
                mo.vstack([E_a_perp, D_perp]),
            ]
    mo.hstack(_options)
    return (
        D_bulk,
        D_para,
        D_perp,
        E_a_bulk,
        E_a_para,
        E_a_perp,
        apply_arrhenius,
        diff_bulk_input,
        diff_para_input,
        diff_perp_input,
        temperature,
    )


@app.cell(hide_code=True)
def _(
    D_bulk,
    D_para,
    D_perp,
    E_a_bulk,
    E_a_para,
    E_a_perp,
    apply_arrhenius,
    diff_bulk_input,
    diff_para_input,
    diff_perp_input,
    np,
    temperature,
):
    def _apply_arrhenius(D_0, E_a, T):
        """Compute diffusion coefficient for given temperature T based on reference D_0 and activation energy E_a following the Arrhenius relation

        :param D_0: Reference diffusion coefficient (in 1.e-7 cm^2/s)
        :param E_a: Activation energy (in meV)
        :param T: Temperature (in K)
        :return: Resulting diffusion coefficient (in 1.e-7 cm^2/s)
        """

        # Boltzmann constant (in meV/K)
        k = 8.617333262e-2

        return D_0 * np.exp(-E_a / (k * T))

    if apply_arrhenius:
        _T = float(temperature.value)
        diff_bulk = _apply_arrhenius(float(D_bulk.value), float(E_a_bulk.value), _T)
        diff_para = _apply_arrhenius(float(D_para.value), float(E_a_para.value), _T)
        diff_perp = _apply_arrhenius(float(D_perp.value), float(E_a_perp.value), _T)
    else:
        diff_bulk = float(diff_bulk_input.value)
        diff_para = float(diff_para_input.value)
        diff_perp = float(diff_perp_input.value)
    return diff_bulk, diff_para, diff_perp


@app.cell(hide_code=True)
def _(diff_bulk, diff_para, diff_perp, mo):
    mo.output.replace(mo.md("__Material properties:__"))
    mo.output.append(
        mo.hstack(
            [
                mo.md(
                    rf"$D^{{\mathrm{{bulk}}}}$: {diff_bulk:.4f} $[1.\mathrm{{e-}}7\mathrm{{cm}}^2/\mathrm{{s}}]$"
                ),
                mo.md(
                    rf"$D^{{\mathrm{{GB}}}}_{{\Vert}}$: {diff_para:.4f} $[1.\mathrm{{e-}}7\mathrm{{cm}}^2/\mathrm{{s}}]$"
                ),
                mo.md(
                    rf"$D^{{\mathrm{{GB}}}}_{{\perp}}$: {diff_perp:.4f} $[1.\mathrm{{e-}}7\mathrm{{cm}}^2/\mathrm{{s}}]$"
                ),
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Simulation""")
    return


@app.cell(hide_code=True)
def _(mo):
    _options_fields = ["full-field results", "effective diffusivity"]
    sim_type_select = mo.ui.dropdown(
        label="Selection simulation type:", options=_options_fields
    )
    sim_type_select
    return (sim_type_select,)


@app.cell(hide_code=True)
def _(mo, sim_type_select):
    if sim_type_select.value == "full-field results":
        _options_fields = [
            "concentration field",
            "concentration fluctuation field",
            "concentration gradient field",
            "flux field",
        ]
        sim_output_select = mo.ui.multiselect(
            label="Select all fields to be computed:", options=_options_fields
        )
        mo.output.replace(sim_output_select)
    return (sim_output_select,)


@app.cell(hide_code=True)
def _(mo, sim_type_select):
    if sim_type_select.value == "full-field results":
        load_input = mo.ui.text(label="Loading in x,y,z:", value="1,0,0")
        mo.output.replace(load_input)
    return (load_input,)


@app.cell(hide_code=True)
def _(mo):
    result_prefix_input = mo.ui.text(label="Dataset name:", value="sample_results")
    result_prefix_input
    return (result_prefix_input,)


@app.cell(hide_code=True)
def _(mo):
    n_processes_input = mo.ui.text(label="Number of processes:", value="16")
    n_processes_input
    return (n_processes_input,)


@app.cell(hide_code=True)
def _(mo):
    input_fn_input = mo.ui.text(label="File name for FANS input:", value="data/input")
    input_fn_input
    return (input_fn_input,)


@app.cell(hide_code=True)
def _(
    diff_bulk,
    diff_para,
    diff_perp,
    domain_size,
    load_input,
    ms_dsn,
    ms_fn,
    result_prefix_input,
    sim_output_select,
    sim_type_select,
):
    # Generate input file for FANS simulation

    input_FANS = dict()
    input_FANS["microstructure"] = dict()
    input_FANS["microstructure"]["filepath"] = ms_fn
    input_FANS["microstructure"]["datasetname"] = f"{ms_dsn}/eroded_image"
    input_FANS["microstructure"]["L"] = domain_size

    input_FANS["problem_type"] = "thermal"
    input_FANS["matmodel"] = "GBDiffusion"
    input_FANS["material_properties"] = dict()
    input_FANS["material_properties"]["GB_unformity"] = True
    input_FANS["material_properties"]["D_bulk"] = diff_bulk
    input_FANS["material_properties"]["D_perp"] = diff_perp
    input_FANS["material_properties"]["D_par"] = diff_para

    input_FANS["method"] = "cg"
    input_FANS["error_parameters"] = dict()
    input_FANS["error_parameters"]["measure"] = "Linfinity"
    input_FANS["error_parameters"]["type"] = "absolute"
    input_FANS["error_parameters"]["tolerance"] = 1e-10

    input_FANS["n_it"] = 1000

    _loading = []
    _sim_output_list = []
    match sim_type_select.value:
        case "full-field results":
            _loading.append([[float(_n) for _n in load_input.value.split(",")]])
            if "concentration field" in sim_output_select.value:
                _sim_output_list.append("displacement")
            if "concentration fluctuation field" in sim_output_select.value:
                _sim_output_list.append("displacement_fluctuation")
            if "concentration gradient field" in sim_output_select.value:
                _sim_output_list.append("strain")
            if "flux field" in sim_output_select.value:
                _sim_output_list.append("stress")
        case "effective diffusivity":
            _loading.append([[1, 0, 0]])
            _sim_output_list.append("homogenized_tangent")

    input_FANS["macroscale_loading"] = _loading

    input_FANS["results_prefix"] = result_prefix_input.value
    input_FANS["results"] = _sim_output_list
    return (input_FANS,)


@app.cell(hide_code=True)
def _(mo):
    export_input = mo.ui.run_button(label="Export input file")
    export_input
    return (export_input,)


@app.cell(hide_code=True)
def _(export_input, input_FANS, input_fn_input, json):
    input_fn = f"{input_fn_input.value}.json"
    if export_input.value:
        with open(f"{input_fn}", "w", encoding="utf-8") as _f:
            json.dump(input_FANS, _f, ensure_ascii=False, indent=4)
    return (input_fn,)


@app.cell(hide_code=True)
def _(mo):
    result_fn_input = mo.ui.text(
        label="File name for FANS results:", value="data/results"
    )
    result_fn_input
    return (result_fn_input,)


@app.cell(hide_code=True)
def _(mo):
    run_fans = mo.ui.run_button(label="Run FANS simulation")
    run_fans
    return (run_fans,)


@app.cell(hide_code=True)
def _(input_fn, mo, n_processes_input, result_fn_input, run_fans, subprocess):
    result_fn = f"{result_fn_input.value}.h5"
    if run_fans.value:
        mo.output.replace(mo.md("Running FANS..."))
        with mo.capture_stdout() as _buffer:
            subprocess.run(
                f"mpiexec -n {n_processes_input.value} FANS {input_fn} {result_fn}",
                shell=True,
            )
            mo.output.append(mo.md("Done!"))
        mo.output.append(_buffer.getvalue())
    return (result_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results and visualization""")
    return


@app.cell(hide_code=True)
def _(
    h5py,
    mo,
    ms_dsn,
    np,
    os,
    result_fn,
    result_prefix_input,
    sim_type_select,
):
    visu_available = False
    if sim_type_select.value == "full-field results":
        generate_xdmf = mo.ui.run_button(label="Generate xdmf")
        visu_available = True
        mo.output.replace(generate_xdmf)
    elif sim_type_select.value == "effective diffusivity":
        generate_xdmf = mo.md("Run simulation first to display results here!")
        _homog_tangent_path = f"{ms_dsn}/eroded_image_results/{result_prefix_input.value}/load0/time_step0/homogenized_tangent"
        if os.path.isfile(result_fn):
            with h5py.File(result_fn, "r") as _f:
                if _homog_tangent_path in _f:
                    _homog_tangent = _f[_homog_tangent_path][:]
                    _homog_tangent_eigvals = np.linalg.eigvals(_homog_tangent)
                    _homog_tangent_eigvals.sort()
                    generate_xdmf = mo.vstack(
                        [
                            mo.md("__Effective diffusivity tensor:__"),
                            mo.md(
                                f"{_homog_tangent[0,0]:.4f} &nbsp;&nbsp; {_homog_tangent[0,1]:.4f} &nbsp;&nbsp; {_homog_tangent[0,2]:.4f}"
                            ),
                            mo.md(
                                f"{_homog_tangent[1,0]:.4f} &nbsp;&nbsp; {_homog_tangent[1,1]:.4f} &nbsp;&nbsp; {_homog_tangent[1,2]:.4f}"
                            ),
                            mo.md(
                                f"{_homog_tangent[2,0]:.4f} &nbsp;&nbsp; {_homog_tangent[2,1]:.4f} &nbsp;&nbsp; {_homog_tangent[2,2]:.4f}"
                            ),
                            mo.md(
                                f"_Eigenvalues:_ &nbsp;&nbsp; {_homog_tangent_eigvals[0]:.4f}, &nbsp;&nbsp; {_homog_tangent_eigvals[0]:.4f}, &nbsp;&nbsp; {_homog_tangent_eigvals[0]:.4f}"
                            ),
                        ]
                    )

        mo.output.replace(generate_xdmf)
    return generate_xdmf, visu_available


@app.cell(hide_code=True)
def _(generate_xdmf, os, result_fn, subprocess, visu_available):
    if visu_available:
        if generate_xdmf.value:
            if os.path.dirname(result_fn) in ["", "."]:
                subprocess.run(
                    f"python -m MSUtils.general.h52xdmf {os.path.basename(result_fn)}",
                    shell=True,
                )
            else:
                subprocess.run(
                    f"cd {os.path.dirname(result_fn)} && python -m MSUtils.general.h52xdmf {os.path.basename(result_fn)}",
                    shell=True,
                )
    return


@app.cell(hide_code=True)
def _(mo, resolution, sim_output_select, visu_available):
    do_inapp_visu = False
    if visu_available:
        if (
            resolution[0] * resolution[1] * resolution[2] <= 64 * 64 * 64
        ):  # and (len(sim_output_select.value) > 0):
            do_inapp_visu = True
            _options = ["microstructure"] + sim_output_select.value
            visu_output_select = mo.ui.dropdown(
                label="Select field to visualize:", options=_options
            )
            mo.output.replace(visu_output_select)
        else:
            do_inapp_visu = False
            mo.output.replace(
                mo.md(
                    "Resolution too fine for in-app visualization! Please use ParaView to view the generated xdmf-file."
                )
            )
    return do_inapp_visu, visu_output_select


@app.cell(hide_code=True)
def _(do_inapp_visu, domain_size, measure, np, resolution):
    if do_inapp_visu:
        # Parameters from your XDMF
        origin = np.array([0, 0, 0])
        spacing = np.array([domain_size[_i] / resolution[_i] for _i in range(3)])

        _surface_mask = np.zeros(resolution, dtype=bool)
        _surface_mask[0, :, :] = True
        _surface_mask[-1, :, :] = True
        _surface_mask[:, 0, :] = True
        _surface_mask[:, -1, :] = True
        _surface_mask[:, :, 0] = True
        _surface_mask[:, :, -1] = True

        # Use marching cubes to extract isosurface based on level
        level = 0.5
        verts, faces, normals, values = measure.marching_cubes(
            _surface_mask, level=level
        )

        # Scale verts to physical coordinates
        verts = origin + verts * spacing
    return faces, spacing, verts


@app.cell(hide_code=True)
def _(
    do_inapp_visu,
    faces,
    h5py,
    map_coordinates,
    mo,
    ms_dsn,
    ms_fn,
    n_grains,
    np,
    result_fn,
    result_prefix_input,
    spacing,
    trimesh,
    verts,
    visu_output_select,
):
    if do_inapp_visu:
        if visu_output_select.value is not None:
            # Load scalar data from HDF5
            result_dsn = f"{ms_dsn}/eroded_image_results/{result_prefix_input.value}/load0/time_step0/"
            _h5_fn = result_fn
            match visu_output_select.value:
                case "microstructure":
                    result_dsn = f"{ms_dsn}/eroded_image/"
                    _h5_fn = ms_fn
                case "concentration field":
                    result_dsn += "displacement"
                case "concentration fluctuation field":
                    result_dsn += "displacement_fluctuation"
                case "concentration gradient field":
                    result_dsn += "strain"
                case "flux field":
                    result_dsn += "stress"

            with h5py.File(_h5_fn, "r") as f:
                _field_ref = f[result_dsn]
                if len(_field_ref.shape) > 3:
                    match _field_ref.shape[-1]:
                        case 1:
                            _field = _field_ref[:][:, :, :, 0]
                        case 3:
                            _field = np.linalg.norm(_field_ref[:], axis=-1)
                else:
                    _field = _field_ref[:]
                    if visu_output_select.value == "microstructure":
                        _field = _field < n_grains

            vertex_scalars = map_coordinates(
                _field, verts.T / spacing[:, None], order=1
            )
            vertex_colors = trimesh.visual.interpolate(
                vertex_scalars, color_map="viridis"
            )

            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_colors=vertex_colors, process=True
            )
            mo.output.replace(mesh.show(viewer="marimo"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
