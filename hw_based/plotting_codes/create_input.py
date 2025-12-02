def generate_cdd_param_file(filename,halo_cen,halo_r,snapshot_dir,snapshot,
                            output_dir,halo,domain_dir=None,r_scale=1.2,verbose="T",
                            deut2H_nb_ratio=3.0e-5,f_ion=0.01,Zref=0.005,
                            itemp=5,imetal=6,ihii=7,iheii=8,iheiii=9):
    """
    """
    if domain_dir is not None:
        dom_dump_dir = f"{domain_dir}"
    else:
        dom_dump_dir = f"{output_dir}/halo_{halo}"

    lb = "!--------------------------------------------------------------------------------"
    file_structure = [
        lb,
        "[CreateDomDump]",
        f"DomDumpDir = {dom_dump_dir}",
        f"repository = {snapshot_dir}",
        f"snapnum    = {snapshot}",
        "reading_method = hilbert",
        "",
        "comput_dom_type      = sphere",
        f"comput_dom_pos       = {halo_cen[0]}, {halo_cen[1]}, {halo_cen[2]}",
        f"comput_dom_rsp       = {halo_r}",
        "",
        "decomp_dom_type      = sphere",
        "decomp_dom_ndomain   = 1",
        f"decomp_dom_xc        = {halo_cen[0]}",
        f"decomp_dom_yc        = {halo_cen[1]}",
        f"decomp_dom_zc        = {halo_cen[2]}",
        f"decomp_dom_rsp       = {r_scale * halo_r}",
        ""
        f"verbose    = {verbose}",
        lb,
        "",
        lb,
        "[mesh]",
        "verbose = T",
        lb,
        "",
        lb,
        "[gas_composition]",
        "# mixture parameters",
        f"deut2H_nb_ratio     = {deut2H_nb_ratio}",
        f"f_ion               = {f_ion}",
        f"Zref                = {Zref}",
        "# overwrite parameters",
        "gas_overwrite       = F",
        lb,
        "",
        lb,
        "[ramses]",
        "self_shielding   = F",
        "ramses_rt        = T",
        "verbose          = T",
        "cosmo            = T",
        "use_initial_mass = T",
        "use_proper_time  = F",
        f"itemp  = {itemp}",
        f"imetal = {imetal}",
        f"ihii   = {ihii}",
        f"iheii  = {iheii}",
        f"iheiii = {iheiii}",
        lb,
    ]

    # Open the file
    target = open(filename, 'w')

    for line in file_structure:
        target.write(f"{line}\n")

    target.close()

    return