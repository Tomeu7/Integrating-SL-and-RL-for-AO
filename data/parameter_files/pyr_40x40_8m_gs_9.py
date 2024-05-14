import shesha.config as conf
import numpy as np

# loop
p_loop = conf.Param_loop()
p_loop.set_niter(10000000)
p_loop.set_ittime(0.001)
p_loop.set_devices([0, 1, 2, 3])
# geom
p_geom = conf.Param_geom()
p_geom.set_zenithangle(0.)
# p_geom.set_pupdiam(56 * 8)

# tel
p_tel = conf.Param_tel()
p_tel.set_diam(8)
p_tel.set_cobs(0.14)


# atmos
p_atmos = conf.Param_atmos()
p_atmos.set_r0(0.16)
p_atmos.set_nscreens(3)
p_atmos.set_frac([0.50, 0.35, 0.15])
p_atmos.set_alt([0, 4000, 10000])
p_atmos.set_windspeed([15, 15, 35])
p_atmos.set_winddir([0, 20, 180])
p_atmos.set_L0([30, 30, 30])


# target
p_target = conf.Param_target()
p_targets = [p_target]
p_target.set_dms_seen([0, 1])
p_target.set_xpos(0.)
p_target.set_ypos(0.)
p_target.set_Lambda(1.6)
p_target.set_mag(0.)


p_wfs0 = conf.Param_wfs()
p_wfss = [p_wfs0]

p_wfs0.set_type("pyrhr")
p_wfs0.set_nxsub(56)
p_wfs0.set_fssize(2.5)  # arcsec
p_wfs0.set_fracsub(.1)
p_wfs0.set_xpos(0.)
p_wfs0.set_ypos(0.)
p_wfs0.set_Lambda(0.85)  # microns
p_wfs0.set_gsmag(9.)
p_wfs0.set_optthroughput(0.15) # throughput + multip. L3 CCD noise
p_wfs0.set_zerop(1.7e10)
p_wfs0.set_noise(3)
p_wfs0.set_fstop("round")
"""
rMod = 3
p_wfs0.set_pyr_ampl(rMod)
nbPtMod = int(np.ceil(int(rMod * 2 * 3.141592653589793) / 4.) * 4)
if rMod == 0:
    nbPtMod = 1
"""
p_wfs0.set_pyr_npts(1)
p_wfs0.set_pyr_ampl(0)
p_wfs0.set_atmos_seen(1)
p_wfs0.set_dms_seen(np.array([0, 1]))

# dm
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dms = [p_dm0, p_dm1]
p_dm0.set_type("pzt")
nact = 40
p_dm0.set_nact(nact)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3)
p_dm0.set_coupling(0.3)
p_dm0.set_unitpervolt(0.01)  # microns/volt (“pzt”)
p_dm0.set_push4imat(1)  # volt
p_dm0.set_influ_type("gaussian")

p_dm1.set_type("tt")
p_dm1.set_alt(0.)
p_dm1.set_unitpervolt(0.0005)
p_dm1.set_push4imat(10.)


# centroiders
p_centroider0 = conf.Param_centroider()
p_centroiders = [p_centroider0]

p_centroider0.set_nwfs(0)
p_centroider0.set_type("maskedpix")

# controllers
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("ls")
p_controller0.set_nwfs([0])
p_controller0.set_ndm([0, 1])
p_controller0.set_maxcond(100000)
p_controller0.set_delay(1)
p_controller0.set_gain(0.6)
