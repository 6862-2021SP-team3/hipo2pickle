#!/usr/bin/env python3
"""
A simple script to save Z and X of 6862 nflow project.
"""

import uproot
import pandas as pd
import numpy as np
import argparse
from copy import copy

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector

def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

def mag2(vec1):
    # square of L2 norm
    return  dot(vec1, vec1)

def cosTheta(vec1, vec2):
    # cosine angle between two 3d vectors
    return dot(vec1,vec2)/np.sqrt(mag2(vec1) * mag2(vec2))

def angle(vec1, vec2):
    # angle between two 3d vectors
    return 180/np.pi*np.arccos(np.minimum(1, cosTheta(vec1, vec2)))

def cross(vec1, vec2):
    # cross product of two 3d vectors
    return [vec1[1]*vec2[2]-vec1[2]*vec2[1], vec1[2]*vec2[0]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0]]

def vecAdd(gam1, gam2):
    # add two 3d vectors
    return [gam1[0]+gam2[0], gam1[1]+gam2[1], gam1[2]+gam2[2]]

def pi0Energy(gam1, gam2):
    # reconstructed pi0 energy of two 3d photon momenta
    return mag(gam1)+mag(gam2)

def pi0InvMass(gam1, gam2):
    # pi0 invariant mass of two 3d photon momenta
    pi0mass2 = pi0Energy(gam1, gam2)**2-mag2(vecAdd(gam1, gam2))
    pi0mass2 = np.where(pi0mass2 >= 0, pi0mass2, 10**6)
    pi0mass = np.sqrt(pi0mass2)
    pi0mass = np.where(pi0mass > 100, -1000, pi0mass)
    return pi0mass

def getPhi(vec1):
    # azimuthal angle of one 3d vector
    return 180/np.pi*np.arctan2(vec1[1], vec1[0])

def getTheta(vec1):
    # polar angle of one 3d vector
    return 180/np.pi*np.arctan2(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]), vec1[2])

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)


class root2pickle():
    #class to read root to make epg pairs, inherited from epg
    def __init__(self, fname, entry_stop = None, gen = "dvcs"):
        self.fname = fname
        self.readEPGG(entry_stop, gen = gen)
        self.saveDVpi0vars()
        self.makeDVpi0()
        self.saveDVCSvars()
        self.makeDVCS()
        self.saveDfCartesian()

    def readFile(self):
        #read root using uproot
        self.file = uproot.open(self.fname)
        self.tree = self.file["T"]

    def closeFile(self):
        #close file for saving memory
        self.file = None
        self.tree = None

    def readEPGG(self, entry_stop = None, gen = 'dvcs'):
        #save data into df_epg, df_epgg for parent class epg
        self.readFile()

        if entry_stop is not None:
            entry_stop = int(entry_stop)

        # data frames and their keys to read Z part
        df_electronGen = pd.DataFrame()
        df_protonGen = pd.DataFrame()
        df_gammaGen = pd.DataFrame()
        eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
        proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
        gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
        # read keys
        for key in eleKeysGen:
            df_electronGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in proKeysGen:
            df_protonGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in gamKeysGen:
            df_gammaGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

        #convert data type to standard double
        df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
        df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
        df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})

        #set up a dummy index for merging
        df_electronGen.loc[:,'event'] = df_electronGen.index
        df_protonGen.loc[:,'event'] = df_protonGen.index
        df_gammaGen.loc[:,'event'] = df_gammaGen.index.get_level_values('entry')

        #sort columns for readability
        df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz"]]

        if gen == "dvcs":
            df_gammaGen = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
            # gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            # df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
            # df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
            # df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

        elif gen=="pi0":
            #two g's to one gg.
            gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)

            gam1 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
            gam1 = gam1.reset_index(drop=True)
            gam2 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==1]
            gam2 = gam2.reset_index(drop=True)

            gam1.loc[:,"GenGp2"] = gam2.loc[:,"GenGp"]
            gam1.loc[:,"GenGpx2"] = gam2.loc[:,"GenGpx"]
            gam1.loc[:,"GenGpy2"] = gam2.loc[:,"GenGpy"]
            gam1.loc[:,"GenGpz2"] = gam2.loc[:,"GenGpz"]
            df_gammaGen = gam1

            #sort GenG indices so that GenGp > GenGp2. This is because Gp > Gp2 at reconstruction level.
            df_gammaGencopy = copy(df_gammaGen)
            df_gammaGencopy.loc[:, "GenGp"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp"], df_gammaGen.loc[:, "GenGp2"])
            df_gammaGencopy.loc[:, "GenGpx"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx"], df_gammaGen.loc[:, "GenGpx2"])
            df_gammaGencopy.loc[:, "GenGpy"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy"], df_gammaGen.loc[:, "GenGpy2"])
            df_gammaGencopy.loc[:, "GenGpz"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz"], df_gammaGen.loc[:, "GenGpz2"])
            df_gammaGencopy.loc[:, "GenGp2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp2"], df_gammaGen.loc[:, "GenGp"])
            df_gammaGencopy.loc[:, "GenGpx2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx2"], df_gammaGen.loc[:, "GenGpx"])
            df_gammaGencopy.loc[:, "GenGpy2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy2"], df_gammaGen.loc[:, "GenGpy"])
            df_gammaGencopy.loc[:, "GenGpz2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz2"], df_gammaGen.loc[:, "GenGpz"])
            df_gammaGen = df_gammaGencopy


        # #spherical coordinates
        # eleGen = [df_electronGen["GenEpx"], df_electronGen["GenEpy"], df_electronGen["GenEpz"]]
        # df_electronGen.loc[:, 'GenEe'] = getEnergy(eleGen, me)
        # df_electronGen.loc[:, 'GenEp'] = mag(eleGen)
        # df_electronGen.loc[:, 'GenEtheta'] = getTheta(eleGen)
        # df_electronGen.loc[:, 'GenEphi'] = getPhi(eleGen)

        # proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
        # df_protonGen.loc[:, 'GenPe'] = getEnergy(proGen, M)
        # df_protonGen.loc[:, 'GenPp'] = mag(proGen)
        # df_protonGen.loc[:, 'GenPtheta'] = getTheta(proGen)
        # df_protonGen.loc[:, 'GenPphi'] = getPhi(proGen)

        # gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
        # # df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
        # df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
        # df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

        # gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
        # debug = df_gammaGen.loc[:, 'GenGp2'] == mag(gamGen2)
        # df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
        # df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

        df_z = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
        df_z = pd.merge(df_z, df_gammaGen, how='inner', on='event')
        self.df_z = df_z    #done with saving z

        # data frames and their keys to read X part
        df_electronRec = pd.DataFrame()
        df_protonRec = pd.DataFrame()
        df_gammaRec = pd.DataFrame()
        eleKeysRec = ["Epx", "Epy", "Epz", "Esector"]
        proKeysRec = ["Ppx", "Ppy", "Ppz", "Psector"]
        gamKeysRec = ["Gpx", "Gpy", "Gpz", "Gsector"]
        # read them
        for key in eleKeysRec:
            df_electronRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in proKeysRec:
            df_protonRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in gamKeysRec:
            df_gammaRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

        self.closeFile()

        #convert data type to standard double
        df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float})
        df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float})
        df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float})

        #set up a dummy index for merging
        df_electronRec.loc[:,'event'] = df_electronRec.index
        df_protonRec.loc[:,'event'] = df_protonRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'event'] = df_gammaRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'GIndex'] = df_gammaRec.index.get_level_values('subentry')

        #save only FD protons and photons
        df_protonRec = df_protonRec[df_protonRec["Psector"]<7]
        df_gammaRec = df_gammaRec[df_gammaRec["Gsector"]<7]

        df_gg = pd.merge(df_gammaRec, df_gammaRec,
                         how='outer', on='event', suffixes=("", "2"))
        df_gg = df_gg[df_gg["GIndex"] < df_gg["GIndex2"]]
        df_ep = pd.merge(df_electronRec, df_protonRec, how='outer', on='event')

        df_epgg = pd.merge(df_ep, df_gg, how='outer', on='event')
        df_epgg = df_epgg[~np.isnan(df_epgg["Ppx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx2"])]

        self.df_epgg = df_epgg #temporarily save df_epgg

        df_epg = pd.merge(df_ep, df_gammaRec, how='outer', on='event')
        df_epg = df_epg[~np.isnan(df_epg["Ppx"])]
        df_epg = df_epg[~np.isnan(df_epg["Gpx"])]

        self.df_epg = df_epg #temporarily save df_epgg


    def saveDVpi0vars(self):
        #set up pi0 variables
        df_epgg = self.df_epgg

        # useful objects
        ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
        df_epgg.loc[:, 'Ep'] = mag(ele)
        df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
        df_epgg.loc[:, 'Etheta'] = getTheta(ele)
        df_epgg.loc[:, 'Ephi'] = getPhi(ele)

        pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
        df_epgg.loc[:, 'Pp'] = mag(pro)
        df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
        df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
        df_epgg.loc[:, 'Pphi'] = getPhi(pro)

        gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
        df_epgg.loc[:, 'Gp'] = mag(gam)
        df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
        df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
        df_epgg.loc[:, 'Gphi'] = getPhi(gam)

        gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
        df_epgg.loc[:, 'Gp2'] = mag(gam2)
        df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
        df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
        df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

        pi0 = vecAdd(gam, gam2)
        VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, gam)
        VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
                    df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
        VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
                    df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
                    -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
                    pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]

        df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

        # binning kinematics
        df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
        df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
        df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
        df_epgg.loc[:,'t'] = 2 * M * (df_epgg['Pe'] - M)
        df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
        df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
                                 (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)

        # exclusivity variables
        df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
                             df_epgg["Pe"])**2 - mag2(VmissPi0)
        df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
                             df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
        df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
                             df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
        df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
        df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
        df_epgg.loc[:,'reconPi'] = angle(VmissPi0, pi0)
        df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']
        self.df_epgg = df_epgg

    def makeDVpi0(self):
        #make dvpi0 pairs
        df_epgg = self.df_epgg

        # if mode == "pi0":
        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = df_epgg.loc[:, "Q2"] > 1  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        # Exclusivity cuts
        cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
        cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
        cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
        cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])

        df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_mmep & cut_meepgg &
                           cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]

        #For an event, there can be two gg's passed conditions above.
        #Take only one gg's that makes pi0 invariant mass
        #This case is very rare.
        #For now, duplicated proton is not considered.
        df_dvpi0 = df_dvpi0.sort_values(by='closeness', ascending=True)
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event') 

        # elif mode == "epgg":     
        #     # I assume they are all sorted already, but sorting twice won't hurt.
        #     df_dvpi0 = df_epgg
        #     df_dvpi0['Epi0'] = df_dvpi0['Gp'] + df_dvpi0['Gp2']
        #     df_dvpi0 = df_dvpi0.sort_values(by='Epi0', ascending = False)
        #     df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        #     df_dvpi0 = df_dvpi0.sort_values(by='event')
        # else:
        #     print("mode must be pi0 or epgg")
        #     exit()     

        df_dvpi0 = df_dvpi0.loc[:, ["event", "Epx", "Epy", "Epz", "Ep", "Etheta", "Ephi", "Ee", "Ppx", "Ppy", "Ppz", "Pp", "Ptheta", "Pphi", "Pe", "Gpx", "Gpy", "Gpz", "Gp", "Gtheta", "Gphi", "Ge", "Gpx2", "Gpy2", "Gpz2", "Gp2", "Gtheta2", "Gphi2", "Ge2", "Esector", "Gsector", "Gsector2"]]
        self.df_dvpi0 = df_dvpi0 #done with saving x

    def saveDVCSvars(self, correction=None):
        #set up dvcs variables
        df_epg = self.df_epg

        ele = [df_epg['Epx'], df_epg['Epy'], df_epg['Epz']]
        df_epg.loc[:, 'Ep'] = mag(ele)
        df_epg.loc[:, 'Ee'] = getEnergy(ele, me)
        df_epg.loc[:, 'Etheta'] = getTheta(ele)
        df_epg.loc[:, 'Ephi'] = getPhi(ele)

        pro = [df_epg['Ppx'], df_epg['Ppy'], df_epg['Ppz']]

        gam = [df_epg['Gpx'], df_epg['Gpy'], df_epg['Gpz']]
        df_epg.loc[:, 'Gp'] = mag(gam)
        df_epg.loc[:, 'Ge'] = getEnergy(gam, 0)
        df_epg.loc[:, 'Gtheta'] = getTheta(gam)
        df_epg.loc[:, 'Gphi'] = getPhi(gam)

        Ppt = mag([df_epg['Ppx'], df_epg['Ppy'], 0])

        VGS = [-df_epg['Epx'], -df_epg['Epy'], pbeam - df_epg['Epz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, gam)
        VmissG = [-df_epg["Epx"] - df_epg["Ppx"], -df_epg["Epy"] - df_epg["Ppy"],
                  pbeam - df_epg["Epz"] - df_epg["Ppz"]]
        VmissP = [-(df_epg["Epx"] + df_epg["Gpx"]), -(df_epg["Epy"] + df_epg["Gpy"]),
                  -(-pbeam + df_epg["Epz"] + df_epg["Gpz"])]
        Vmiss = [-(df_epg["Epx"] + df_epg["Ppx"] + df_epg["Gpx"]), -(df_epg["Epy"] + df_epg["Ppy"] + df_epg["Gpy"]),
                 -(-pbeam + df_epg["Epz"] + df_epg["Ppz"] + df_epg["Gpz"])]
        costheta = cosTheta(VGS, gam)

        df_epg.loc[:, 'Mpx'], df_epg.loc[:, 'Mpy'], df_epg.loc[:, 'Mpz'] = Vmiss

        # binning kinematics
        df_epg.loc[:,'Q2'] = -((ebeam - df_epg['Ee'])**2 - mag2(VGS))
        df_epg.loc[:,'nu'] = (ebeam - df_epg['Ee'])
        df_epg.loc[:,'y'] = df_epg['nu']/ebeam
        df_epg.loc[:,'xB'] = df_epg['Q2'] / 2.0 / M / df_epg['nu']
        df_epg.loc[:,'t1'] = 2 * M * (df_epg['Pe'] - M)
        df_epg.loc[:,'t2'] = (M * df_epg['Q2'] + 2 * M * df_epg['nu'] * (df_epg['nu'] - np.sqrt(df_epg['nu'] * df_epg['nu'] + df_epg['Q2']) * costheta))\
        / (M + df_epg['nu'] - np.sqrt(df_epg['nu'] * df_epg['nu'] + df_epg['Q2']) * costheta)
        df_epg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epg['Ee'])**2 - mag2(VGS)))

        # trento angles
        df_epg.loc[:,'phi1'] = angle(v3l, v3h)
        df_epg.loc[:,'phi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                  df_epg['phi1'], df_epg['phi1'])
        df_epg.loc[:,'phi2'] = angle(v3l, v3g)
        df_epg.loc[:,'phi2'] = np.where(dot(v3l, gam) <
                                  0, 360.0 - df_epg['phi2'], df_epg['phi2'])

        # exclusivity variables
        df_epg.loc[:,'MM2_epg'] = (-M - ebeam + df_epg["Ee"] +
                             df_epg["Pe"] + df_epg["Ge"])**2 - mag2(Vmiss)
        df_epg.loc[:,'ME_epg'] = (M + ebeam - df_epg["Ee"] - df_epg["Pe"] - df_epg["Ge"])
        df_epg.loc[:,'MM2_ep'] = (-M - ebeam + df_epg["Ee"] + df_epg["Pe"])**2 - mag2(VmissG)
        df_epg.loc[:,'MM2_eg'] = (-M - ebeam + df_epg["Ee"] + df_epg["Ge"])**2 - mag2(VmissP)
        df_epg.loc[:,'MPt'] = np.sqrt((df_epg["Epx"] + df_epg["Ppx"] + df_epg["Gpx"])**2 +
                                (df_epg["Epy"] + df_epg["Ppy"] + df_epg["Gpy"])**2)
        df_epg.loc[:,'coneAngle'] = angle(ele, gam)
        df_epg.loc[:,'reconGam'] = angle(gam, VmissG)
        df_epg.loc[:,'coplanarity'] = angle(v3h, v3g)
        self.df_epg = df_epg

    def makeDVCS(self):
        #make dvcs pairs
        df_dvcs = self.df_epg
        df_dvcs = df_dvcs[df_dvcs["MM2_eg"] > 0]  # mmeg

        cut_xBupper = df_dvcs["xB"] < 1  # xB
        cut_xBlower = df_dvcs["xB"] > 0  # xB
        cut_Q2 = df_dvcs["Q2"] > 1  # Q2
        cut_W = df_dvcs["W"] > 2  # W
        cut_Ee = df_dvcs["Ee"] > 2  # Ee
        cut_Ge = df_dvcs["Ge"] > 3  # Ge
        cut_Pp = mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]]) > 0.12  # Pp
        cut_Vz = 1#np.abs(df_dvcs["Evz"] - df_dvcs["Pvz"]) < 2.5 + 2.5 / mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]])

        #   Exclusivity cuts
        cut_mmepg = np.abs(df_dvcs["MM2_epg"]) < 0.1  # mmepg
        cut_mmep = np.abs(df_dvcs["MM2_ep"]) < 0.6  # mmep
        cut_mmegupper = df_dvcs["MM2_eg"] < 3  # mmeg
        cut_mmeglower = df_dvcs["MM2_eg"] > 0  # mmeg
        cut_meepgupper = df_dvcs["ME_epg"] < 1.5  # meepg
        cut_meepglower = df_dvcs["ME_epg"] > -0.5  # meepg
        cut_mpt = df_dvcs["MPt"] < 0.25  # mpt
        cut_cone = df_dvcs["coneAngle"] > 5  # coneangle
        cut_recon = df_dvcs["reconGam"] < 2.5  # recon gam angle
        cut_coplanarity = df_dvcs["coplanarity"] < 25  # coplanarity angle
        if "Esector" in df_dvcs:
            cut_sector = df_dvcs["Esector"]!=df_dvcs["Gsector"]
        else:
            cut_sector = 1

        df_dvcs = df_dvcs[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Pp & cut_Vz & cut_mmepg & cut_mmep &
                         cut_mmegupper & cut_mmeglower & cut_meepgupper & cut_meepglower & cut_mpt & cut_cone & cut_recon & cut_sector]

        #dealing with duplicates
        df_dvcs = df_dvcs.sort_values(by='Ge', ascending = False)
        df_dvcs = df_dvcs.loc[~df_dvcs.event.duplicated(), :]
        df_dvcs = df_dvcs.sort_values(by='event')
        self.df_dvcs = df_dvcs               

    def pi02gSubtraction(self):
        #exclude dvpi0 from dvcs. use only when both set up.
        df_dvcs = self.df_dvcs
        pi0to2gammas = df_dvcs["event"].isin(self.df_dvpi0["event"])
        df_dvcs = df_dvcs[~pi0to2gammas]
        self.df_dvcs = df_dvcs

    def saveDfCartesian(self):
        df_z = self.df_z
        df_x = self.df_x

        df_z = df_z.rename(columns = {"GenEpx": "z1", "GenEpy": "z2", "GenEpz": "z3", "GenEe": "z0", "GenPpx": "z5", "GenPpy": "z6", "GenPpz": "z7", "GenPe": "z4", "GenGpx": "z9", "GenGpy": "z10", "GenGpz": "z11"})
        # df_z.loc[:, "z03"] = 1 # electron
        # df_z.loc[:, "z13"] = 2 # proton
        df_z.loc[:, "z8"] = df_z.loc[:, "GenGp"] # photon
        # df_z.loc[:, "z12"] = df_z.loc[:, "GenGp2"] # photon2
        df_z = df_z.loc[:, ["event", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11"]]#, "z12", "z13", "z14", "z15"]]

        df_x = df_x.rename(columns = {"Epx": "1", "Epy": "2", "Epz": "3", "Ee": "0","Ppx": "5", "Ppy": "6", "Ppz": "7", "Pe": "4", "Gpx": "9", "Gpy": "10", "Gpz": "11", "Ge": "8"})#, "Gpx2": "13", "Gpy2": "14", "Gpz2": "15", "Ge2": "12"})
        # df_x.loc[:, "3"] = 1 # electron
        # df_x.loc[:, "x13"] = 2 # proton
        # df_x.loc[:, "x23"] = 3 # photon
        # df_x.loc[:, "x33"] = 3 # photon2
        df_x = df_x.loc[:, ["event", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]]#, "12", "13", "14", "15", "Esector", "Gsector", "Gsector2"]]

        df = pd.merge(df_x, df_z, how = 'inner', on='event')
        self.df = df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="/Users/sangbaek/Dropbox (MIT)/data/project/merged_9628_files.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="goodbyeRoot.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    # parser.add_argument("-m","--mode", help="dvcs cuts or all epg", default = "dvcs")
    parser.add_argument("-g","--generator", help="dvcs or pi0", default = "dvcs")
    
    args = parser.parse_args()

    converter = root2pickle(args.fname, entry_stop = args.entry_stop, gen = args.generator)#, mode = args.mode)
    df = converter.df

    df.to_pickle(args.out)