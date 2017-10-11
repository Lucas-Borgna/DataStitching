from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import root_numpy as rtnp
import pandas as pd
import ROOT
import time
import progressbar
from array import array
import pdb


#-----------------------------------------------------------------------------------------
#   REMEMBER TO REMOVE ANY PREVIOUSLY CREATED STITCHED FILES
#-----------------------------------------------------------------------------------------

#Original Files
sig_storage_all = "/mnt/storage/lborgna/Wprime/"
m600_file = sig_storage_all+"m600.root"
m800_file = sig_storage_all+"m800.root"
m1000_file = sig_storage_all+"m1000.root"
m1100_file = sig_storage_all+"m1100.root"
m1200_file = sig_storage_all+"m1200.root"
m1300_file = sig_storage_all+"m1300.root"
m1400_file = sig_storage_all+"m1400.root"
m1500_file = sig_storage_all+"m1500.root"
m1600_file = sig_storage_all+"m1600.root"
m1700_file = sig_storage_all+"m1700.root"
m1800_file = sig_storage_all+"m1800.root"
m1900_file = sig_storage_all+"m1900.root"
m2000_file = sig_storage_all+"m2000.root"
m2100_file = sig_storage_all+"m2100.root"
m2200_file = sig_storage_all+"m2200.root"
m2300_file = sig_storage_all+"m2300.root"
m2400_file = sig_storage_all+"m2400.root"
m2500_file = sig_storage_all+"m2500.root"
m2600_file = sig_storage_all+"m2600.root"
m2700_file = sig_storage_all+"m2700.root"
m2800_file = sig_storage_all+"m2800.root"
m2900_file = sig_storage_all+"m2900.root"
m3000_file = sig_storage_all+"m3000.root"
m3200_file = sig_storage_all+"m3200.root"
m3400_file = sig_storage_all+"m3400.root"
m3600_file = sig_storage_all+"m3600.root"
m3800_file = sig_storage_all+"m3800.root"
m4000_file = sig_storage_all+"m4000.root"
m4200_file = sig_storage_all+"m4200.root"
m4400_file = sig_storage_all+"m4400.root"
m4600_file = sig_storage_all+"m4600.root"
m4800_file = sig_storage_all+"m4800.root"
m5000_file = sig_storage_all+"m5000.root"

treename = "FlatSubstructureJetTree"


#2 Trees, with different number labels, lowest number corresponds to training
# Higher number corresponds to testing sets.
# This file is for creating the training sets.
pt1 = rtnp.root2array(m600_file, treename = treename +';6', branches = "fjet_pt")
pt2 = rtnp.root2array(m800_file, treename = treename +';4', branches = "fjet_pt")
pt3 = rtnp.root2array(m1000_file, treename = treename +';4', branches = "fjet_pt")
pt4 = rtnp.root2array(m1100_file, treename = treename +';3', branches = "fjet_pt")
pt5 = rtnp.root2array(m1200_file, treename = treename + ';3', branches = "fjet_pt")
pt6 = rtnp.root2array(m1300_file, treename = treename + ';3', branches = "fjet_pt")
pt7 = rtnp.root2array(m1400_file, treename = treename + ';3', branches = "fjet_pt")
pt8 = rtnp.root2array(m1500_file, treename = treename + ';3', branches = "fjet_pt")
pt9 = rtnp.root2array(m1600_file, treename = treename + ';3', branches = "fjet_pt")
pt10 = rtnp.root2array(m1700_file, treename = treename + ';3', branches = "fjet_pt")
pt11 = rtnp.root2array(m1800_file, treename = treename + ';3', branches = "fjet_pt")
pt12 = rtnp.root2array(m1900_file, treename = treename + ';3', branches = "fjet_pt")
pt13 = rtnp.root2array(m2000_file, treename = treename + ';4', branches = "fjet_pt")
pt14 = rtnp.root2array(m2100_file, treename = treename + ';3', branches = "fjet_pt")
pt15 = rtnp.root2array(m2200_file, treename = treename + ';3', branches = "fjet_pt")
pt16 = rtnp.root2array(m2300_file, treename = treename + ';3', branches = "fjet_pt")
pt17 = rtnp.root2array(m2400_file, treename = treename + ';3', branches = "fjet_pt")
pt18 = rtnp.root2array(m2500_file, treename = treename + ';3', branches = "fjet_pt")
pt19 = rtnp.root2array(m2600_file, treename = treename + ';3', branches = "fjet_pt")
pt20 = rtnp.root2array(m2700_file, treename = treename + ';3', branches = "fjet_pt")
pt21 = rtnp.root2array(m2800_file, treename = treename + ';3', branches = "fjet_pt")
pt22 = rtnp.root2array(m2900_file, treename = treename + ';3', branches = "fjet_pt")
pt23 = rtnp.root2array(m3000_file, treename = treename + ';3', branches = "fjet_pt")
pt24 = rtnp.root2array(m3200_file, treename = treename + ';3', branches = "fjet_pt")
pt25 = rtnp.root2array(m3400_file, treename = treename + ';3', branches = "fjet_pt")
pt26 = rtnp.root2array(m3600_file, treename = treename + ';3', branches = "fjet_pt")
pt27 = rtnp.root2array(m3800_file, treename = treename + ';2', branches = "fjet_pt")
pt28 = rtnp.root2array(m4000_file, treename = treename + ';2', branches = "fjet_pt")
pt29 = rtnp.root2array(m4200_file, treename = treename + ';2', branches = "fjet_pt")
pt30 = rtnp.root2array(m4400_file, treename = treename + ';2', branches = "fjet_pt")
pt31 = rtnp.root2array(m4600_file, treename = treename + ';2', branches = "fjet_pt")
pt32 = rtnp.root2array(m4800_file, treename = treename + ';2', branches = "fjet_pt")
pt33 = rtnp.root2array(m5000_file, treename = treename + ';6', branches = "fjet_pt")






length1 = len(pt1)
length2 = len(pt2)
length3 = len(pt3)
length4 = len(pt4)
length5 = len(pt5)
length6 = len(pt6)
length7 = len(pt7)
length8 = len(pt8)
length9 = len(pt9)
length10 = len(pt10)
length11 = len(pt11)
length12 = len(pt12)
length13 = len(pt13)
length14 = len(pt14)
length15 = len(pt15)
length16 = len(pt16)
length17 = len(pt17)
length18 = len(pt18)
length19 = len(pt19)
length20 = len(pt20)
length21 = len(pt21)
length22 = len(pt22)
length23 = len(pt23)
length24 = len(pt24)
length25 = len(pt25)
length26 = len(pt26)
length27 = len(pt27)
length28 = len(pt28)
length29 = len(pt29)
length30 = len(pt30)
length31 = len(pt31)
length32 = len(pt32)
length33 = len(pt33)


#Creating New ROOT file via Update Method
out_storage = "/mnt/storage/lborgna/SignalStitch/"
out_file = "SigAll2_Test.root"

try:
    os.remove(out_storage+out_file)
    print("File Removed")
except OSError:
    pass

fnew = ROOT.TFile(out_storage+out_file,"update")
Tree = ROOT.TTree("FlatSubstructureJetTree", "Reconst ntuple")

#Gets the old file
test = ROOT.TFile.Open(m600_file)
old_tree = test.Get("FlatSubstructureJetTree;6")

cluster_E_entry = ROOT.vector('float')()
cluster_eta_entry = ROOT.vector('float')()
cluster_phi_entry = ROOT.vector('float')()
cluster_pt_entry  = ROOT.vector('float')()

fjet_pt_entry = array('f', [0.0])
fjet_feta_entry = array('f', [0.0])
fjet_fphi_entry = array('f', [0.0])
fjet_fE_entry = array('f', [0.0])
fjet_D2_entry = array('f', [0.0])
fjet_Tau21_wta_entry = array('f', [0.0])
fjet_truthJet_pt_entry = array('f', [0.0])
fjet_fatjet_dRmatched_particle_flavor_entry = array('I', [0])

#New Branches!
Tree.Branch('fjet_pt', fjet_pt_entry, 'fjet_pt/F')
Tree.Branch('fjet_eta', fjet_feta_entry, 'fjet_eta/F')
Tree.Branch('fjet_phi', fjet_fphi_entry, 'fjet_phi/F')
Tree.Branch('fjet_E', fjet_fE_entry, 'fjet_E/F')
Tree.Branch('fjet_fatjet_dRmatched_particle_flavor', fjet_fatjet_dRmatched_particle_flavor_entry, 'fjet_fatjet_dRmatched_particle_flavor/I')
Tree.Branch('fjet_Tau21_wta', fjet_Tau21_wta_entry,'fjet_Tau21_wta/F')
Tree.Branch('fjet_D2', fjet_D2_entry, 'fjet_D2/F')
Tree.Branch('fjet_truthJet_pt', fjet_truthJet_pt_entry, 'fjet_truthJet_pt/F')
Tree.Branch('clus_E', cluster_E_entry)
Tree.Branch('clus_eta', cluster_eta_entry)
Tree.Branch('clus_phi', cluster_phi_entry)
Tree.Branch('clus_pt', cluster_pt_entry)

tosscount = 0

bar = progressbar.ProgressBar()


for i in bar(xrange(length1)):
    old_tree.GetEntry(i)
    old_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    old_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    old_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    old_tree.SetBranchAddress("clus_E", cluster_E_entry)
    old_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    old_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    old_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    old_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    old_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    old_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    old_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    old_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)
    Tree.Fill()


test = ROOT.TFile.Open(m800_file)
m800_tree = test.Get("FlatSubstructureJetTree;4")

bar = progressbar.ProgressBar()

for i in bar(xrange(length2)):
    m800_tree.GetEntry(i)
    m800_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m800_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m800_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m800_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m800_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m800_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m800_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m800_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m800_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m800_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m800_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m800_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1000_file)
m1000_tree = test.Get("FlatSubstructureJetTree;4")

bar = progressbar.ProgressBar()

for i in bar(xrange(length3)):
    m1000_tree.GetEntry(i)
    m1000_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1000_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1000_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1000_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1000_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1000_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1000_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1000_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1000_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1000_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1000_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1000_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1100_file)
m1100_tree = test.Get("FlatSubstructureJetTree;3")

bar = progressbar.ProgressBar()

for i in bar(xrange(length4)):
    m1100_tree.GetEntry(i)
    m1100_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1100_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1100_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1100_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1100_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1100_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1100_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1100_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1100_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1100_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1100_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1100_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1200_file)
m1200_tree = test.Get("FlatSubstructureJetTree;3")

bar = progressbar.ProgressBar()

for i in bar(xrange(length5)):
    m1200_tree.GetEntry(i)
    m1200_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1200_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1200_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1200_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1200_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1200_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1200_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1200_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1200_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1200_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1200_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1200_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1300_file)
m1300_tree = test.Get("FlatSubstructureJetTree;3")

bar = progressbar.ProgressBar()

for i in bar(xrange(length6)):
    m1300_tree.GetEntry(i)
    m1300_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1300_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1300_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1300_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1300_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1300_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1300_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1300_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1300_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1300_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1300_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1300_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m1400_file)
m1400_tree = test.Get("FlatSubstructureJetTree;3")

bar = progressbar.ProgressBar()

for i in bar(xrange(length7)):
    m1400_tree.GetEntry(i)
    m1400_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1400_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1400_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1400_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1400_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1400_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1400_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1400_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1400_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1400_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1400_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1400_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1500_file)
m1500_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length8)):
    m1500_tree.GetEntry(i)
    m1500_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1500_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1500_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1500_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1500_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1500_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1500_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1500_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1500_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1500_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1500_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1500_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(m1600_file)
m1600_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length9)):
    m1600_tree.GetEntry(i)
    m1600_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1600_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1600_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1600_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1600_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1600_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1600_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1600_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1600_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1600_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1600_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1600_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m1700_file)
m1700_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length10)):
    m1700_tree.GetEntry(i)
    m1700_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1700_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1700_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1700_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1700_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1700_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1700_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1700_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1700_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1700_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1700_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1700_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m1800_file)
m1800_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length11)):
    m1800_tree.GetEntry(i)
    m1800_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1800_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1800_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1800_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1800_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1800_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1800_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1800_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1800_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1800_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1800_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1800_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m1900_file)
m1900_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length12)):
    m1900_tree.GetEntry(i)
    m1900_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m1900_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m1900_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m1900_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m1900_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m1900_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m1900_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m1900_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m1900_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m1900_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m1900_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m1900_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2000_file)
m2000_tree = test.Get("FlatSubstructureJetTree;4")
bar = progressbar.ProgressBar()


for i in bar(xrange(length13)):
    m2000_tree.GetEntry(i)
    m2000_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2000_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2000_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2000_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2000_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2000_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2000_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2000_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2000_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2000_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2000_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2000_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2100_file)
m2100_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length14)):
    m2100_tree.GetEntry(i)
    m2100_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2100_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2100_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2100_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2100_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2100_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2100_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2100_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2100_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2100_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2100_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2100_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2200_file)
m2200_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length15)):
    m2200_tree.GetEntry(i)
    m2200_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2200_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2200_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2200_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2200_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2200_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2200_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2200_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2200_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2200_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2200_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2200_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2300_file)
m2300_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length16)):
    m2300_tree.GetEntry(i)
    m2300_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2300_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2300_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2300_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2300_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2300_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2300_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2300_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2300_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2300_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2300_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2300_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2400_file)
m2400_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length17)):
    m2400_tree.GetEntry(i)
    m2400_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2400_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2400_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2400_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2400_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2400_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2400_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2400_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2400_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2400_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2400_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2400_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2500_file)
m2500_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length18)):
    m2500_tree.GetEntry(i)
    m2500_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2500_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2500_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2500_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2500_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2500_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2500_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2500_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2500_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2500_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2500_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2500_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2600_file)
m2600_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length19)):
    m2600_tree.GetEntry(i)
    m2600_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2600_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2600_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2600_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2600_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2600_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2600_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2600_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2600_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2600_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2600_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2600_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2700_file)
m2700_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length20)):
    m2700_tree.GetEntry(i)
    m2700_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2700_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2700_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2700_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2700_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2700_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2700_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2700_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2700_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2700_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2700_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2700_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2800_file)
m2800_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length21)):
    m2800_tree.GetEntry(i)
    m2800_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2800_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2800_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2800_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2800_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2800_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2800_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2800_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2800_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2800_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2800_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2800_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m2900_file)
m2900_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length22)):
    m2900_tree.GetEntry(i)
    m2900_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m2900_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m2900_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m2900_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m2900_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m2900_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m2900_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m2900_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m2900_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m2900_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m2900_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m2900_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m3000_file)
m3000_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length23)):
    m3000_tree.GetEntry(i)
    m3000_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m3000_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m3000_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m3000_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m3000_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m3000_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m3000_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m3000_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m3000_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m3000_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m3000_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m3000_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m3200_file)
m3200_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length24)):
    m3200_tree.GetEntry(i)
    m3200_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m3200_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m3200_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m3200_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m3200_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m3200_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m3200_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m3200_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m3200_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m3200_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m3200_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m3200_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m3400_file)
m3400_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length25)):
    m3400_tree.GetEntry(i)
    m3400_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m3400_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m3400_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m3400_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m3400_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m3400_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m3400_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m3400_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m3400_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m3400_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m3400_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m3400_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m3600_file)
m3600_tree = test.Get("FlatSubstructureJetTree;3")
bar = progressbar.ProgressBar()


for i in bar(xrange(length26)):
    m3600_tree.GetEntry(i)
    m3600_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m3600_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m3600_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m3600_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m3600_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m3600_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m3600_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m3600_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m3600_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m3600_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m3600_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m3600_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m3800_file)
m3800_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length27)):
    m3800_tree.GetEntry(i)
    m3800_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m3800_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m3800_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m3800_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m3800_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m3800_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m3800_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m3800_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m3800_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m3800_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m3800_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m3800_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m4000_file)
m4000_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length28)):
    m4000_tree.GetEntry(i)
    m4000_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m4000_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m4000_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m4000_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m4000_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m4000_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m4000_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m4000_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m4000_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m4000_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m4000_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m4000_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m4200_file)
m4200_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length29)):
    m4200_tree.GetEntry(i)
    m4200_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m4200_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m4200_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m4200_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m4200_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m4200_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m4200_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m4200_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m4200_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m4200_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m4200_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m4200_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m4400_file)
m4400_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length30)):
    m4400_tree.GetEntry(i)
    m4400_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m4400_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m4400_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m4400_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m4400_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m4400_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m4400_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m4400_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m4400_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m4400_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m4400_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m4400_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m4600_file)
m4600_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length31)):
    m4600_tree.GetEntry(i)
    m4600_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m4600_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m4600_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m4600_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m4600_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m4600_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m4600_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m4600_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m4600_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m4600_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m4600_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m4600_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m4800_file)
m4800_tree = test.Get("FlatSubstructureJetTree;2")
bar = progressbar.ProgressBar()


for i in bar(xrange(length32)):
    m4800_tree.GetEntry(i)
    m4800_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m4800_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m4800_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m4800_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m4800_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m4800_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m4800_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m4800_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m4800_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m4800_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m4800_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m4800_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

test = ROOT.TFile.Open(m5000_file)
m5000_tree = test.Get("FlatSubstructureJetTree;6")
bar = progressbar.ProgressBar()


for i in bar(xrange(length33)):
    m5000_tree.GetEntry(i)
    m5000_tree.SetBranchAddress("clus_pt", cluster_pt_entry)
    m5000_tree.SetBranchAddress("clus_phi", cluster_phi_entry)
    m5000_tree.SetBranchAddress("clus_eta", cluster_eta_entry)
    m5000_tree.SetBranchAddress("clus_E", cluster_E_entry)
    m5000_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    m5000_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    m5000_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    m5000_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    m5000_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    m5000_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    m5000_tree.SetBranchAddress("fjet_fatjet_dRmatched_particle_flavor", fjet_fatjet_dRmatched_particle_flavor_entry)
    m5000_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()

fnew.Write()
fnew.Close()


new_pt = rtnp.root2array(out_storage+out_file, treename = treename, branches = "fjet_pt")

plt.figure()
plt.hist(new_pt/1000, 100, normed = True)
plt.xlabel('Jet Pt (GeV)')
plt.ylabel('Probability')
plt.title('Stitched Jet Pt')
plt.grid(True)
plt.savefig('Sig_stitchedpt_test.png')
