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
bkg_storage_all = "/mnt/storage/lborgna/BackgroundStitch/Testing/"
JZ3W_file = bkg_storage_all+"JZ3W.root"
JZ4W_file = bkg_storage_all+"JZ4W.root"
JZ5W_file = bkg_storage_all+"JZ5W.root"
JZ6W_file = bkg_storage_all+"JZ6W.root"
JZ7W_file = bkg_storage_all+"JZ7W.root"
JZ8W_file = bkg_storage_all+"JZ8W.root"
treename = "FlatSubstructureJetTree"


pt1 = rtnp.root2array(JZ3W_file, treename = treename, branches = "fjet_pt")
pt2 = rtnp.root2array(JZ4W_file, treename = treename, branches = "fjet_pt")
pt3 = rtnp.root2array(JZ5W_file, treename = treename, branches = "fjet_pt")
pt4 = rtnp.root2array(JZ6W_file, treename = treename, branches = "fjet_pt")
pt5 = rtnp.root2array(JZ7W_file, treename = treename, branches = "fjet_pt")
pt6 = rtnp.root2array(JZ8W_file, treename = treename, branches = "fjet_pt")
length1 = len(pt1)
length2 = len(pt2)
length3 = len(pt3)
length4 = len(pt4)
length5 = len(pt5)
length6 = len(pt6)

#Creating New ROOT file via Update Method
fnew = ROOT.TFile("/mnt/storage/lborgna/BackgroundStitch/Testing/BkgAll2_Test.root","update")
Tree = ROOT.TTree("FlatSubstructureJetTree", "Reconst ntuple")

#Gets the old file
test = ROOT.TFile.Open(JZ3W_file)
old_tree = test.Get("FlatSubstructureJetTree")

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

fjet_dRmatched_maxEParton_flavor_entry = array('i', [0])

#New Branches!
Tree.Branch('fjet_pt', fjet_pt_entry, 'fjet_pt/F')
Tree.Branch('fjet_eta', fjet_feta_entry, 'fjet_eta/F')
Tree.Branch('fjet_phi', fjet_fphi_entry, 'fjet_phi/F')
Tree.Branch('fjet_E', fjet_fE_entry, 'fjet_E/F')
Tree.Branch('fjet_Tau21_wta', fjet_Tau21_wta_entry,'fjet_Tau21_wta/F')
Tree.Branch('fjet_D2', fjet_D2_entry, 'fjet_D2/F')
Tree.Branch('fjet_truthJet_pt', fjet_truthJet_pt_entry, 'fjet_truthJet_pt/F')

Tree.Branch('fjet_dRmatched_maxEParton_flavor',fjet_dRmatched_maxEParton_flavor_entry,'fjet_dRmatched_maxEParton_flavor/I')
Tree.Branch('clus_E', cluster_E_entry)
Tree.Branch('clus_eta', cluster_eta_entry)
Tree.Branch('clus_phi', cluster_phi_entry)
Tree.Branch('clus_pt', cluster_pt_entry)

tosscount = 0

bar = progressbar.ProgressBar()

QCD_pt_new  = []

for i in bar(xrange(length1)):
    old_tree.GetEntry(i)
    old_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    old_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    old_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    old_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    old_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    old_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    old_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    old_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    old_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    old_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    old_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    old_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)
    Tree.Fill()


test = ROOT.TFile.Open(JZ4W_file)
JZ4W_tree = test.Get("FlatSubstructureJetTree")

bar = progressbar.ProgressBar()

for i in bar(xrange(length2)):
    JZ4W_tree.GetEntry(i)
    JZ4W_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    JZ4W_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    JZ4W_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    JZ4W_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    JZ4W_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    JZ4W_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    JZ4W_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    JZ4W_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    JZ4W_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    JZ4W_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    JZ4W_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    JZ4W_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(JZ5W_file)
JZ5W_tree = test.Get("FlatSubstructureJetTree")

bar = progressbar.ProgressBar()

for i in bar(xrange(length3)):
    JZ5W_tree.GetEntry(i)
    JZ5W_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    JZ5W_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    JZ5W_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    JZ5W_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    JZ5W_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    JZ5W_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    JZ5W_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    JZ5W_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    JZ5W_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    JZ5W_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    JZ5W_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    JZ5W_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(JZ6W_file)
JZ6W_tree = test.Get("FlatSubstructureJetTree")

bar = progressbar.ProgressBar()

for i in bar(xrange(length4)):
    JZ6W_tree.GetEntry(i)
    JZ6W_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    JZ6W_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    JZ6W_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    JZ6W_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    JZ6W_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    JZ6W_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    JZ6W_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    JZ6W_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    JZ6W_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    JZ6W_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    JZ6W_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    JZ6W_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(JZ7W_file)
JZ7W_tree = test.Get("FlatSubstructureJetTree")

bar = progressbar.ProgressBar()

for i in bar(xrange(length5)):
    JZ7W_tree.GetEntry(i)
    JZ7W_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    JZ7W_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    JZ7W_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    JZ7W_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    JZ7W_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    JZ7W_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    JZ7W_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    JZ7W_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    JZ7W_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    JZ7W_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    JZ7W_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    JZ7W_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()


test = ROOT.TFile.Open(JZ8W_file)
JZ8W_tree = test.Get("FlatSubstructureJetTree")

bar = progressbar.ProgressBar()

for i in bar(xrange(length6)):
    JZ8W_tree.GetEntry(i)
    JZ8W_tree.SetBranchAddress("fjet_clus_pt", cluster_pt_entry)
    JZ8W_tree.SetBranchAddress("fjet_clus_phi", cluster_phi_entry)
    JZ8W_tree.SetBranchAddress("fjet_clus_eta", cluster_eta_entry)
    JZ8W_tree.SetBranchAddress("fjet_clus_E", cluster_E_entry)
    JZ8W_tree.SetBranchAddress("fjet_pt", fjet_pt_entry)
    JZ8W_tree.SetBranchAddress("fjet_eta", fjet_feta_entry)
    JZ8W_tree.SetBranchAddress("fjet_phi", fjet_fphi_entry)
    JZ8W_tree.SetBranchAddress("fjet_E", fjet_fE_entry)
    JZ8W_tree.SetBranchAddress("fjet_D2", fjet_D2_entry)
    JZ8W_tree.SetBranchAddress("fjet_Tau21_wta", fjet_Tau21_wta_entry)
    JZ8W_tree.SetBranchAddress("fjet_dRmatched_maxEParton_flavor",fjet_dRmatched_maxEParton_flavor_entry )
    JZ8W_tree.SetBranchAddress("fjet_truthJet_pt", fjet_truthJet_pt_entry)

    Tree.Fill()



fnew.Write()
fnew.Close()


new_pt = rtnp.root2array("/mnt/storage/lborgna/BackgroundStitch/Testing/BkgAll2_Test.root", treename = treename, branches = "fjet_pt")

plt.figure()
plt.hist(new_pt/1000, 100, normed = True)
plt.xlabel('Jet Pt (GeV)')
plt.ylabel('Probability')
plt.title('Stitched Jet Pt')
plt.grid(True)
plt.savefig('stitchedpt_training.png')
