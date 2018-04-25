import os, sys
import numpy as np
import pandas as pd
import ROOT as r
from ROOT import TTree, TObjArray, TCanvas, TPad, TFile, TPaveText
import json
from collections import OrderedDict
from multiprocessing import Pool
from memory_profiler import profile
from root_pandas import read_root
import rootpy
from rootpy.io import root_open
from tqdm import tqdm, trange
from collections import defaultdict

selection = sys.argv[1]


def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)
    
    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')


def fill_lepton_vars(tree, name, SF):

    out_dict = {}
    # o. Filling Event Info
    #out_dict['tauMVA']       =  tree.tauMVA
    if selection == 'emu':
        out_dict['runNumber']    =  tree.runNumber
        out_dict['evtNumber']    =  tree.evtNumber

    out_dict['nMuons']       =  tree.nMuons
    out_dict['nElectrons']   =  tree.nElectrons
    out_dict['nJets']        =  tree.nJets
    out_dict['nBJets']       =  tree.nBJets
    out_dict['nPV']          =  tree.nPV

    weight = SF * tree.eventWeight
    if name in ['zjets_m-50','zjets_m-10to50']:
        if 0< tree.nPartons < 5:
            weight  = 0
    out_dict['eventWeight']  =  weight
    out_dict['eventWeightSF']=  SF

    out_dict['met']          =  tree.met
    out_dict['metPhi']       =  tree.metPhi
    out_dict['genCategory']  =  tree.genCategory
    
    # 1. Filling leptons
    lep1 = tree.leptonOneP4
    out_dict['lepton1_flavor']  = tree.leptonOneFlavor
    out_dict['lepton1_q']       = np.sign(tree.leptonOneFlavor)
    out_dict['lepton1_iso']     = tree.leptonOneIso
    out_dict['lepton1_reliso']  = tree.leptonOneIso/lep1.Pt()
    out_dict['lepton1_mother']  = tree.leptonOneMother
    out_dict['lepton1_d0']      = abs(tree.leptonOneD0)
    out_dict['lepton1_dz']      = abs(tree.leptonOneDZ)
    out_dict['lepton1_pt']      = lep1.Pt()
    out_dict['lepton1_eta']     = lep1.Eta()
    out_dict['lepton1_phi']     = lep1.Phi()
    out_dict['lepton1_mt']      = (2*lep1.Pt()*tree.met*(1-np.cos(lep1.Phi()-tree.metPhi )))**0.5
    out_dict['lepton1_energy']  = lep1.Energy()
        
    if selection in ['e4j','mu4j']:
        jet1, jet2, jet3, jet4 = tree.jetOneP4, tree.jetTwoP4, tree.jetThreeP4, tree.jetFourP4
        tag1, tag2, tag3 ,tag4 = tree.jetOneTag, tree.jetTwoTag, tree.jetThreeTag, tree.jetFourTag   
        
        lightjets = np.argsort([tag1,tag2,tag3,tag4])[:2]
        if (0 in lightjets) and (1 in lightjets):
            jet_b1= jet3
            jet_b2= jet4
            dijet = jet1+jet2
        if (0 in lightjets) and (2 in lightjets):
            jet_b1= jet2
            jet_b2= jet4
            dijet = jet1+jet3
        if (0 in lightjets) and (3 in lightjets):
            jet_b1= jet2
            jet_b2= jet3
            dijet = jet1+jet4
        if (1 in lightjets) and (2 in lightjets):
            jet_b1= jet1
            jet_b2= jet4
            dijet = jet2+jet3
        if (1 in lightjets) and (3 in lightjets):
            jet_b1= jet1
            jet_b2= jet3
            dijet = jet2+jet4
        if (2 in lightjets) and (3 in lightjets):
            jet_b1= jet1
            jet_b2= jet2
            dijet = jet3+jet4

        out_dict['dijet_m']      = dijet.M()
        deltaphi_jet_b1 = abs(dijet.DeltaPhi(jet_b1))
        deltaphi_jet_b2 = abs(dijet.DeltaPhi(jet_b2))
        deltar_jet_b1 = abs(dijet.DeltaR(jet_b1))
        deltar_jet_b2 = abs(dijet.DeltaR(jet_b2))
        trijet_mass1 = (dijet+jet_b1).M()
        trijet_mass2 = (dijet+jet_b2).M()

        if deltar_jet_b1>deltar_jet_b2:
            deltar_jet_b1,deltar_jet_b2     = deltar_jet_b2,  deltar_jet_b1
            deltaphi_jet_b1,deltaphi_jet_b2 = deltaphi_jet_b2,deltaphi_jet_b1
            trijet_mass1,   trijet_mass2    = trijet_mass2,   trijet_mass1

        out_dict['deltaphi_jet_b1'] = deltaphi_jet_b1
        out_dict['deltaphi_jet_b2'] = deltaphi_jet_b2
        out_dict['deltar_jet_b1'] = deltar_jet_b1
        out_dict['deltar_jet_b2'] = deltar_jet_b2
        out_dict['trijet_mass1'] = trijet_mass1
        out_dict['trijet_mass2'] = trijet_mass2


        
        # 2. Filling Jets
        out_dict['jet1_pt']      = jet1.Pt()
        out_dict['jet1_eta']     = jet1.Eta()
        #out_dict['jet1_phi']     = jet1.Phi()
        #out_dict['jet1_energy']  = jet1.Energy()
        
        out_dict['jet2_pt']      = jet2.Pt()
        out_dict['jet2_eta']     = jet2.Eta()
        #out_dict['jet2_phi']     = jet2.Phi()
        #out_dict['jet2_energy']  = jet2.Energy()
        
        out_dict['jet3_pt']      = jet3.Pt()
        out_dict['jet3_eta']     = jet3.Eta()
        #out_dict['jet3_phi']     = jet3.Phi()
        #out_dict['jet3_energy']  = jet3.Energy()
        
        out_dict['jet4_pt']      = jet4.Pt()
        out_dict['jet4_eta']     = jet4.Eta()
        #out_dict['jet4_phi']     = jet4.Phi()
        #out_dict['jet4_energy']  = jet4.Energy()

    else:
        # lepton2
        lep2 = tree.leptonTwoP4

        out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
        out_dict['lepton2_q']       = np.sign(tree.leptonTwoFlavor)
        out_dict['lepton2_iso']     = tree.leptonTwoIso
        out_dict['lepton2_reliso']  = tree.leptonTwoIso/lep2.Pt()
        out_dict['lepton2_mother']  = tree.leptonTwoMother
        out_dict['lepton2_d0']      = abs(tree.leptonTwoD0)
        out_dict['lepton2_dz']      = abs(tree.leptonTwoDZ)
        
        out_dict['lepton2_pt']      = lep2.Pt()
        out_dict['lepton2_eta']     = lep2.Eta()
        out_dict['lepton2_phi']     = lep2.Phi()
        out_dict['lepton2_mt']      = (2*lep2.Pt()*tree.met*(1-np.cos(lep2.Phi()-tree.metPhi )))**0.5
        out_dict['lepton2_energy']  = lep2.Energy() 

        out_dict['lepton_delta_eta']= abs(lep1.Eta() - lep2.Eta())
        out_dict['lepton_delta_phi']= abs(lep1.DeltaPhi(lep2))
        out_dict['lepton_delta_r']  = lep1.DeltaR(lep2)
        
        # dilepton
        dilepton   = lep1 + lep2

        out_dict['dilepton_mass']      = dilepton.M()
        out_dict['dilepton_pt']        = dilepton.Pt()
        out_dict['dilepton_eta']       = dilepton.Eta()
        out_dict['dilepton_phi']       = dilepton.Phi()
        out_dict['dilepton_pt_over_m'] = dilepton.Pt()/dilepton.M()
        
        
        # 2. Filling Jets
        jet1, jet2 = tree.jetOneP4,    tree.jetTwoP4

        out_dict['jet1_pt']      = jet1.Pt()
        out_dict['jet1_eta']     = jet1.Eta()
        out_dict['jet1_phi']     = jet1.Phi()
        out_dict['jet1_energy']  = jet1.Energy()
        out_dict['jet1_tag']     = tree.jetOneTag

        out_dict['jet2_pt']      = jet2.Pt()
        out_dict['jet2_eta']     = jet2.Eta()
        out_dict['jet2_phi']     = jet2.Phi()
        out_dict['jet2_energy']  = jet2.Energy()
        out_dict['jet2_tag']    = tree.jetTwoTag
        
        out_dict['jet_delta_eta']   = abs(jet1.Eta() - jet2.Eta())
        out_dict['jet_delta_phi']   = abs(jet1.DeltaPhi(jet2))
        out_dict['jet_delta_r']     = jet1.DeltaR(jet2)
        # dijet
        dijet      = jet1 + jet2
        out_dict['dijet_mass']      = dijet.M()
        out_dict['dijet_pt']        = dijet.Pt()
        out_dict['dijet_eta']       = dijet.Eta()
        out_dict['dijet_phi']       = dijet.Phi()
        out_dict['dijet_pt_over_m'] = dijet.Pt()/dijet.M()

    
    
    return out_dict



def fill_ntuple(tree, name, SF):
    n = int(tree.GetEntriesFast())

    for i in trange(n,
                    desc       = selection+' selection *** '+name,
                    leave      = True,
                    unit_scale = True,
                    ncols      = 100,
                    total      = n
                    ):
        tree.GetEntry(i)
        entry = {}
        entry.update(fill_lepton_vars(tree, name, SF))
        n -= 1
        yield entry

def pickle_ntuple(ntuple_data):
    # unpack input data
    name        = ntuple_data[0]
    SF          = ntuple_data[1]
    input_file  = ntuple_data[2]
    output_path = ntuple_data[3]
    if name in datalist:
        output_path +="data2016/"
    elif name in mcdibosonlist:
        output_path +="mcdiboson/"
    elif name in mcdylist:
        output_path +="mcdy/"
    elif name in mctlist:
        output_path +="mct/"
    elif name in mcttlist:
        output_path +="mctt/"    
    make_directory(output_path, clear=False)
    
    # get the tree, convert to dataframe, and save df to pickle
    froot  = TFile(input_file)
    tree   = froot.Get('{}/bltTree_{}'.format(selection,name))

    if tree.GetEntriesFast() >0:

        ntuple = fill_ntuple(tree, name, SF)
        df     = pd.DataFrame(ntuple)
        df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, name))


    
    
###########################################    
# My Main function
input_root_file  = "/home/zchen/Documents/Analysis/workplace/data/root/2016MC.root"
output_directory = "/home/zchen/Documents/Analysis/workplace/data/pickle/{}/".format(selection)

if selection in ["mumu","mutau","mu4j"]:
    datalist  = ['muon_2016B', 'muon_2016C', 
                'muon_2016D','muon_2016E','muon_2016F','muon_2016G','muon_2016H']

elif selection in ["ee","etau","e4j"]:
    datalist  = ['electron_2016B', 'electron_2016C', 'electron_2016D','electron_2016E',
                 'electron_2016F','electron_2016G','electron_2016H']

elif selection in ["emu"]:
    datalist  = ['muon_2016B', 'muon_2016C', 
                'muon_2016D','muon_2016E','muon_2016F','muon_2016G','muon_2016H',
                'electron_2016B', 'electron_2016C', 'electron_2016D','electron_2016E',
                'electron_2016F','electron_2016G','electron_2016H']

else:
    print( "data2016 for this selection is not defined")
    datalist  = []
    
mcdibosonlist = ['ww','wz_2l2q','wz_3lnu','zz_2l2nu','zz_2l2q','zz_4l' ]

mcdylist      = ['z1jets_m-10to50','z2jets_m-10to50','z3jets_m-10to50','z4jets_m-10to50','zjets_m-10to50',

                 'z1jets_m-50','z2jets_m-50','z3jets_m-50','z4jets_m-50','zjets_m-50',

                 'w1jets','w2jets','w3jets','w4jets']
mctlist       = ['t_tw','tbar_tw']
mcttlist      = ['ttbar_inclusive']
dataset_list  = datalist + mcdibosonlist + mcdylist + mctlist + mcttlist

SF_table ={ 'ww_2l2nu'        :  0.219,
            'wz_2l2q'         :  0.008,
            'wz_3lnu'         :  0.080,
            'zz_2l2nu'        :  0.002,
            'zz_2l2q'         :  0.008,
            'zz_4l'           :  0.004,
           

           'zjets_m-10to50'   : 18.935,
           'z1jets_m-10to50'  : 0.778,
           'z2jets_m-10to50'  : 0.846,
           'z3jets_m-10to50'  : 0.813,
           'z4jets_m-10to50'  : 0.871,

           'zjets_m-50'       :  4.227,
           'z1jets_m-50'      :  0.689,
           'z2jets_m-50'      :  0.709,
           'z3jets_m-50'      :  0.741,
           'z4jets_m-50'      :  0.550,

           'w1jets'           : 17.476,
           'w2jets'           :  3.734,
           'w3jets'           :  1.709,
           'w4jets'           :  2.068,
           
           't_tw'             :  1.29739,
           'tbar_tw'          :  1.30265,
           'ttbar_inclusive'  :  0.19244,

           'qcd_ht100to200'   :12436.1,
           'qcd_ht200to300'   :1138.16,
           'qcd_ht300to500'   :277.387,
           'qcd_ht500to700'   : 21.341,
           'qcd_ht700to1000'  :  5.703,
           'qcd_ht1000to1500' :  3.333,
           'qcd_ht1500to2000' :  0.371,
           'qcd_ht2000'       :  0.154,

          }
SF_table = defaultdict(lambda: 1.0, SF_table)

for dataset in dataset_list:
    SF = SF_table[dataset]
    if not dataset in datalist:
        SF = SF * (35.864/35.9)
    if ('tau' in selection) & (dataset not in datalist):
        SF = 1.0 * SF 
    root2df_config = [dataset,SF,input_root_file,output_directory]
    pickle_ntuple(root2df_config)


#out_dict['lepton1_q']       = tree.leptonOneQ
#out_dict['lepton1_iso']     = tree.leptonOneIso
#out_dict['lepton1_flavor']  = tree.leptonOneFlavor
#out_dict['lepton1_trigger'] = tree.leptonOneTrigger
#out_dict['lepton2_q']       = tree.leptonTwoQ
#out_dict['lepton2_iso']     = tree.leptonTwoIso
#out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
#out_dict['lepton2_trigger'] = tree.leptonTwoTrigger
