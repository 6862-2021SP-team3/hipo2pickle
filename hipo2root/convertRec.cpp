#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include <string>
#include <iostream>
#include <vector>
#include "HipoChain.h"
#include "clas12reader.h"
#include <fstream>


using namespace std;

int main(int argc, char **argv){
    clas12root::HipoChain chain;

	char File[200];
	system("ls -1 *.hipo > dataFiles.txt");
	ifstream in("dataFiles.txt", ios::in);
	if(!in){
		cerr<< "File Not Opened!" <<endl;
		exit(1);
	}
	while( in >> File){
		cout<<" File Name = "<<File<<endl;
		chain.Add(File);	
	}
    
    TFile *rFile = TFile::Open("recwithgen.root","RECREATE");
    TTree *T=new TTree("T","Rec");

   // =====  proton =====
    Int_t nmb;
    Float_t Ppx[100];
    Float_t Ppy[100];
    Float_t Ppz[100];
    Int_t Pstat[100];
    Int_t Psector[100];

    Float_t GenPpx;
    Float_t GenPpy;
    Float_t GenPpz;
    // ==== electron =====
    Float_t Epx;
    Float_t Epy;
    Float_t Epz;
    Int_t Estat;
    Int_t Esector;
    
    Float_t GenEpx;
    Float_t GenEpy;
    Float_t GenEpz;
    
    // ==== gammas =====
    Int_t nmg;
    Float_t Gpx[100];
    Float_t Gpy[100];
    Float_t Gpz[100];
    Int_t Gstat[100];
    Int_t Gsector[100];

    Int_t nmG;
    Float_t GenGpx[2];
    Float_t GenGpy[2];
    Float_t GenGpz[2];

    Int_t PcalSector[100];
    Int_t Ftof1aSector[100];
    Int_t Ftof1bSector[100];
    Int_t Ftof2Sector[100];

   
 ///   protons ================================== 
    T->Branch("nmb",&nmb,"nmb/I");
    T->Branch("Ppx",&Ppx,"Ppx[nmb]/F");
    T->Branch("Ppy",&Ppy,"Ppy[nmb]/F");
    T->Branch("Ppz",&Ppz,"Ppz[nmb]/F");
    T->Branch("Pstat",&Pstat,"Pstat[nmb]/I");
    T->Branch("Psector",&Psector,"Psector[nmb]/I");


// ===============    Electrons ==============    
    T->Branch("Epx",&Epx,"Epx/F");
    T->Branch("Epy",&Epy,"Epy/F");
    T->Branch("Epz",&Epz,"Epz/F");
    T->Branch("Estat",&Estat,"Estat/I");
    T->Branch("Esector",&Esector,"Esector/I");

// ================   Gamma  ===============    
    T->Branch("nmg",&nmg,"nmg/I");
    T->Branch("Gpx",&Gpx,"Gpx[nmg]/F");
    T->Branch("Gpy",&Gpy,"Gpy[nmg]/F");
    T->Branch("Gpz",&Gpz,"Gpz[nmg]/F");
    T->Branch("Gstat",&Gstat,"Gstat[nmg]/I");
    T->Branch("Gsector",&Gsector,"Gsector[nmg]/I");

//=================  

// MC bank
    T->Branch("GenEpx",&GenEpx,"GenEpx/F");
    T->Branch("GenEpy",&GenEpy,"GenEpy/F");
    T->Branch("GenEpz",&GenEpz,"GenEpz/F");
    T->Branch("GenPpx",&GenPpx,"GenPpx/F");
    T->Branch("GenPpy",&GenPpy,"GenPpy/F");
    T->Branch("GenPpz",&GenPpz,"GenPpz/F");
    T->Branch("nmG", &nmG, "nmG/I");
    T->Branch("GenGpx",&GenGpx,"GenGpx[nmG]/F");
    T->Branch("GenGpy",&GenGpy,"GenGpy[nmG]/F");
    T->Branch("GenGpz",&GenGpz,"GenGpz[nmG]/F");
 
    //
    //loop over files
    //
    for(int ifile=0; ifile<chain.GetNFiles();++ifile){
        clas12::clas12reader c12{chain.GetFileName(ifile).Data()};


// main particle bank ========
    	auto idx_RECPart = c12.addBank("REC::Particle");
       	auto iPid = c12.getBankOrder(idx_RECPart,"pid");
    	auto iPx  = c12.getBankOrder(idx_RECPart,"px");
    	auto iPy  = c12.getBankOrder(idx_RECPart,"py");
       	auto iPz  = c12.getBankOrder(idx_RECPart,"pz");
        auto iStat = c12.getBankOrder(idx_RECPart,"status");
//===================


//  Filter bank created by Sangbaek
    	auto idx_FILTER = c12.addBank("FILTER::Index");
    	auto iPcalSector = c12.getBankOrder(idx_FILTER, "pcal_sector");
    	auto iFtof1aSector = c12.getBankOrder(idx_FILTER, "ftof1a_sector");
    	auto iFtof1bSector = c12.getBankOrder(idx_FILTER, "ftof1b_sector");
    	auto iFtof2Sector = c12.getBankOrder(idx_FILTER, "ftof2_sector");
//=========



// MC bank
    	auto idx_GenPart = c12.addBank("MC::Particle");
       	auto iGenPid = c12.getBankOrder(idx_GenPart,"pid");
    	auto iGenPx  = c12.getBankOrder(idx_GenPart,"px");
    	auto iGenPy  = c12.getBankOrder(idx_GenPart,"py");
       	auto iGenPz  = c12.getBankOrder(idx_GenPart,"pz");

        while(c12.next() == true){
		
	        nmb=0;
        	nmg=0;
    		nmG=0;

            for(auto ipa = 0;ipa<c12.getBank(idx_FILTER)->getRows();ipa++){
               auto tempPcalSector = c12.getBank(idx_FILTER)->getInt(iPcalSector,ipa);
               auto tempFtof1aSector = c12.getBank(idx_FILTER)->getInt(iFtof1aSector,ipa);
               auto tempFtof1bSector = c12.getBank(idx_FILTER)->getInt(iFtof1bSector,ipa);
               auto tempFtof2Sector = c12.getBank(idx_FILTER)->getInt(iFtof2Sector,ipa);
               PcalSector[ipa] = tempPcalSector;
               Ftof1aSector[ipa] = tempFtof1aSector;
               Ftof1bSector[ipa] = tempFtof1bSector;
               Ftof2Sector[ipa] = tempFtof2Sector;
            }

        	for(auto ipa=0;ipa<c12.getBank(idx_GenPart)->getRows();ipa++){
                
            	auto tGenPx = c12.getBank(idx_GenPart)->getFloat(iGenPx,ipa);
            	auto tGenPy = c12.getBank(idx_GenPart)->getFloat(iGenPy,ipa);
            	auto tGenPz = c12.getBank(idx_GenPart)->getFloat(iGenPz,ipa);
 
            	if( (c12.getBank(idx_GenPart)->getInt(iGenPid,ipa)) == 11  ){  // electrons
    				GenEpx = tGenPx;
    				GenEpy = tGenPy;
    				GenEpz = tGenPz;
            	}
                
                if((c12.getBank(idx_GenPart)->getInt(iGenPid,ipa)) == 2212  ){  // protons
            		GenPpx = tGenPx;
            		GenPpy = tGenPy;
            		GenPpz = tGenPz;
            	}
                    
            	if((c12.getBank(idx_GenPart)->getInt(iGenPid,ipa)) == 22  ){  // photons
            		GenGpx[nmG] = tGenPx;
            		GenGpy[nmG] = tGenPy;
            		GenGpz[nmG] = tGenPz;
				    nmG++;
                }
        	}


    		for(auto ipa=0;ipa<c12.getBank(idx_RECPart)->getRows();ipa++){
                
            	auto tPx = c12.getBank(idx_RECPart)->getFloat(iPx,ipa);
            	auto tPy = c12.getBank(idx_RECPart)->getFloat(iPy,ipa);
            	auto tPz = c12.getBank(idx_RECPart)->getFloat(iPz,ipa);
    			auto tStat = c12.getBank(idx_RECPart)->getInt(iStat,ipa);

            	if( (c12.getBank(idx_RECPart)->getInt(iPid,ipa)) == 11  ){  // electrons
    				Epx = tPx;
    				Epy = tPy;
    				Epz = tPz;
    				Esector = PcalSector[ipa];
            	}
                
                if((c12.getBank(idx_RECPart)->getInt(iPid,ipa)) == 2212  && tStat<4000){  // protons
         
            		Ppx[nmb] = tPx;
            		Ppy[nmb] = tPy;
            		Ppz[nmb] = tPz;
    				Pstat[nmb] = tStat;
    				if (Pstat[nmb] >4000) Psector[nmb] = Pstat[nmb];
    				else if (Ftof1aSector[ipa]>0) Psector[nmb] = Ftof1aSector[ipa];	
    				else if (Ftof1bSector[ipa]>0) Psector[nmb] = Ftof1bSector[ipa];	
    				else if (Ftof2Sector[ipa]>0) Psector[nmb] = Ftof2Sector[ipa];	

               		nmb++;
                    
            	} // if for protons
                    
            	if((c12.getBank(idx_RECPart)->getInt(iPid,ipa)) == 22  ){  // photons

            		Gpx[nmg] = tPx;
            		Gpy[nmg] = tPy;
            		Gpz[nmg] = tPz;
                    Gstat[nmg] = tStat;
    				if (Gstat[nmg]<2000) Gsector[nmg] = Gstat[nmg];
    				else Gsector[nmg] = PcalSector[ipa];
               		nmg++;
                    
            	}
            }

    		if(nmb>0 && nmg>=nmG) T->Fill();
        }

    }

	rFile->Write();
	rFile->Close();

    return 1;
}
