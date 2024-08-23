# lowmass_HiggsDNA
####################################################################################################
#HiggsDNA Installation (Follow the instructions here 
#####################################################################################################

https://higgs-dna.readthedocs.io/en/latest/installation.html

# Steps for validation studies 

########################################################################################################
The Z(\mu \mu \gamma) processors test the single photon from Z \rightarrow \mu \mu final state radiation (FSR). There is a  processor for ntuplizing ZmmyProcessor, a processor for generating histograms ZmmyHist, and a processor ZmmyZptHist for generating Zpt distribution to derive Zpt reweighting.

########################################################################################################
cd HiggsDNA
git pull


1. get authentication for Grid  to access the datasets remotely :

voms-proxy-init -voms cms --valid 192:00


#Replace the photon preselection file here (HiggsDNA/higgs_dna/selections/) to the new lowmass preselection : copy and paste the new file
photon_selections.py
 
#modify the base.py(HiggsDNA/higgs_dna/workflows/base.py) file using the pT cuts for low mass below: 
2. diphoton preselection cuts                                                                                                                                                                               
self.min_pt_photon = 18.0
self.min_pt_lead_photon = 30.0

####################################################################################################

without applying the normalizing flow correction to photons :
####################################################################################################


3. to produce zmmy ntuples before applying shower shape and isolation corrections to photons.

#Processing the files
python ../scripts/run_analysis.py --json-analysis (path to runner.json file) --dump (path to parquet file processing location) --executor futures --save (save it in coffea file)


####################################################################################################
                                                                               
   with applying the normalizing flow correction to photons :
####################################################################################################
4. to produce zmmy ntuples after  applying shower shape and isolation corrections to photons.

python ../scripts/run_analysis.py --json-analysis (path to runner.json file) --doFlow_corrections --dump (path to parquet file processing location) --executor futures --save (save it in coffea file)
####################################################################################################


All the above steps produce the zmmy ntuples in parquet files found in the output folder created in the end.
You can merge the output parquet files inside the output_folder into a single parquet file for analysis using the script below.

5. python merge_parquet.py -i output_folder/ -o output.parquet

Then also the final parquet file can be converted to .root format using the script below.

6. python parquet_converter2.py -i output.parquet -o output.root

####################################################################################################
After producing the ntuples from Zmmy processor, We need to apply low mass photon preselections.
####################################################################################################

