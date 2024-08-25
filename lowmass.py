from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.selections.photon_selections_lowmass import photon_preselection_lowmass
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.selections.Hgg import get_Hgg, Cxx

from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
    dress_branches,
)
from higgs_dna.utils.misc_utils import choose_jet
from higgs_dna.tools.flow_corrections import calculate_flow_corrections
from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution
from higgs_dna.tools.diphoton_mva_lowmass import (
    add_diphoton_mva_inputs_for_lowmass,
    eval_diphoton_mva_for_lowmass,
)

# from higgs_dna.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import functools
import operator
import warnings
from typing import Any, Dict, List, Optional
import awkward
import awkward as ak
import numpy
import sys
import vector
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import candidate
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


def get_fiducial_mask(diphotons, fiducial_cut):
    if fiducial_cut == "classical":
        fid_det_passed = (
            (diphotons.pho_lead.pt / diphotons.mass > 1 / 3)
            & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4)
            & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10)
            & (
                (
                    diphotons.pho_sublead.pfRelIso03_all_quadratic
                    * diphotons.pho_sublead.pt
                )
                < 10
            )
            & (numpy.abs(diphotons.pho_lead.eta) < 2.5)
            & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
        )
    elif fiducial_cut == "geometric":
        fid_det_passed = (
            (
                numpy.sqrt(diphotons.pho_lead.pt * diphotons.pho_sublead.pt)
                / diphotons.mass
                > 1 / 3
            )
            & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4)
            & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10)
            & (
                diphotons.pho_sublead.pfRelIso03_all_quadratic
                * diphotons.pho_sublead.pt
                < 10
            )
            & (numpy.abs(diphotons.pho_lead.eta) < 2.5)
            & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
        )
    elif fiducial_cut == "none":
        fid_det_passed = (
            diphotons.pho_lead.pt > -10
        )  # This is a very dummy way but I do not know how to make a true array of outer shape of diphotons
    else:
        warnings.warn(
            "You chose %s the fiducialCuts mode, but this is currently not supported. You should check your settings. For this run, no fiducial selection at detector level is applied."
            % fiducial_cut
        )
        fid_det_passed = diphotons.pho_lead.pt > -10

    return fid_det_passed


def get_mass_resolution_uncertainty(diphotons, mc_flow_corrected=True):
    if mc_flow_corrected:
        diphotons["sigma_m_over_m"] = 0.5 * numpy.sqrt(
            (
                diphotons["pho_lead"].raw_energyErr
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                diphotons["pho_sublead"].raw_energyErr
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

        diphotons["sigma_m_over_m_corr"] = 0.5 * numpy.sqrt(
            (
                diphotons["pho_lead"].energyErr
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                diphotons["pho_sublead"].energyErr
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

    else:
        diphotons["sigma_m_over_m"] = 0.5 * numpy.sqrt(
            (
                diphotons["pho_lead"].energyErr
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                diphotons["pho_sublead"].energyErr
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

    return diphotons


def get_mass_resolution_smearing(diphotons, mc_flow_corrected=True):
    if mc_flow_corrected:
        # Adding the smeared BDT error to the ntuples!
        diphotons["pho_lead", "energyErr_Smeared"] = numpy.sqrt(
            (diphotons["pho_lead"].raw_energyErr) ** 2
            + (
                diphotons["pho_lead"].rho_smear
                * ((diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta)))
            )
            ** 2
        )
        diphotons["pho_sublead", "energyErr_Smeared"] = numpy.sqrt(
            (diphotons["pho_sublead"].raw_energyErr) ** 2
            + (
                diphotons["pho_sublead"].rho_smear
                * (
                    (
                        diphotons["pho_sublead"].pt
                        * numpy.cosh(diphotons["pho_sublead"].eta)
                    )
                )
            )
            ** 2
        )

        diphotons["sigma_m_over_m_Smeared"] = 0.5 * numpy.sqrt(
            (
                numpy.sqrt(
                    (diphotons["pho_lead"].raw_energyErr) ** 2
                    + (
                        diphotons["pho_lead"].rho_smear
                        * (
                            (
                                diphotons["pho_lead"].pt
                                * numpy.cosh(diphotons["pho_lead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                numpy.sqrt(
                    (diphotons["pho_sublead"].raw_energyErr) ** 2
                    + (
                        diphotons["pho_sublead"].rho_smear
                        * (
                            (
                                diphotons["pho_sublead"].pt
                                * numpy.cosh(diphotons["pho_sublead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

        diphotons["sigma_m_over_m_Smeared_corr"] = 0.5 * numpy.sqrt(
            (
                numpy.sqrt(
                    (diphotons["pho_lead"].energyErr) ** 2
                    + (
                        diphotons["pho_lead"].rho_smear
                        * (
                            (
                                diphotons["pho_lead"].pt
                                * numpy.cosh(diphotons["pho_lead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                numpy.sqrt(
                    (diphotons["pho_sublead"].energyErr) ** 2
                    + (
                        diphotons["pho_sublead"].rho_smear
                        * (
                            (
                                diphotons["pho_sublead"].pt
                                * numpy.cosh(diphotons["pho_sublead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

    else:
        # Adding the smeared BDT error to the ntuples!
        diphotons["pho_lead", "energyErr_Smeared"] = numpy.sqrt(
            (diphotons["pho_lead"].energyErr) ** 2
            + (
                diphotons["pho_lead"].rho_smear
                * ((diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta)))
            )
            ** 2
        )
        diphotons["pho_sublead", "energyErr_Smeared"] = numpy.sqrt(
            (diphotons["pho_sublead"].energyErr) ** 2
            + (
                diphotons["pho_sublead"].rho_smear
                * (
                    (
                        diphotons["pho_sublead"].pt
                        * numpy.cosh(diphotons["pho_sublead"].eta)
                    )
                )
            )
            ** 2
        )

        diphotons["sigma_m_over_m_Smeared"] = 0.5 * numpy.sqrt(
            (
                numpy.sqrt(
                    (diphotons["pho_lead"].energyErr) ** 2
                    + (
                        diphotons["pho_lead"].rho_smear
                        * (
                            (
                                diphotons["pho_lead"].pt
                                * numpy.cosh(diphotons["pho_lead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (diphotons["pho_lead"].pt * numpy.cosh(diphotons["pho_lead"].eta))
            )
            ** 2
            + (
                numpy.sqrt(
                    (diphotons["pho_sublead"].energyErr) ** 2
                    + (
                        diphotons["pho_sublead"].rho_smear
                        * (
                            (
                                diphotons["pho_sublead"].pt
                                * numpy.cosh(diphotons["pho_sublead"].eta)
                            )
                        )
                    )
                    ** 2
                )
                / (
                    diphotons["pho_sublead"].pt
                    * numpy.cosh(diphotons["pho_sublead"].eta)
                )
            )
            ** 2
        )

    return diphotons

class lowmassProcessor(HggBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "lowMassAnalysis",
        skipCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "none",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet",
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            output_location=output_location,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            skipCQR=skipCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format,
        )
        # diphoton preselection cuts
        self.min_pt_photon = 18.0
        self.min_pt_lead_photon = 30.0
        self.e_veto = "presel"  # presel/single_invert/double_invert
        self.trigger_group = ".*DoubleEG.*"
        self.analysis = "lowMassAnalysis"

    def process_extra(self, events: awkward.Array) -> awkward.Array:
        return events, {}

    def apply_filters(self, events: awkward.Array) -> awkward.Array:
        # met filters
        met_filters = self.meta["flashggMetFilters"][self.data_kind]
        filtered = functools.reduce(
            operator.and_,
            (events.Flag[metfilter.split("_")[-1]] for metfilter in met_filters),
        )

        return events[filtered]

    def apply_triggers(
        self, events: awkward.Array, apply_to_mc: bool = False
    ) -> awkward.Array:
        # trigger selection
        logger.debug(
            f"[apply_triggers] {self.trigger_group} {self.analysis} {self.data_kind} {apply_to_mc}"
        )
        triggered = awkward.ones_like(events.event)

        if self.apply_trigger:
            if not apply_to_mc and self.data_kind == "mc":
                return events
            else:
                trigger_names = []
                triggers = self.meta["TriggerPaths"][self.trigger_group][self.analysis]
                hlt = events.HLT
                for trigger in triggers:
                    actual_trigger = trigger.replace("HLT_", "").replace("*", "")
                    for field in hlt.fields:
                        if field.startswith(actual_trigger):
                            trigger_names.append(field)
                triggered = functools.reduce(
                    operator.or_, (hlt[trigger_name] for trigger_name in trigger_names)
                )
                return events[triggered]

        return events

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                awkward.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(awkward.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(awkward.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                awkward.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap
        if not self.skipJetVetoMap:
            events = jetvetomap(
                events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(awkward.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters(events)
        # ! by default, trigger is not applied to MC
        # ! if need to be applied, change apply_to_mc to True
        events = self.apply_triggers(events, apply_to_mc=False)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, which is not present in NanoAODv11 but can be calculated using PV
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # If --Smear_sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt scpectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and "Smearing" not in correction_names
        ):
            warnings.warn(
                "Smearing should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # Since now we are applying Smearing term to the sigma_m_over_m i added this portion of code
        # specially for the estimation of smearing terms for the data events [data pt/energy] are not smeared!
        if self.data_kind == "data" and self.Smear_sigma_m:
            correction_name = "Smearing"

            logger.info(
                f"\nApplying correction {correction_name} to dataset {dataset_name}\n"
            )
            varying_function = available_object_corrections[correction_name]
            events = varying_function(events=events, year=self.year[dataset_name][0])

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # Computing the normalizing flow correction
        if self.data_kind == "mc" and self.doFlow_corrections:

            # Applyting the Flow corrections to all photons before pre-selection
            counts = awkward.num(original_photons)
            corrected_inputs, var_list = calculate_flow_corrections(
                original_photons,
                events,
                self.meta["flashggPhotons"]["flow_inputs"],
                self.meta["flashggPhotons"]["Isolation_transform_order"],
                year=self.year[dataset_name][0],
            )

            # Store the raw nanoAOD value and update photon ID MVA value for preselection
            original_photons["mvaID_nano"] = original_photons["mvaID"]

            # Store the raw values of the inputs and update the input values with the corrections since some variables used in the preselection
            for i in range(len(var_list)):
                original_photons["raw_" + str(var_list[i])] = original_photons[
                    str(var_list[i])
                ]
                original_photons[str(var_list[i])] = awkward.unflatten(
                    corrected_inputs[:, i], counts
                )

            original_photons["mvaID"] = awkward.unflatten(
                self.add_photonid_mva_run3(original_photons, events), counts
            )

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        ),
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(
            variations_combined, []
        )  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [
            item + suffix
            for item in variations_flattened
            for suffix in ["_down", "_up"]
        ]
        # Add nominal to the list
        variations.append("nominal")
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [
                *photons_dct
            ]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = (
                variation  # We can also simplify this a bit but for now it works
            )

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)
            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            photons = photon_preselection_lowmass(
                self, photons, events, year=self.year[dataset_name][0]
            )

            # sort photons in each event descending in pt
            # make descending-pt combinations of photons
            photons = photons[awkward.argsort(photons.pt, ascending=False)]
            photons["charge"] = awkward.zeros_like(
                photons.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            diphotons = awkward.combinations(
                photons, 2, fields=["pho_lead", "pho_sublead"]
            )
            # ! only keep both pass/single pass/none pass pairs
            # presel: both photons don't have pixelSeed
            # single_invert: one photon has pixelSeed, another doesn't have pixelSeed
            # double_invert: both photons have pixelSeed
            if self.e_veto == "presel":
                diphotons = diphotons[
                    (diphotons["pho_lead"].pixelSeed < 0.5)
                    & (diphotons["pho_sublead"].pixelSeed < 0.5)
                ]
            elif self.e_veto == "single_invert":
                diphotons = diphotons[
                    (
                        (diphotons["pho_lead"].pixelSeed > 0.5)
                        & (diphotons["pho_sublead"].pixelSeed < 0.5)
                    )
                    | (
                        (diphotons["pho_lead"].pixelSeed < 0.5)
                        & (diphotons["pho_sublead"].pixelSeed > 0.5)
                    )
                ]
            elif self.e_veto == "double_invert":
                diphotons = diphotons[
                    (diphotons["pho_lead"].pixelSeed > 0.5)
                    & (diphotons["pho_sublead"].pixelSeed > 0.5)
                ]
            else:
                logger.error(
                    f"[lowmass processor] '{self.e_veto}' is not allowed, please use presel/single_invert/double_invert"
                )
            # the remaining cut is to select the leading photons
            # the previous sort assures the order
            diphotons = diphotons[diphotons["pho_lead"].pt > self.min_pt_lead_photon]

            # now turn the diphotons into candidates with four momenta and such
            diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
            diphotons["pt"] = diphoton_4mom.pt
            diphotons["eta"] = diphoton_4mom.eta
            diphotons["phi"] = diphoton_4mom.phi
            diphotons["mass"] = diphoton_4mom.mass
            diphotons["charge"] = diphoton_4mom.charge

            diphoton_pz = diphoton_4mom.z
            diphoton_e = diphoton_4mom.energy

            diphotons["rapidity"] = 0.5 * numpy.log(
                (diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz)
            )

            diphotons = awkward.with_name(diphotons, "PtEtaPhiMCandidate")

            # sort diphotons by pT
            diphotons = diphotons[awkward.argsort(diphotons.pt, ascending=False)]

            # Determine if event passes fiducial Hgg cuts at detector-level
            # fid_det_passed = get_fiducial_mask(diphotons, self.fiducialCuts)
            # diphotons = diphotons[fid_det_passed]
            #
            # if self.data_kind == "mc":

            #     # Add the fiducial flags for particle level
            #     diphotons["fiducialClassicalFlag"] = get_fiducial_flag(
            #         events, flavour="Classical"
            #     )
            #     diphotons["fiducialGeometricFlag"] = get_fiducial_flag(
            #         events, flavour="Geometric"
            #     )

            #     diphotons["PTH"], diphotons["YH"] = get_higgs_gen_attributes(events)

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            # jet_variables
            jets = awkward.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": awkward.zeros_like(
                        jets.pt
                    ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                    "hFlav": (
                        jets.hadronFlavour
                        if self.data_kind == "mc"
                        else awkward.zeros_like(jets.pt)
                    ),
                    "btagDeepFlav_B": jets.btagDeepFlavB,
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                }
            )
            jets = awkward.with_name(jets, "PtEtaPhiMCandidate")

            electrons = awkward.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = awkward.with_name(electrons, "PtEtaPhiMCandidate")

            muons = awkward.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                }
            )
            muons = awkward.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[select_electrons(self, electrons, diphotons)]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[select_jets(self, jets, diphotons, sel_muons, sel_electrons)]
            jets = jets[awkward.argsort(jets.pt, ascending=False)]

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets
            n_jets = awkward.num(jets)
            Njets2p5 = awkward.num(jets[(jets.pt > 20) & (numpy.abs(jets.eta) < 2.5)])
            #num_jets = 2
            #jet_properties = ["pt", "eta", "phi", "mass", "charge"]
            #for i in range(num_jets):
                #for prop in jet_properties:
                    #key = f"jet{i+1}_{prop}"
                    #value = choose_jet(getattr(jets, prop), i, -999.0)
                            # Store the value in the diphotons dictionary
                    #diphotons[key] = value
            #diphotons["n_jets"] = n_jets
            
            

            dijets = awkward.combinations(
                jets, 2, fields=("first_jet", "second_jet")
            )
            
            ## now turn the dijets into candidates with four momenta and such
            dijets_4mom = dijets["first_jet"] + dijets["second_jet"]
            dijets["pt"] = dijets_4mom.pt
            dijets["eta"] = dijets_4mom.eta
            dijets["phi"] = dijets_4mom.phi
            dijets["mass"] = dijets_4mom.mass
            dijets["charge"] = dijets_4mom.charge
            dijets["pt_sum"] = dijets["first_jet"].pt + dijets["second_jet"].pt
            dijets = awkward.with_name(dijets, "PtEtaPhiMCandidate")
            #
            ## add delta eta and delta phi of the dijet system 
            dijets["delta_eta"] = abs(dijets["first_jet"].eta - dijets["second_jet"].eta)
            dijets["delta_phi"] = dijets.first_jet.delta_phi(dijets.second_jet) # PtEtaPhiMCandidate already has a method from which you can calculate delta phi
           #Selection on Dijets
            #dijets = dijets[dijets.pt > 20]
            #dijets = dijets[(numpy.abs(dijets["first_jet"].eta) < 4.7) & (numpy.abs(dijets["second_jet"].eta) < 4.7)]
            dijets = dijets[dijets.pt_sum > 0]
            dijets = dijets[awkward.argsort(dijets.pt_sum, ascending=False)]
                   
            
            
            lead_jet_pt = choose_jet(jets.pt, 0, -999.0)
            lead_jet_eta = choose_jet(jets.eta, 0, -999.0)
            lead_jet_phi = choose_jet(jets.phi, 0, -999.0)
            lead_jet_mass = choose_jet(jets.mass, 0, -999.0)
            lead_jet_charge = choose_jet(jets.charge, 0, -999.0)

            sublead_jet_pt = choose_jet(jets.pt, 1, -999.0)
            sublead_jet_eta = choose_jet(jets.eta, 1, -999.0)
            sublead_jet_phi = choose_jet(jets.phi, 1, -999.0)
            sublead_jet_mass = choose_jet(jets.mass, 1, -999.0)
            sublead_jet_charge = choose_jet(jets.charge, 1, -999.0)
            
            dijet_pt = choose_jet(dijets.pt, 0, -999.0)
            dijet_eta = choose_jet(dijets.eta, 0, -999.0)
            dijet_phi = choose_jet(dijets.phi, 0, -999.0)
            dijet_mass = choose_jet(dijets.mass, 0, -999.0)
            dijet_charge = choose_jet(dijets.charge, 0, -999.0)
            
            diphotons["dijet_pt"] = dijet_pt
            diphotons["dijet_eta"] = dijet_eta
            diphotons["dijet_phi"] = dijet_phi
            diphotons["dijet_mass"] = dijet_mass
            diphotons["dijet_charge"] = dijet_charge



           #diphotons["lead_jet_pt"] = lead_jet_pt
            #diphotons["lead_jet_eta"] = lead_jet_eta
            #diphotons["lead_jet_phi"] = lead_jet_phi
            #diphotons["lead_jet_mass"] = lead_jet_mass
            #diphotons["lead_jet_charge"] = lead_jet_charge

            #diphotons["sublead_jet_pt"] = sublead_jet_pt
            #diphotons["sublead_jet_eta"] = sublead_jet_eta
            #diphotons["sublead_jet_phi"] = sublead_jet_phi
            #diphotons["sublead_jet_mass"] = sublead_jet_mass
            #diphotons["sublead_jet_charge"] = sublead_jet_charge

            Hgg = get_Hgg(self, diphotons, dijets)
            diphotons["VBFHggCandidate_pt"] = Hgg.obj_Hgg.pt
            diphotons["VBFHggCandidate_eta"] = Hgg.obj_Hgg.eta
            diphotons["VBFHggCandidate_phi"] = Hgg.obj_Hgg.phi
            diphotons["VBFHggCandidate_mass"] = Hgg.obj_Hgg.mass


            
            #diphotons["leadJet_PtOverM"] = diphotons["lead_jet_pt"]/diphotons["dijet_mass"]
            #diphotons["subleadJet_PtOverM"] = diphotons["sublead_jet_pt"]/diphotons["dijet_mass"]
            
            diphotons["pholead_PtOverM"] = Hgg.pho_lead.pt / Hgg.obj_diphoton.mass
            diphotons["phosublead_PtOverM"] = Hgg.pho_sublead.pt / Hgg.obj_diphoton.mass
          
            
            
            #Add VBF jets information
            Hgg = awkward.with_name(Hgg, "PtEtaPhiMCandidate", behavior=candidate.behavior)
            jets = awkward.with_name(jets, "PtEtaPhiMCandidate", behavior=candidate.behavior)
            jets["dr_VBFj_g1"] = jets.delta_r(Hgg.pho_lead)
            jets["dr_VBFj_g2"] = jets.delta_r(Hgg.pho_sublead)

            #Selection on for VBF
            vbf_jets = jets[(jets.pt > 20) & (numpy.abs(jets.eta) < 4.7) ]
            vbf_jet_pair = awkward.combinations(
                vbf_jets, 2, fields=("first_jet", "second_jet")
            )
            vbf = awkward.zip({
                "first_jet": vbf_jet_pair["0"],
                "second_jet": vbf_jet_pair["1"],
                "dijet": vbf_jet_pair["0"] + vbf_jet_pair["1"],
            })
            #vbf = vbf[(vbf.first_jet.pt > 30) & (vbf.second_jet.pt > 20) & (numpy.abs(vbf.first_jet.eta - vbf.second_jet.eta) > 3.5) ]
            vbf = vbf[(vbf.first_jet.pt > 30) & (vbf.second_jet.pt > 20)]
            #vbf = vbf[awkward.argsort((vbf.first_jet.pt + vbf.second_jet.pt) > 0, ascending=False)]
            vbf = vbf[awkward.argsort(vbf.dijet.mass, ascending=False)]
            vbf = awkward.firsts(vbf)
            vbf_jets_properties = ["pt", "eta", "phi", "mass", "charge", "nLHEPart"]
            for i in vbf.fields:
                vbf_properties = vbf_jets_properties if i != "dijet" else vbf_jets_properties[:5]
                for prop in vbf_properties:
                    key = f"VBF_{i}_{prop}"
                
                    value = awkward.fill_none(getattr(vbf[i],prop), -999.0)
                
                #store the values in diphotons dictionary
                    diphotons[key] = value
                    
            #diphotons["VBF_first_jet_PtOverM"] = awkward.where(diphotons.VBF_first_jet_pt != -999, diphotons.VBF_first_jet_pt / diphotons.VBF_dijet_mass, -999)
            #diphotons["VBF_second_jet_PtOverM"] = awkward.where(diphotons.VBF_second_jet_pt != -999, diphotons.VBF_second_jet_pt / diphotons.VBF_dijet_mass, -999)
            diphotons["VBF_jet_eta_prod"] = awkward.fill_none(vbf.first_jet.eta * vbf.second_jet.eta, -999)
            diphotons["VBF_jet_eta_diff"] = awkward.fill_none(vbf.first_jet.eta - vbf.second_jet.eta, -999)
            diphotons["VBF_jet_eta_sum"] = awkward.fill_none(vbf.first_jet.eta + vbf.second_jet.eta, -999)
            #diphotons["VBF_first_jet_pt"] = awkward.fill_none(vbf.first_jet.pt, -999)
            diphotons["VBF_first_jet_PtOverM"] = awkward.fill_none(vbf.first_jet.pt / vbf.dijet.mass, -999)
            diphotons["VBF_second_jet_PtOverM"] = awkward.fill_none(vbf.second_jet.pt / vbf.dijet.mass, -999)

            diphotons["VBF_jet_phi_diff"] = awkward.fill_none(vbf.first_jet.phi - vbf.second_jet.phi, -999)
            diphotons["VBF_dipho_dijet_phi_diff"] = awkward.fill_none(vbf.dijet.phi - Hgg.obj_diphoton.phi, -999)

            diphotons["VBF_DeltaR_j1g1"] = awkward.fill_none(vbf.first_jet.dr_VBFj_g1, -999)
            diphotons["VBF_DeltaR_j1g2"] = awkward.fill_none(vbf.first_jet.dr_VBFj_g2, -999)
            diphotons["VBF_DeltaR_j2g1"] = awkward.fill_none(vbf.second_jet.dr_VBFj_g1, -999)
            diphotons["VBF_DeltaR_j2g2"] = awkward.fill_none(vbf.second_jet.dr_VBFj_g2, -999)
            DeltaR_jg = awkward.Array([diphotons["VBF_DeltaR_j1g1"], diphotons["VBF_DeltaR_j2g1"], diphotons["VBF_DeltaR_j1g2"], diphotons["VBF_DeltaR_j2g2"]])
            diphotons["VBF_DeltaR_jg_min"] = awkward.min(DeltaR_jg, axis=0)
            diphotons["VBF_Cgg_v1"] = awkward.where(diphotons.VBF_jet_eta_diff != -999, Cxx(diphotons.eta, diphotons.VBF_jet_eta_diff, diphotons.VBF_jet_eta_sum), -999)
            diphotons["VBF_Cgg_v2"] = awkward.where(diphotons.VBF_jet_eta_diff != -999, Cxx(Hgg.obj_diphoton.eta, diphotons.VBF_jet_eta_diff, diphotons.VBF_jet_eta_sum), -999)

            

            # * add diphoton mva inputs
            if self.data_kind == "mc" and self.doFlow_corrections:
                diphotons = add_diphoton_mva_inputs_for_lowmass(
                    diphotons, events, mc_flow_corrected=True
                )
            else:
                diphotons = add_diphoton_mva_inputs_for_lowmass(
                    diphotons, events, mc_flow_corrected=False
                )

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            diphotons["_".join([tagger.name, str(tagger.priority)])]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            diphotons = awkward.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            diphotons = dress_branches(diphotons, events.PV, "PV")
            diphotons = dress_branches(diphotons, events.Rho, "Rho")
            # * evaluate diphoton mva
            # * after all selection and
            # * before map pho_lead and pho_sublead with self.prefixes
            diphotons = eval_diphoton_mva_for_lowmass(
                diphotons, year=self.year[dataset_name][0]
            )

            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = (
                    events.HTXS.njets30
                )  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(diphotons) | awkward.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~awkward.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[correction_name]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"diphotons_{do_variation}"][selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons["weight_LHEScale"] = (
                                        events.LHEScaleWeight[selection_mask]
                                    )
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons["weight_LHEPdf"] = events.LHEPdfWeight[
                                        selection_mask
                                    ][:, :-2]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask] * diphotons["weight_central"]
                )
                diphotons["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = awkward.ones_like(diphotons["event"])
                diphotons["weight"] = awkward.ones_like(diphotons["event"])

            ### Add mass resolution uncertainty
            # Note that pt*cosh(eta) is equal to the energy of a four vector
            # Note that you need to call it slightly different than in the output of HiggsDNA as pho_lead -> lead is only done in dumping utils
            if self.data_kind == "mc" and self.doFlow_corrections:
                diphotons = get_mass_resolution_uncertainty(
                    diphotons, mc_flow_corrected=True
                )
            else:
                diphotons = get_mass_resolution_uncertainty(
                    diphotons, mc_flow_corrected=False
                )

            # This is the mass SigmaM/M value including the smearing term from the Scale and smearing
            # The implementation follows the flashGG implementation -> https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/DataFormats/src/Photon.cc#L293
            # adittional flashGG link when the smearing of the SigmaE/E smearing is called -> https://github.com/cms-analysis/flashgg/blob/4edea8897e2a4b0518dca76ba6c9909c20c40ae7/Systematics/plugins/PhotonSigEoverESmearingEGMTool.cc#L83C40-L83C45
