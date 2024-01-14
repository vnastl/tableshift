"""
Experiment configs for the 'official' TableShift benchmark tasks.

All other configs are in non_benchmark_configs.py.
"""

from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tableshift.core import Grouper, PreprocessorConfig, DomainSplitter
from tableshift.datasets import BRFSS_YEARS, ACS_YEARS, NHANES_YEARS
from tableshift.datasets import ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS_NUMBER, ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER, BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS_NUMBER, BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER, \
    DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER, DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    ANES_FEATURES_CAUSAL_SUBSETS_NUMBER, ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    ASSISTMENTS_FEATURES_CAUSAL_SUBSETS_NUMBER, ASSISTMENTS_FEATURES_CAUSAL_SUBSETS, ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER, COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS, MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS_NUMBER,\
    MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS, MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS_NUMBER
    # MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS, MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,\
    # MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS, MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER
from tableshift.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES, \
    MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL, MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL,\
    MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL, MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL
from tableshift.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES

# We passthrough all non-static columns because we use
# MIMIC-extract's default preprocessing/imputation and do not
# wish to modify it for these features
# (static features are not preprocessed by MIMIC-extract). See
# tableshift.datasets.mimic_extract.preprocess_mimic_extract().
_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

BENCHMARK_CONFIGS = {
    # Foodstamps
    "acsfoodstamps": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),
    "acsfoodstamps_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),
    "acsfoodstamps_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    # Income
    "acsincome": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),
    "acsincome_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),
    "acsincome_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),
    "acsincome_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),

    # Public Coverage
    "acspubcov": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov",
                                "years": ACS_YEARS}),
    "acspubcov_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_causal",
                                "years": ACS_YEARS}),
    "acspubcov_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_arguablycausal",
                                "years": ACS_YEARS}),

    # Unemployment
    "acsunemployment": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),
    "acsunemployment_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),
    "acsunemployment_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),
    "acsunemployment_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),

    # ANES, Split by region; OOD is south: (AL, AR, DE, D.C., FL, GA, KY, LA,
    # MD, MS, NC, OK, SC,TN, TX, VA, WV)

    # Voting
    "anes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),
    "anes_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),
    "anes_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    # Hypertension
    "brfss_blood_pressure": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),
    "brfss_blood_pressure_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_causal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),
    "brfss_blood_pressure_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_arguablycausal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),
    "brfss_blood_pressure_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_anticausal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),

    # Diabetes
    # "White nonhispanic" (in-domain) vs. all other race/ethnicity codes (OOD)
    "brfss_diabetes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),
    "brfss_diabetes_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_causal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),
    "brfss_diabetes_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_arguablycausal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),
    "brfss_diabetes_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_anticausal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),

    # Hospital readmission
    "diabetes_readmission": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),
    "diabetes_readmission_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),
    "diabetes_readmission_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),

    "heloc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ExternalRiskEstimateLow',
                                domain_split_ood_values=[0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "heloc"},
    ),

    # ICU Length of Stay
    "mimic_extract_los_3": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3"}),
    "mimic_extract_los_3_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL.names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_causal"}),
    "mimic_extract_los_3_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL.names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_arguablycausal"}),

    # ICU Mortality
    "mimic_extract_mort_hosp": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp"}),
    "mimic_extract_mort_hosp_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL.names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_causal"}),
    "mimic_extract_mort_hosp_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL.names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_arguablycausal"}),

    "nhanes_cholesterol": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='RIDRETH_merged',
                                domain_split_ood_values=[1, 2, 4, 6, 7],
                                domain_split_id_values=[3],
                                ),
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "cholesterol",
                                "years": NHANES_YEARS}),

    # Assistments
    "assistments": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),
    "assistments_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),
    "assistments_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),

    # College scorecard
    "college_scorecard": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),
    "college_scorecard_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),
    "college_scorecard_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),

    # Childhood lead
    "nhanes_lead": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),
    "nhanes_lead_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),

    # Sepsis
    # LOS >= 47 is roughly the 80th %ile of data.
    "physionet": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet"}),
    "physionet_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_causal"}),
    "physionet_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_anticausal"}),
}

for index in range(ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["acsincome_causal_test_"+f"{index}"] = ExperimentConfig(
                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                    domain_split_varname="DIVISION",
                                                                                    domain_split_ood_values=['01']),
                                                            grouper=None,
                                                            preprocessor_config=PreprocessorConfig(),
                                                            tabular_dataset_kwargs={"acs_task": "acsincome"})

for index in range(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["acsincome_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                    domain_split_varname="DIVISION",
                                                                                    domain_split_ood_values=['01']),
                                                            grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
                                                            preprocessor_config=PreprocessorConfig(),
                                                            tabular_dataset_kwargs={"acs_task": "acsincome"})
    
for index in range(ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["acsfoodstamps_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                        ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                        random_state=DEFAULT_RANDOM_STATE,
                                                                                        id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                        domain_split_varname="DIVISION",
                                                                                        domain_split_ood_values=['06']),
                                                                grouper=None,
                                                                preprocessor_config=PreprocessorConfig(),
                                                                tabular_dataset_kwargs={"acs_task": "acsfoodstamps"})

for index in range(ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["acsfoodstamps_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname="DIVISION",
                                                                                                domain_split_ood_values=['06']),
                                                                        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
                                                                        preprocessor_config=PreprocessorConfig(),
                                                                        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"})
for index in range(ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["acspubcov_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname="DIS",
                                                                                                domain_split_ood_values=['1.0']),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(),
                                                                        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_causal_test_"+f"{index}",
                                                                                                "years": ACS_YEARS})

for index in range(ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["acspubcov_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname="DIS",
                                                                                                domain_split_ood_values=['1.0']),
                                                                        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
                                                                        preprocessor_config=PreprocessorConfig(),
                                                                        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_causal_test_"+f"{index}",
                                                                                                "years": ACS_YEARS})

for index in range(ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["acsunemployment_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='SCHL',
                                                                                                # No high school diploma vs. GED/diploma or higher.
                                                                                                domain_split_ood_values=['01', '02', '03', '04',
                                                                                                                        '05', '06', '07', '08',
                                                                                                                        '09', '10', '11', '12',
                                                                                                                        '13', '14', '15']),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(),
                                                                        tabular_dataset_kwargs={"acs_task": "acsunemployment"})

for index in range(ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["acsunemployment_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='SCHL',
                                                                                                # No high school diploma vs. GED/diploma or higher.
                                                                                                domain_split_ood_values=['01', '02', '03', '04',
                                                                                                                        '05', '06', '07', '08',
                                                                                                                        '09', '10', '11', '12',
                                                                                                                        '13', '14', '15']),
                                                                        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
                                                                        preprocessor_config=PreprocessorConfig(),
                                                                        tabular_dataset_kwargs={"acs_task": "acsunemployment"})


for index in range(BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["brfss_diabetes_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                    splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                            ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                            random_state=DEFAULT_RANDOM_STATE,
                                                                                            id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                            domain_split_varname="PRACE1",
                                                                                            domain_split_ood_values=[2, 3, 4, 5, 6],
                                                                                            domain_split_id_values=[1, ]),
                                                                    grouper=None,
                                                                    preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
                                                                    tabular_dataset_kwargs={"name": "brfss_diabetes_causal_test_"+f"{index}",
                                                                                            "task": "diabetes", "years": BRFSS_YEARS})
for index in range(BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["brfss_diabetes_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname="PRACE1",
                                                                                                    domain_split_ood_values=[2, 3, 4, 5, 6],
                                                                                                    domain_split_id_values=[1, ]),
                                                                            grouper=Grouper({"SEX": [1, ]}, drop=False),
                                                                            preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
                                                                            tabular_dataset_kwargs={"name": "brfss_diabetes_arguablycausal_test_"+f"{index}",
                                                                                                    "task": "diabetes", "years": BRFSS_YEARS})

for index in range(BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["brfss_blood_pressure_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname="BMI5CAT",
                                                                                                    # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                                                                                    domain_split_ood_values=['3.0', '4.0']),
                                                                            grouper=None,
                                                                            preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
                                                                            tabular_dataset_kwargs={"name": "brfss_blood_pressure_causal_test_"+f"{index}",
                                                                                                    "task": "blood_pressure",
                                                                                                    "years": BRFSS_YEARS},
                                                                        )
for index in range(BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["brfss_blood_pressure_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname="BMI5CAT",
                                                                                                    # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                                                                                    domain_split_ood_values=['3.0', '4.0']),
                                                                            grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
                                                                            preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
                                                                            tabular_dataset_kwargs={"name": "brfss_blood_pressure_arguablycausal_test_"+f"{index}",
                                                                                                    "task": "blood_pressure",
                                                                                                    "years": BRFSS_YEARS},
                                                                        )
    
for index in range(DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER):
    BENCHMARK_CONFIGS["diabetes_readmission_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname='admission_source_id',
                                                                                                    domain_split_ood_values=[7, ]),
                                                                            # male vs. all others; white non-hispanic vs. others
                                                                            grouper=None,
                                                                            # Note: using min_frequency=0.01 reduces data
                                                                            # dimensionality from ~2400 -> 169 columns.
                                                                            # This is due to high cardinality of 'diag_*' features.
                                                                            preprocessor_config=PreprocessorConfig(min_frequency=0.01),
                                                                            tabular_dataset_kwargs={})
for index in range(DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["diabetes_readmission_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname='admission_source_id',
                                                                                                    domain_split_ood_values=[7, ]),
                                                                            # male vs. all others; white non-hispanic vs. others
                                                                            grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                                                                                            drop=False),
                                                                            # Note: using min_frequency=0.01 reduces data
                                                                            # dimensionality from ~2400 -> 169 columns.
                                                                            # This is due to high cardinality of 'diag_*' features.
                                                                            preprocessor_config=PreprocessorConfig(min_frequency=0.01),
                                                                            tabular_dataset_kwargs={})
for index in range(ANES_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["anes_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname='VCF0112',
                                                                                                    domain_split_ood_values=['3.0']),
                                                                            # male vs. all others; white non-hispanic vs. others
                                                                            grouper=None,
                                                                            preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                                                                                                dropna=None),
                                                                            tabular_dataset_kwargs={})
for index in range(ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["anes_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                            splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                    ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                    random_state=DEFAULT_RANDOM_STATE,
                                                                                                    id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                    domain_split_varname='VCF0112',
                                                                                                    domain_split_ood_values=['3.0']),
                                                                            # male vs. all others; white non-hispanic vs. others
                                                                            grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                                                                                            drop=False),
                                                                            preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                                                                                                dropna=None),
                                                                            tabular_dataset_kwargs={})

for index in range(ASSISTMENTS_FEATURES_CAUSAL_SUBSETS_NUMBER):
    list_passthrough = [feature.name for feature in ASSISTMENTS_FEATURES_CAUSAL_SUBSETS[index] if feature.name in ["skill_id", "bottom_hint", "first_action"]]
    BENCHMARK_CONFIGS["assistments_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='school_id',
                                                                                                domain_split_ood_values=[5040.0,
                                                                                                                        11502.0,
                                                                                                                        11318.0,
                                                                                                                        11976.0,
                                                                                                                        12421.0,
                                                                                                                        12379.0,
                                                                                                                        11791.0,
                                                                                                                        8359.0,
                                                                                                                        12406.0,
                                                                                                                        7594.0]),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(
                                                                            passthrough_columns=list_passthrough,
                                                                        ),
                                                                        tabular_dataset_kwargs={},
                                                                    )
for index in range(ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["assistments_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='school_id',
                                                                                                domain_split_ood_values=[5040.0,
                                                                                                                        11502.0,
                                                                                                                        11318.0,
                                                                                                                        11976.0,
                                                                                                                        12421.0,
                                                                                                                        12379.0,
                                                                                                                        11791.0,
                                                                                                                        8359.0,
                                                                                                                        12406.0,
                                                                                                                        7594.0]),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(
                                                                            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
                                                                        ),
                                                                        tabular_dataset_kwargs={},
                                                                    )


for index in range(COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["college_scorecard_causal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='CCBASIC',
                                                                                                domain_split_ood_values=[
                                                                                                    'Special Focus Institutions--Other special-focus institutions',
                                                                                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                                                                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                                                                                    'Baccalaureate Colleges--Diverse Fields',
                                                                                                    'Special Focus Institutions--Schools of art, music, and design',
                                                                                                    "Associate's--Private Not-for-profit",
                                                                                                    "Baccalaureate/Associate's Colleges",
                                                                                                    "Master's Colleges and Universities (larger programs)"]
                                                                                                ),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(
                                                                            # Several categorical features in college scorecard have > 10k
                                                                            # unique values; so we label-encode instead of one-hot encoding.
                                                                            categorical_features="label_encode",
                                                                            # Some important numeric features are not reported by universities
                                                                            # in a way that could be systematic (and we would like these included
                                                                            # in the sample, not excluded), so we use kbins
                                                                            numeric_features="kbins",
                                                                            n_bins=100,
                                                                            dropna=None,
                                                                        ),
                                                                        tabular_dataset_kwargs={},
                                                                    )
for index in range(COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    BENCHMARK_CONFIGS["college_scorecard_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
                                                                        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                                                                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                                                                                random_state=DEFAULT_RANDOM_STATE,
                                                                                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                                                                                domain_split_varname='CCBASIC',
                                                                                                domain_split_ood_values=[
                                                                                                    'Special Focus Institutions--Other special-focus institutions',
                                                                                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                                                                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                                                                                    'Baccalaureate Colleges--Diverse Fields',
                                                                                                    'Special Focus Institutions--Schools of art, music, and design',
                                                                                                    "Associate's--Private Not-for-profit",
                                                                                                    "Baccalaureate/Associate's Colleges",
                                                                                                    "Master's Colleges and Universities (larger programs)"]
                                                                                                ),
                                                                        grouper=None,
                                                                        preprocessor_config=PreprocessorConfig(
                                                                            # Several categorical features in college scorecard have > 10k
                                                                            # unique values; so we label-encode instead of one-hot encoding.
                                                                            categorical_features="label_encode",
                                                                            # Some important numeric features are not reported by universities
                                                                            # in a way that could be systematic (and we would like these included
                                                                            # in the sample, not excluded), so we use kbins
                                                                            numeric_features="kbins",
                                                                            n_bins=100,
                                                                            dropna=None,
                                                                        ),
                                                                        tabular_dataset_kwargs={},
                                                                    )

for index in range(MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["mimic_extract_los_3_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS[index].names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_causal_test_"+f"{index}"})
# for index in range(MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#     BENCHMARK_CONFIGS["mimic_extract_los_3_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
#         splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
#                                 ood_val_size=DEFAULT_OOD_VAL_SIZE,
#                                 random_state=DEFAULT_RANDOM_STATE,
#                                 id_test_size=DEFAULT_ID_TEST_SIZE,
#                                 domain_split_varname="insurance",
#                                 domain_split_ood_values=["Medicare"]),

#         grouper=Grouper({"gender": ['M'], }, drop=False),
#         preprocessor_config=PreprocessorConfig(
#             passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
#                                  if f in MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS[index].names]),
#         tabular_dataset_kwargs={"task": "los_3",
#                                 "name": "mimic_extract_los_3_arguablycausal_test_"+f"{index}"})


for index in range(MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS_NUMBER):
    BENCHMARK_CONFIGS["mimic_extract_mort_hosp_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS[index].names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_causal_test_"+f"{index}"})

# for index in range(MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#     BENCHMARK_CONFIGS["mimic_extract_mort_hosp_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
#         splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
#                                 ood_val_size=DEFAULT_OOD_VAL_SIZE,
#                                 random_state=DEFAULT_RANDOM_STATE,
#                                 id_test_size=DEFAULT_ID_TEST_SIZE,
#                                 domain_split_varname="insurance",
#                                 domain_split_ood_values=["Medicare"]),

#         grouper=Grouper({"gender": ['M'], }, drop=False),
#         preprocessor_config=PreprocessorConfig(
#             passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
#                                  if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS[index].names]),
#         tabular_dataset_kwargs={"task": "mort_hosp",
#                                 "name": "mimic_extract_mort_hosp_arguablycausal_test_"+f"{index}"})