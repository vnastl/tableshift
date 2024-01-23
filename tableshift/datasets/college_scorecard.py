"""
Utilities for the College Scorecard dataset.

This is a public data source and no special action is required
to access it.

For more information on datasets and access in TableShift, see:
* https://tableshift.org/datasets.html
* https://github.com/mlfoundations/tableshift
"""
import pandas as pd
from tableshift.core.features import Feature, FeatureList, cat_dtype
from tableshift.datasets.robustness import select_subset_minus_one, select_superset_plus_one

################################################################################
# Feature list
################################################################################
COLLEGE_SCORECARD_FEATURES = FeatureList(features=[
        Feature('C150_4', int, is_target=True,
                name_extended="Completion rate for first-time, full-time "
                              "students at four-year institutions (150% of "
                              "expected time to completion/6 years)"),
        Feature('STABBR', cat_dtype, name_extended='State postcode'),
        Feature('AccredAgency', cat_dtype, name_extended='Accreditor for institution'),
        Feature('sch_deg', float, name_extended='Predominant degree awarded (recoded 0s and 4s)'),
        Feature('main', cat_dtype, name_extended='Flag for main campus'),
        Feature('NUMBRANCH', int, name_extended='Number of branch campuses'),
        Feature('HIGHDEG', cat_dtype, name_extended='Highest degree awarded',
                value_mapping={
                        0: "Non-degree-granting",
                        1: "Certificate degree",
                        2: "Associate degree",
                        3: "Bachelor's degree",
                        4: "Graduate degree"}),
        Feature('CONTROL', cat_dtype, name_extended='Control of institution'),
        Feature('region', cat_dtype, name_extended='Region (IPEDS)'),
        Feature('LOCALE', cat_dtype, name_extended='Locale of institution'),
        Feature('locale2', float, name_extended='Degree of urbanization of institution'),
        Feature('CCBASIC', cat_dtype, name_extended='Carnegie Classification -- basic'),
        Feature('CCSIZSET', cat_dtype, name_extended='Carnegie Classification -- size and setting'),
        Feature('HBCU', cat_dtype, name_extended='Flag for Historically Black College and University'),
        Feature('ADM_RATE', float, name_extended='Admission rate'),
        Feature('ADM_RATE_ALL', float, name_extended='Admission rate for all campuses rolled up to the 6-digit OPE ID'),
        Feature('SATVRMID', float, name_extended='Midpoint of SAT scores at the institution (critical reading)'),
        Feature('SATMTMID', float, name_extended='Midpoint of SAT scores at the institution (math)'),
        Feature('SATWRMID', float, name_extended='Midpoint of SAT scores at the institution (writing)'),
        Feature('ACTCMMID', float, name_extended='Midpoint of the ACT cumulative score'),
        Feature('ACTENMID', float, name_extended='Midpoint of the ACT English score'),
        Feature('ACTMTMID', float, name_extended='Midpoint of the ACT math score'),
        Feature('ACTWRMID', float, name_extended='Midpoint of the ACT writing score'),
        Feature('PCIP01', float, name_extended='Percentage of degrees awarded in Agriculture, Agriculture Operations, And Related Sciences.'),
        Feature('PCIP03', float, name_extended='Percentage of degrees awarded in Natural Resources And Conservation.'),
        Feature('PCIP04', float, name_extended='Percentage of degrees awarded in Architecture And Related Services.'),
        Feature('PCIP05', float, name_extended='Percentage of degrees awarded in Area, Ethnic, Cultural, Gender, And Group Studies.'),
        Feature('PCIP09', float, name_extended='Percentage of degrees awarded in Communication, Journalism, And Related Programs.'),
        Feature('PCIP10', float, name_extended='Percentage of degrees awarded in Communications Technologies/Technicians And Support Services.'),
        Feature('PCIP11', float, name_extended='Percentage of degrees awarded in Computer And Information Sciences And Support Services.'),
        Feature('PCIP12', float, name_extended='Percentage of degrees awarded in Personal And Culinary Services.'),
        Feature('PCIP13', float, name_extended='Percentage of degrees awarded in Education.'),
        Feature('PCIP14', float, name_extended='Percentage of degrees awarded in Engineering.'),
        Feature('PCIP15', float, name_extended='Percentage of degrees awarded in Engineering Technologies And Engineering-Related Fields.'),
        Feature('PCIP16', float, name_extended='Percentage of degrees awarded in Foreign Languages, Literatures, And Linguistics.'),
        Feature('PCIP19', float, name_extended='Percentage of degrees awarded in Family And Consumer Sciences/Human Sciences.'),
        Feature('PCIP22', float, name_extended='Percentage of degrees awarded in Legal Professions And Studies.'),
        Feature('PCIP23', float, name_extended='Percentage of degrees awarded in English Language And Literature/Letters.'),
        Feature('PCIP24', float, name_extended='Percentage of degrees awarded in Liberal Arts And Sciences, General Studies And Humanities.'),
        Feature('PCIP25', float, name_extended='Percentage of degrees awarded in Library Science.'),
        Feature('PCIP26', float, name_extended='Percentage of degrees awarded in Biological And Biomedical Sciences.'),
        Feature('PCIP27', float, name_extended='Percentage of degrees awarded in Mathematics And Statistics.'),
        Feature('PCIP29', float, name_extended='Percentage of degrees awarded in Military Technologies And Applied Sciences.'),
        Feature('PCIP30', float, name_extended='Percentage of degrees awarded in Multi/Interdisciplinary Studies.'),
        Feature('PCIP31', float, name_extended='Percentage of degrees awarded in Parks, Recreation, Leisure, And Fitness Studies.'),
        Feature('PCIP38', float, name_extended='Percentage of degrees awarded in Philosophy And Religious Studies.'),
        Feature('PCIP39', float, name_extended='Percentage of degrees awarded in Theology And Religious Vocations.'),
        Feature('PCIP40', float, name_extended='Percentage of degrees awarded in Physical Sciences.'),
        Feature('PCIP41', float, name_extended='Percentage of degrees awarded in Science Technologies/Technicians.'),
        Feature('PCIP42', float, name_extended='Percentage of degrees awarded in Psychology.'),
        Feature('PCIP43', float, name_extended='Percentage of degrees awarded in Homeland Security, Law Enforcement, Firefighting And Related Protective Services.'),
        Feature('PCIP44', float, name_extended='Percentage of degrees awarded in Public Administration And Social Service Professions.'),
        Feature('PCIP45', float, name_extended='Percentage of degrees awarded in Social Sciences.'),
        Feature('PCIP46', float, name_extended='Percentage of degrees awarded in Construction Trades.'),
        Feature('PCIP47', float, name_extended='Percentage of degrees awarded in Mechanic And Repair Technologies/Technicians.'),
        Feature('PCIP48', float, name_extended='Percentage of degrees awarded in Precision Production.'),
        Feature('PCIP49', float, name_extended='Percentage of degrees awarded in Transportation And Materials Moving.'),
        Feature('PCIP50', float, name_extended='Percentage of degrees awarded in Visual And Performing Arts.'),
        Feature('PCIP51', float, name_extended='Percentage of degrees awarded in Health Professions And Related Programs.'),
        Feature('PCIP52', float, name_extended='Percentage of degrees awarded in Business, Management, Marketing, And Related Support Services.'),
        Feature('PCIP54', float, name_extended='Percentage of degrees awarded in History.'),
        Feature('DISTANCEONLY', cat_dtype, name_extended='Flag for distance-education-only education'),
        Feature('UGDS', float, name_extended='Enrollment of undergraduate degree-seeking students'),
        Feature('UG', float, name_extended='Enrollment of all undergraduate students'),
        Feature('UGDS_WHITE', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are white'),
        Feature('UGDS_BLACK', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are black'),
        Feature('UGDS_HISP', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are Hispanic'),
        Feature('UGDS_ASIAN', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are Asian'),
        Feature('UGDS_AIAN', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are American Indian/Alaska Native'),
        Feature('UGDS_NHPI', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are Native Hawaiian/Pacific Islander'),
        Feature('UGDS_2MOR', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are two or more races'),
        Feature('UGDS_NRA', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are non-resident aliens'),
        Feature('UGDS_UNKN', float, name_extended='Total share of enrollment of undergraduate degree-seeking students whose race is unknown'),
        Feature('UGDS_WHITENH', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are white non-Hispanic'),
        Feature('UGDS_BLACKNH', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are black non-Hispanic'),
        Feature('UGDS_API', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are Asian/Pacific Islander'),
        Feature('UGDS_AIANOld', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are American Indian/Alaska Native'),
        Feature('UGDS_HISPOld', float, name_extended='Total share of enrollment of undergraduate degree-seeking students who are Hispanic'),
        Feature('UG_NRA', float, name_extended='Total share of enrollment of undergraduate students who are non-resident aliens'),
        Feature('UG_UNKN', float, name_extended='Total share of enrollment of undergraduate students whose race is unknown'),
        Feature('UG_WHITENH', float, name_extended='Total share of enrollment of undergraduate students who are white non-Hispanic'),
        Feature('UG_BLACKNH', float, name_extended='Total share of enrollment of undergraduate students who are black non-Hispanic'),
        Feature('UG_API', float, name_extended='Total share of enrollment of undergraduate students who are Asian/Pacific Islander'),
        Feature('UG_AIANOld', float, name_extended='Total share of enrollment of undergraduate students who are American Indian/Alaska Native'),
        Feature('UG_HISPOld', float, name_extended='Total share of enrollment of undergraduate students who are Hispanic'),
        Feature('PPTUG_EF', float, name_extended='Share of undergraduate, degree-/certificate-seeking students who are part-time'),
        Feature('PPTUG_EF2', float, name_extended='Share of undergraduate, degree-/certificate-seeking students who are part-time'),
        Feature('NPT4_PROG', float, name_extended='Average net price for the largest program at the institution for program-year institutions'),
        Feature('COSTT4_A', float, name_extended='Average cost of attendance (academic year institutions)'),
        Feature('COSTT4_P', float, name_extended='Average cost of attendance (program-year institutions)'),
        Feature('TUITIONFEE_IN', float, name_extended='In-state tuition and fees'),
        Feature('TUITIONFEE_OUT', float, name_extended='Out-of-state tuition and fees'),
        Feature('TUITIONFEE_PROG', float, name_extended='Tuition and fees for program-year institutions'),
        Feature('TUITFTE', float, name_extended='Net tuition revenue per full-time equivalent student'),
        Feature('INEXPFTE', float, name_extended='Instructional expenditures per full-time equivalent student'),
        Feature('AVGFACSAL', float, name_extended='Average faculty salary'),
        Feature('PFTFAC', float, name_extended='Proportion of faculty that is full-time'),
        Feature('PCTPELL', float, name_extended='Percentage of undergraduates who receive a Pell Grant'),
        Feature('loan_ever', float, name_extended='Share of students who received a federal loan while in school',
                na_values=('PrivacySuppressed',)),
        Feature('pell_ever', float, name_extended='Share of students who received a Pell Grant while in school',
                na_values=('PrivacySuppressed',)),
        Feature('age_entry', float, name_extended='Average age of entry, via SSA data',
                na_values=('PrivacySuppressed',)),
        Feature('age_entry_sq', float, name_extended='Average of the age of entry squared',
                na_values=('PrivacySuppressed',)),
        Feature('agege24', float, name_extended='Percent of students over 23 at entry',
                na_values=('PrivacySuppressed',)),
        Feature('female', float, name_extended='Share of female students, via SSA data',
                na_values=('PrivacySuppressed',)),
        Feature('married', float, name_extended='Share of married students',
                na_values=('PrivacySuppressed',)),
        Feature('dependent', float, name_extended='Share of dependent students',
                na_values=('PrivacySuppressed',)),
        Feature('veteran', float, name_extended='Share of veteran students',
                na_values=('PrivacySuppressed',)),
        Feature('first_gen', float, name_extended='Share of first-generation students',
                na_values=('PrivacySuppressed',)),
        Feature('faminc', float, name_extended='Average family income',
                na_values=('PrivacySuppressed',)),
        Feature('md_faminc', float, name_extended='Median family income',
                na_values=('PrivacySuppressed',)),
        Feature('pct_white', float, name_extended="Percent of the population from students' zip codes that is White, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_black', float, name_extended="Percent of the population from students' zip codes that is Black, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_asian', float, name_extended="Percent of the population from students' zip codes that is Asian, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_hispanic', float, name_extended="Percent of the population from students' zip codes that is Hispanic, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_ba', float, name_extended="Percent of the population from students' zip codes with a bachelor's degree over the age 25, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_grad_prof', float, name_extended="Percent of the population from students' zip codes over 25 with a professional degree, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('pct_born_us', float, name_extended="Percent of the population from students' zip codes that was born in the US, via Census data",
                na_values=('PrivacySuppressed',)),
        Feature('median_hh_inc', float, name_extended='Median household income',
                na_values=('PrivacySuppressed',)),
        Feature('poverty_rate', float, name_extended='Poverty rate, via Census data',
                na_values=('PrivacySuppressed',)),
        Feature('unemp_rate', float, name_extended='Unemployment rate, via Census data',
                na_values=('PrivacySuppressed',)),
])

################################################################################
# Causal feature list
################################################################################
COLLEGE_SCORECARD_FEATURES_CAUSAL = FeatureList(features=[
        Feature('C150_4', int, is_target=True,
                name_extended="Completion rate for first-time, full-time "
                              "students at four-year institutions (150% of "
                              "expected time to completion/6 years)"),
        Feature('AccredAgency', cat_dtype, name_extended='Accreditor for institution'),
        Feature('HIGHDEG', cat_dtype, name_extended='Highest degree awarded',
                value_mapping={
                        0: "Non-degree-granting",
                        1: "Certificate degree",
                        2: "Associate degree",
                        3: "Bachelor's degree",
                        4: "Graduate degree"}),
        Feature('CONTROL', cat_dtype, name_extended='Control of institution'),
        Feature('region', cat_dtype, name_extended='Region (IPEDS)'),
        Feature('LOCALE', cat_dtype, name_extended='Locale of institution'),
        Feature('locale2', float, name_extended='Degree of urbanization of institution'),
        Feature('CCBASIC', cat_dtype, name_extended='Carnegie Classification -- basic'),
        # Feature('CCSIZSET', cat_dtype, name_extended='Carnegie Classification -- size and setting'),
        Feature('HBCU', cat_dtype, name_extended='Flag for Historically Black College and University'),
        Feature('DISTANCEONLY', cat_dtype, name_extended='Flag for distance-education-only education'),
        Feature('poverty_rate', float, name_extended='Poverty rate, via Census data',
                na_values=('PrivacySuppressed',)),
        Feature('unemp_rate', float, name_extended='Unemployment rate, via Census data',
                na_values=('PrivacySuppressed',)),
])
target = Feature('C150_4', int, is_target=True,
                name_extended="Completion rate for first-time, full-time "
                              "students at four-year institutions (150% of "
                              "expected time to completion/6 years)")
domain = Feature('CCBASIC', cat_dtype, name_extended='Carnegie Classification -- basic')

causal_features = COLLEGE_SCORECARD_FEATURES_CAUSAL.features.copy()
causal_features.remove(target)
causal_features.remove(domain)
causal_subsets = select_subset_minus_one(causal_features)
COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS = []
for subset in causal_subsets:
    subset.append(target)
    subset.append(domain)
    COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS.append(FeatureList(subset))
COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER = len(causal_subsets)

COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL = FeatureList(features=[
        Feature('C150_4', int, is_target=True,
                name_extended="Completion rate for first-time, full-time "
                              "students at four-year institutions (150% of "
                              "expected time to completion/6 years)"),
        Feature('AccredAgency', cat_dtype, name_extended='Accreditor for institution'),
        Feature('HIGHDEG', cat_dtype, name_extended='Highest degree awarded',
                value_mapping={
                        0: "Non-degree-granting",
                        1: "Certificate degree",
                        2: "Associate degree",
                        3: "Bachelor's degree",
                        4: "Graduate degree"}),
        Feature('CONTROL', cat_dtype, name_extended='Control of institution'),
        Feature('region', cat_dtype, name_extended='Region (IPEDS)'),
        Feature('LOCALE', cat_dtype, name_extended='Locale of institution'),
        Feature('locale2', float, name_extended='Degree of urbanization of institution'),
        Feature('CCBASIC', cat_dtype, name_extended='Carnegie Classification -- basic'),
        # Feature('CCSIZSET', cat_dtype, name_extended='Carnegie Classification -- size and setting'),
        Feature('HBCU', cat_dtype, name_extended='Flag for Historically Black College and University'),
        Feature('DISTANCEONLY', cat_dtype, name_extended='Flag for distance-education-only education'),
        Feature('median_hh_inc', float, name_extended='Median household income',
                na_values=('PrivacySuppressed',)),
        Feature('poverty_rate', float, name_extended='Poverty rate, via Census data',
                na_values=('PrivacySuppressed',)),
        Feature('unemp_rate', float, name_extended='Unemployment rate, via Census data',
                na_values=('PrivacySuppressed',)),
        Feature('TUITIONFEE_IN', float, name_extended='In-state tuition and fees'),
        Feature('TUITIONFEE_OUT', float, name_extended='Out-of-state tuition and fees'),
        Feature('TUITIONFEE_PROG', float, name_extended='Tuition and fees for program-year institutions'),
        Feature('ADM_RATE', float, name_extended='Admission rate'),
        Feature('ADM_RATE_ALL', float, name_extended='Admission rate for all campuses rolled up to the 6-digit OPE ID'),
        Feature('SATVRMID', float, name_extended='Midpoint of SAT scores at the institution (critical reading)'),
        Feature('SATMTMID', float, name_extended='Midpoint of SAT scores at the institution (math)'),
        Feature('SATWRMID', float, name_extended='Midpoint of SAT scores at the institution (writing)'),
        Feature('ACTCMMID', float, name_extended='Midpoint of the ACT cumulative score'),
        Feature('ACTENMID', float, name_extended='Midpoint of the ACT English score'),
        Feature('ACTMTMID', float, name_extended='Midpoint of the ACT math score'),
        Feature('ACTWRMID', float, name_extended='Midpoint of the ACT writing score'),
        Feature('NPT4_PROG', float, name_extended='Average net price for the largest program at the institution for program-year institutions'),
        Feature('COSTT4_A', float, name_extended='Average cost of attendance (academic year institutions)'),
        Feature('COSTT4_P', float, name_extended='Average cost of attendance (program-year institutions)'),
        Feature('loan_ever', float, name_extended='Share of students who received a federal loan while in school',
                na_values=('PrivacySuppressed',)),
        Feature('pell_ever', float, name_extended='Share of students who received a Pell Grant while in school',
                na_values=('PrivacySuppressed',)),
        Feature('PCTPELL', float, name_extended='Percentage of undergraduates who receive a Pell Grant'),\
        Feature('faminc', float, name_extended='Average family income',
                na_values=('PrivacySuppressed',)),
        Feature('md_faminc', float, name_extended='Median family income',
                na_values=('PrivacySuppressed',)),
        Feature('UGDS', float, name_extended='Enrollment of undergraduate degree-seeking students'),
        Feature('UG', float, name_extended='Enrollment of all undergraduate students'),
])

arguablycausal_supersets = select_superset_plus_one(COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL.features, COLLEGE_SCORECARD_FEATURES.features)
COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS = []
for superset in arguablycausal_supersets:
    COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS.append(FeatureList(superset))
COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER = len(arguablycausal_supersets)


def preprocess_college_scorecard(df:pd.DataFrame)->pd.DataFrame:
        df[COLLEGE_SCORECARD_FEATURES.target] = (df[COLLEGE_SCORECARD_FEATURES.target] > 0.5).astype(int)
        return df

# def preprocess_college_scorecard_causal(df:pd.DataFrame)->pd.DataFrame:
#         df[COLLEGE_SCORECARD_FEATURES_CAUSAL.target] = (df[COLLEGE_SCORECARD_FEATURES_CAUSAL.target] > 0.5).astype(int)
#         return df