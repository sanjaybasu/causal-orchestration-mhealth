"""Build requirement_map_DRAFT.json and requirement_map_DRAFT.md from one data definition."""
import json
from collections import OrderedDict

OUT = "/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth_submission/revision_1/checklist/"

SOURCES = [
    # id, title, body, year, url/doi, verified, note
    ("S01", "Standards of Care in Diabetes-2026 (Sections 1, 5, 6, 11, 12)", "American Diabetes Association Professional Practice Committee", 2026,
     "Diabetes Care 2026;49(Suppl 1). Sec 1 doi:10.2337/dc26-S001; Sec 5 doi:10.2337/dc26-S005; Sec 6 doi:10.2337/dc26-S006; Sec 11 doi:10.2337/dc26-S011; Sec 12 doi:10.2337/dc26-S012",
     "yes", "PubMed metadata for all sections; recommendation text 1.7, 1.8, 5.1, 5.2, 5.10, 5.20, 5.40, 6.2, 6.18, 11.1a, 12.4, 12.5, 12.23 read in the full 2026 PDF."),
    ("S02", "2025 AHA/ACC/AANP/AAPA/ABC/ACCP/ACPM/AGS/AMA/ASPC/NMA/PCNA/SGIM Guideline for the Prevention, Detection, Evaluation and Management of High Blood Pressure in Adults", "American Heart Association / American College of Cardiology (Jones DW, chair) and 11 societies", 2025,
     "Hypertension 2025;82(10):e212-e316. doi:10.1161/HYP.0000000000000249 (also Circulation doi:10.1161/CIR.0000000000001356; JACC doi:10.1016/j.jacc.2025.05.007)",
     "yes", "Citation verified. Top take-home messages 3, 5, 8, 9, 10 read in the ACC Guideline-at-a-Glance (Gulati et al., JACC 2025, doi:10.1016/j.jacc.2025.07.010). The monthly-follow-up recommendation text was confirmed only through a search summary of the guideline; the full text was not accessible (HTTP 403)."),
    ("S03", "Global Strategy for the Diagnosis, Management, and Prevention of Chronic Obstructive Pulmonary Disease, 2026 Report (v1.3, 8 Dec 2025)", "Global Initiative for Chronic Obstructive Lung Disease (GOLD)", 2026,
     "https://goldcopd.org/wp-content/uploads/2025/12/GOLD-REPORT-2026-v1.3-8Dec2025_WMV.pdf",
     "yes", "Primary PDF read: inhaler technique and adherence assessed regularly; vaccinations; pulmonary rehabilitation; written action plan; early follow-up within 1 month after discharge."),
    ("S04", "Global Strategy for Asthma Management and Prevention, 2026 update, and GINA Summary Guide 2026", "Global Initiative for Asthma (GINA)", 2026,
     "https://ginasthma.org/2026-gina-strategy-report/ ; https://ginasthma.org/wp-content/uploads/2026/07/GINA-Summary-Guide-2026-WEB-WMS.pdf",
     "yes", "Summary Guide PDF read: written asthma action plan for all patients; check inhaler technique at every opportunity; review 1-3 months after start, then every 3-12 months; review within 1 week after an exacerbation; adults should not be treated with SABA alone."),
    ("S05", "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure", "American Heart Association / American College of Cardiology / Heart Failure Society of America (Heidenreich PA et al.)", 2022,
     "Circulation 2022;145(18):e895-e1032. doi:10.1161/CIR.0000000000001063",
     "yes", "Full text read: Sec 7.1.1 multidisciplinary care (COR 1), self-care education and support (COR 1), respiratory vaccination (COR 2a); assessment of social determinants (housing, food, transportation); early follow-up generally within 7 days after discharge (COR 2a). No newer US AHA/ACC/HFSA comprehensive HF guideline found as of 2026-10-17 (2026 ESC HF guideline exists; not used because the request asks for US guidelines)."),
    ("S06", "Screening for Depression and Suicide Risk in Adults: USPSTF Recommendation Statement", "US Preventive Services Task Force", 2023,
     "JAMA 2023;329(23):2057-2067. doi:10.1001/jama.2023.9297", "yes", "PubMed abstract read (B for depression screening; I for suicide-risk screening of asymptomatic adults)."),
    ("S07", "Screening for Anxiety Disorders in Adults: USPSTF Recommendation Statement", "US Preventive Services Task Force", 2023,
     "JAMA 2023;329(24):2163-2170. doi:10.1001/jama.2023.9301", "yes", "PubMed abstract and USPSTF page read (GAD-2/GAD-7 instruments; confirmatory assessment after positive screen; psychotherapy such as CBT and pharmacotherapy listed as treatments)."),
    ("S08", "Interventions for Tobacco Smoking Cessation in Adults, Including Pregnant Persons: USPSTF Recommendation Statement", "US Preventive Services Task Force", 2021,
     "JAMA 2021;325(3):265-279. doi:10.1001/jama.2020.25019", "yes", "PubMed abstract read (A: ask, advise, provide behavioral interventions and FDA-approved pharmacotherapy)."),
    ("S09", "Screening and Behavioral Counseling Interventions to Reduce Unhealthy Alcohol Use in Adolescents and Adults: USPSTF Recommendation Statement", "US Preventive Services Task Force", 2018,
     "JAMA 2018;320(18):1899-1909. doi:10.1001/jama.2018.16789", "yes", "Current final statement (B). USPSTF posted a draft update in 2025 that is consistent with 2018; no final update as of the USPSTF page fetched 2026-10-17."),
    ("S10", "VA/DoD Clinical Practice Guideline for the Management of Major Depressive Disorder, Version 4.0", "US Department of Veterans Affairs and Department of Defense", 2022,
     "https://www.healthquality.va.gov/HEALTHQUALITY/guidelines/MH/mdd/VADODMDDCPGFinal508.pdf ; synopsis Ann Intern Med 2022;175(10):1440-1451, doi:10.7326/M22-1603",
     "yes", "Full PDF read: Rec 2 quantitative severity measure at regular intervals (weak for); Rec 3 collaborative/integrated care in primary care (strong for); Rec 6 psychotherapy or pharmacotherapy."),
    ("S11", "VA/DoD Clinical Practice Guideline for the Management of Substance Use Disorders", "US Department of Veterans Affairs and Department of Defense", 2021,
     "https://www.healthquality.va.gov/guidelines/MH/sud/VADoDSUDCPG.pdf ; synopsis Ann Intern Med 2022;175(5):720-731, doi:10.7326/M21-4011",
     "yes", "Synopsis verified on PubMed (Module A screening and treatment; Module B stabilization and withdrawal). Full-text PDF download returned HTML, so individual recommendation wording was not read."),
    ("S12", "The ASAM National Practice Guideline for the Treatment of Opioid Use Disorder: 2020 Focused Update", "American Society of Addiction Medicine", 2020,
     "J Addict Med 2020;14(2S Suppl 1):1-91. doi:10.1097/ADM.0000000000000633", "yes", "PubMed metadata; guideline PDF read: naloxone should be provided to patients treated for or with a history of OUD, with training of patients and family; assessment includes tests for hepatitis B and C and HIV."),
    ("S13", "TIP 63: Medications for Opioid Use Disorder (updated 2021)", "Substance Abuse and Mental Health Services Administration", 2021,
     "https://store.samhsa.gov/product/tip-63-medications-opioid-use-disorder/pep21-02-01-002", "yes", "Existence, year, and scope (methadone, buprenorphine, naltrexone) confirmed through search results (ODPHP, AHRQ, ASAM)."),
    ("S14", "The American Psychiatric Association Practice Guideline for the Pharmacological Treatment of Patients With Alcohol Use Disorder", "American Psychiatric Association (Reus VI et al.)", 2018,
     "Am J Psychiatry 2018;175(1):86-90. doi:10.1176/appi.ajp.2017.1750101", "yes", "PubMed metadata verified; naltrexone and acamprosate recommendation confirmed through search summaries (Healio, MDedge), not through the full text."),
    ("S15", "FDA Approves First Over-the-Counter Naloxone Nasal Spray (press release, 29 Mar 2023)", "US Food and Drug Administration", 2023,
     "https://www.fda.gov/news-events/press-announcements/fda-approves-first-over-counter-naloxone-nasal-spray", "yes", "Page fetched."),
    ("S16", "988 Suicide & Crisis Lifeline", "Substance Abuse and Mental Health Services Administration", 2022,
     "https://988lifeline.org ; https://www.samhsa.gov/sites/default/files/988-factsheet.pdf", "yes", "Call, text, or chat 988; confirmed through SAMHSA fact sheet in search results."),
    ("S17", "Quitlines and Other Cessation Support Resources (1-800-QUIT-NOW)", "Centers for Disease Control and Prevention", 2024,
     "https://www.cdc.gov/tobacco/hcp/patient-care/quitlines-and-other-resources.html", "yes", "Page fetched; free counseling in all states. Page year not displayed; 2024 is the year of the page reorganization [UNVERIFIED year — check manually]."),
    ("S18", "American Geriatrics Society 2023 Updated AGS Beers Criteria for Potentially Inappropriate Medication Use in Older Adults", "American Geriatrics Society Beers Criteria Update Expert Panel", 2023,
     "J Am Geriatr Soc 2023;71(7):2052-2081. doi:10.1111/jgs.18372", "yes", "Citation confirmed through search results (AHRQ PSNet, Hartford Foundation)."),
    ("S19", "Pharmacists' Patient Care Process", "Joint Commission of Pharmacy Practitioners (JCPP)", 2014,
     "https://www.oregon.gov/pharmacy/Documents/JCPP_Pharmacists_Patient_Care_Process.pdf", "yes", "Approved 29 May 2014; collect, assess, plan, implement, follow-up; confirmed through search results."),
    ("S20", "National Patient Safety Goal NPSG.03.06.01: Maintain and communicate accurate patient medication information", "The Joint Commission", 2023,
     "https://www.jointcommission.org/-/media/tjc/documents/standards/national-patient-safety-goals/2023/npsg_chapter_ahc_jul2023.pdf", "yes", "Goal text and rationale confirmed through search results; the PDF itself returned HTML when downloaded, so element-of-performance wording was not read."),
    ("S21", "Chronic Care Management Services (MLN909188)", "Centers for Medicare & Medicaid Services, Medicare Learning Network", 2025,
     "https://www.cms.gov/Outreach-and-Education/Medicare-Learning-Network-MLN/MLNProducts/Downloads/ChronicCareManagement.pdf", "yes", "PDF read (June 2025): care plan elements include problem list, measurable goals, symptom management, planned interventions, medication management, coordination with outside resources, periodic review; comprehensive care management includes review of medications and interactions and 24/7 access. Medicare billing document; used here as a description of care-plan content, not as a Medicaid requirement."),
    ("S22", "Integrating Social Care into the Delivery of Health Care: Moving Upstream to Improve the Nation's Health", "National Academies of Sciences, Engineering, and Medicine", 2019,
     "doi:10.17226/25467", "yes", "Five activities confirmed: awareness, adjustment, assistance, alignment, advocacy."),
    ("S23", "Standardized Screening for Health-Related Social Needs in Clinical Settings: The Accountable Health Communities Screening Tool (NAM Perspectives); Accountable Health Communities Model", "Billioux A, Verlander K, Anthony S, Alley D (CMS Innovation Center); CMS", 2017,
     "doi:10.31478/201705b ; https://www.cms.gov/priorities/innovation/innovation-models/ahcm", "yes", "Core domains: housing instability, food insecurity, transportation, utilities, interpersonal safety; supplemental domains include family and social supports (child care, social isolation), employment and financial strain. Model ran May 2017-April 2023 with assistance and alignment tracks."),
    ("S24", "CMCS Informational Bulletin: Medicaid Coverage of Certain Medical Transportation under the Consolidated Appropriations Act, 2021; 42 CFR 431.53 Assurance of transportation", "Centers for Medicare & Medicaid Services, Center for Medicaid and CHIP Services", 2021,
     "https://www.medicaid.gov/federal-policy-guidance/downloads/cib071221.pdf ; https://www.law.cornell.edu/cfr/text/42/431.53", "yes", "Bulletin dated 12 Jul 2021 read: states must ensure necessary transportation to and from providers."),
    ("S25", "Supplemental Nutrition Assistance Program (SNAP): eligibility and how to apply", "US Department of Agriculture, Food and Nutrition Service", 2026,
     "https://www.fns.usda.gov/snap/recipient/eligibility", "yes", "Apply through the state agency; decision within 30 days; eligibility interview. Page notes 2025 statutory changes to work requirements and non-citizen eligibility."),
    ("S26", "Special Supplemental Nutrition Program for Women, Infants, and Children (WIC)", "US Department of Agriculture, Food and Nutrition Service", 2026,
     "https://www.fns.usda.gov/wic", "yes", "Program scope confirmed through search results (pregnant, postpartum, breastfeeding women, infants, children under 5; adjunct income eligibility through Medicaid/SNAP). Direct fetch returned HTTP 403."),
    ("S27", "Low Income Home Energy Assistance Program (LIHEAP)", "US Department of Health and Human Services, Administration for Children and Families, Office of Community Services", 2026,
     "https://acf.gov/ocs/programs/liheap", "yes", "Heating and cooling bill assistance, energy crisis assistance, weatherization; confirmed through ACF fact sheet in search results."),
    ("S28", "Get Legal Help (LSC-funded civil legal aid)", "Legal Services Corporation", 2026,
     "https://www.lsc.gov/about-lsc/what-legal-aid/get-legal-help", "yes", "Page fetched (locator for LSC-funded organizations; LawHelp.org)."),
    ("S29", "Our Epidemic of Loneliness and Isolation: The U.S. Surgeon General's Advisory on the Healing Effects of Social Connection and Community", "Office of the US Surgeon General", 2023,
     "https://www.hhs.gov/sites/default/files/surgeon-general-social-connection-advisory.pdf", "yes", "Released 2 May 2023; recommendations for health systems confirmed through search results."),
    ("S30", "Dietary Guidelines for Americans, 2025-2030", "US Department of Agriculture and US Department of Health and Human Services", 2026,
     "https://www.dietaryguidelines.gov/", "yes", "Released 7 Jan 2026 (AJMC, AACOM reports); site confirms new edition."),
    ("S31", "Child Care and Development Fund (CCDF)", "US Department of Health and Human Services, Administration for Children and Families, Office of Child Care", 2026,
     "https://acf.gov/occ", "yes", "Child care subsidies for low-income working families through state agencies; confirmed through ACF pages in search results. Direct fetch returned HTTP 403."),
    ("S32", "American Job Centers (CareerOneStop locator)", "US Department of Labor, Employment and Training Administration", 2026,
     "https://www.careeronestop.org/LocalHelp/AmericanJobCenters/find-american-job-centers.aspx", "yes", "About 2,500 centers offering free job search, training, and career services; confirmed through DOL TEN 25-14 and search results. Direct fetch returned HTTP 403."),
    ("S33", "Office of Lead Hazard Control and Healthy Homes", "US Department of Housing and Urban Development", 2026,
     "https://www.hud.gov/program_offices/healthy_homes", "yes", "Office page reached; hazards (mold and moisture, pests, lead, carbon monoxide, injury hazards) confirmed through search results."),
    ("S34", "Notice CPD-17-01: Establishing Additional Requirements for a Continuum of Care Centralized or Coordinated Assessment System", "US Department of Housing and Urban Development, Office of Community Planning and Development", 2017,
     "https://files.hudexchange.info/resources/documents/Notice-CPD-17-01-Establishing-Additional-Requirements-or-a-Continuum-of-Care-Centralized-or-Coordinated-Assessment-System.pdf", "yes", "Issued 23 Jan 2017; coordinated entry access, assessment, prioritization, referral; confirmed through search results."),
    ("S35", "211 information and referral helpline", "United Way Worldwide and local 211 providers", 2026,
     "https://www.211.org", "yes", "Free, confidential, 24/7 referral to food, housing, utility, and other community services; confirmed through multiple United Way pages in search results."),
    ("S36", "Free Tax Return Preparation for Qualifying Taxpayers (VITA/TCE)", "Internal Revenue Service", 2026,
     "https://www.irs.gov/individuals/free-tax-return-preparation-for-qualifying-taxpayers", "yes", "Page fetched (free tax preparation; EITC-level incomes; locator)."),
]

def I(id_, text, q, met, notmet, domain, src, detail, eval_only=False, applies_if=None):
    d = OrderedDict(id=id_, requirement=text, question=q, met=met, not_met=notmet,
                    domain=domain, sources=src, source_detail=detail, eval_only=eval_only)
    if applies_if:
        d["applies_if"] = applies_if
    return d

CATS = OrderedDict()

CATS["UNIVERSAL"] = dict(kind="universal", label="(all patients)", items=[
    I("U-1", "State a follow-up contact or review interval.",
      "Does the plan include a specific step that schedules a follow-up contact or plan review at a stated interval or date?",
      "Names a timeframe (e.g., 'call in 2 weeks', 'review plan at 30 days') and who makes the contact.",
      "No timeframe, or only vague wording such as 'follow up as needed' or 'regularly'.",
      "efficiency", ["S21", "S02", "S04"], "CMS CCM care plan element 'periodic review'; condition intervals in S02 (monthly until BP control) and S04 (1-3 months after treatment start)."),
    I("U-2", "Give red-flag symptoms and an escalation path.",
      "Does the plan include a specific step that tells the patient which warning symptoms need urgent care and whom to call or where to go?",
      "Lists at least one condition-relevant warning sign AND an action (call care team/after-hours line, 911/emergency department, or 988).",
      "No warning signs, or warning signs without an action, or only 'seek care if worse'.",
      "safety", ["S21", "S04", "S05", "S16"], "CMS CCM 24/7 access and symptom management; GINA written action plan; HF self-care ('what to do in response to symptoms'); 988 for mental health crisis."),
    I("U-3", "Do a medication reconciliation or review.",
      "Does the plan include a specific step to compile and review the patient's current medication list (including OTC and supplements) with a clinician or pharmacist?",
      "Names a medication review/reconciliation task (e.g., 'pharmacist reviews all bottles at next visit', 'brown-bag review').",
      "No medication review step, or only 'take medications as prescribed'.",
      "safety", ["S20", "S21", "S19"], "NPSG.03.06.01; CMS CCM 'review medications and any potential interactions'; JCPP collect/assess steps.", eval_only=True),
])

CATS["MEDICATION_ADHERENCE"] = dict(kind="clinical", label="medication non-adherence", items=[
    I("MA-1", "Identify the specific reason(s) for non-adherence.",
      "Does the plan include a specific step to ask about and record why the patient misses or stops medications (cost, side effects, forgetting, beliefs, access)?",
      "Names an assessment of adherence barriers with at least one barrier type to explore.",
      "Assumes a reason without assessment, or gives only generic encouragement to take medications.",
      "safety", ["S01", "S05", "S21"], "ADA 1.7 (financial barriers, cost-related nonadherence discussion); HF self-care synopsis (take medications as prescribed); CMS CCM 'oversee medication self-management'."),
    I("MA-2", "Provide a practical adherence aid.",
      "Does the plan include a specific step to set up an adherence aid such as a pill organizer, blister packs, phone reminders, or refill synchronization?",
      "Names at least one concrete aid and who sets it up.",
      "No aid, or only 'remember to take your medications'.",
      "efficiency", ["S19", "S21"], "JCPP implement step; CMS CCM medication self-management."),
    I("MA-3", "Address cost or access barriers to medications.",
      "Does the plan include a specific step to lower medication cost or access barriers (generic substitution request, 90-day supply, mail-order/delivery, copay review)?",
      "Names one concrete cost/access action routed to a pharmacist, prescriber, or plan.",
      "No cost/access step, or suggests options without a responsible person.",
      "equity", ["S01", "S02"], "ADA 1.7 financial barriers; 2025 BP guideline take-home 3 (team-based care addresses access to medications)."),
    I("MA-4", "Route side effects and non-adherence back to the prescriber.",
      "Does the plan include a specific step that tells the patient not to stop medications on their own and that reports side effects or non-adherence to the prescriber or pharmacist for regimen review?",
      "States both: report side effects/concerns AND a care-team handoff to the prescriber or pharmacist.",
      "No handoff, or advice that lets the patient stop or change a medication without clinician input.",
      "safety", ["S02", "S19"], "2025 BP guideline: follow-up of adherence and response monthly until control (search summary); JCPP follow-up/collaboration.", eval_only=True),
])

CATS["HYPERTENSION"] = dict(kind="clinical", label="hypertension", items=[
    I("HTN-1", "Set up home blood pressure monitoring with a validated upper-arm cuff.",
      "Does the plan include a specific step to obtain a validated upper-arm BP monitor, teach measurement technique, and share readings with the care team?",
      "Names home BP monitoring with at least two of: device type/validation, technique, schedule, how readings are sent.",
      "No home monitoring, or a smartwatch/cuffless device as the BP measure, or 'check BP sometimes'.",
      "safety", ["S02"], "2025 BP guideline take-home 9 (home BP monitoring with team interaction; avoid cuffless devices)."),
    I("HTN-2", "Schedule follow-up for BP and adherence monthly until BP is at goal.",
      "Does the plan include a specific step to recheck BP and medication adherence about monthly until BP is controlled?",
      "States a monthly (or shorter) BP/adherence check until control, then a longer interval.",
      "No BP-specific interval, or an interval longer than about 1 month while BP is uncontrolled or unknown.",
      "efficiency", ["S02"], "2025 BP guideline: adults on new or intensified therapy have monthly follow-up for adherence and response until control (confirmed via search summary)."),
    I("HTN-3", "Give specific lifestyle steps for BP.",
      "Does the plan include a specific lifestyle step for BP such as sodium below 2,300 mg/day, a DASH-style eating pattern, more physical activity, or less alcohol?",
      "Names at least one quantified or concrete lifestyle action tied to BP.",
      "Only 'eat healthy' or 'exercise more' without a concrete target.",
      "safety", ["S02", "S01"], "2025 BP guideline take-home 5 and sodium recommendation (<2,300 mg/d); ADA 5.20."),
    I("HTN-4", "Give a severe-BP threshold with an escalation action.",
      "Does the plan include a specific step that tells the patient what to do if home BP is above 180/120 mm Hg, and to call 911 if that comes with chest pain, shortness of breath, weakness, or vision or speech change?",
      "States a numeric severe threshold AND the action (same-day care-team contact; emergency care if symptoms).",
      "No threshold, or a threshold without an action.",
      "safety", ["S02"], "2025 BP guideline take-home 10 (severe hypertension >180/120 without acute target organ damage treated promptly in the outpatient setting); emergency action for symptoms of acute target organ damage.", eval_only=True),
])

CATS["DIABETES"] = dict(kind="clinical", label="type 2 diabetes", items=[
    I("DM-1", "Arrange A1C testing on the recommended schedule.",
      "Does the plan include a specific step to arrange or confirm an A1C test if none was done in the last 3 months (not at goal or recent change) or 6 months (at goal)?",
      "Names A1C testing with a timeframe of 3-6 months or 'if not done in the last 3-6 months'.",
      "No A1C step, or an interval longer than 6 months.",
      "efficiency", ["S01"], "ADA 6.2: assess glycemic status at least twice yearly, about every 3 months if not at goal or after treatment change."),
    I("DM-2", "Teach hypoglycemia recognition and treatment when relevant.",
      "Does the plan include a specific step to teach low-blood-sugar signs and treatment for a patient on insulin or a sulfonylurea (or to check whether the patient takes these)?",
      "Gives hypoglycemia signs and a treatment action, or a conditional step to confirm whether the patient uses insulin/sulfonylurea and teach if so.",
      "No hypoglycemia content and no check of hypoglycemia-risk medications.",
      "safety", ["S01"], "ADA 6.18 structured hypoglycemia education for those on insulin or at risk; 6.16 glucagon for high-risk patients."),
    I("DM-3", "Refer to diabetes self-management education and support or medical nutrition therapy.",
      "Does the plan include a specific step to refer the patient to DSMES or to a registered dietitian nutritionist?",
      "Names a DSMES program or RDN referral (in person or telehealth).",
      "No referral, or diet advice only from the care manager without referral.",
      "efficiency", ["S01"], "ADA 5.1, 5.2 (DSMES at diagnosis, annually, when not at goal); 5.10 (MNT by RDN)."),
    I("DM-4", "Arrange annual complication screening.",
      "Does the plan include a specific step to arrange or confirm a retinal eye exam, a foot exam, and kidney tests (urine albumin-to-creatinine ratio and eGFR)?",
      "Names at least two of the three screenings with a timeframe or 'if not done in the past year'.",
      "No complication screening, or only one screening named.",
      "safety", ["S01"], "ADA 11.1a (UACR and eGFR at least annually); 12.4-12.5 (eye exam; every 1-2 years if normal); 12.23 (comprehensive foot evaluation at least annually).", eval_only=True),
])

CATS["MENTAL_HEALTH"] = dict(kind="clinical", label="mental health condition", items=[
    I("MH-1", "Connect the patient to a behavioral health clinician.",
      "Does the plan include a specific step to make or confirm a behavioral health appointment (named service or clinician type) with a warm handoff or scheduling help?",
      "Names the service type and a concrete scheduling or handoff action.",
      "Only 'consider counseling' or a phone number without scheduling help.",
      "efficiency", ["S10", "S21"], "VA/DoD MDD Rec 3 collaborative/integrated care; CMS CCM coordination with outside practitioners."),
    I("MH-2", "Use a standard symptom measure at baseline and follow-up.",
      "Does the plan include a specific step to measure symptoms with a validated tool (e.g., PHQ-9, GAD-7) now and again at a stated interval?",
      "Names a validated instrument AND a repeat interval.",
      "No instrument, or a single measurement without repeat.",
      "safety", ["S10", "S07"], "VA/DoD MDD Rec 2 (measurement-based care); USPSTF anxiety (GAD instruments).", eval_only=True),
    I("MH-3", "Give a crisis plan with 988.",
      "Does the plan include a specific step that gives crisis instructions, including 988 (call, text, or chat) and 911 for immediate danger?",
      "Names 988 or an equivalent 24/7 crisis line AND when to use it.",
      "No crisis resource.",
      "safety", ["S16", "S10"], "988 Lifeline; VA/DoD MDD suicide-risk assessment."),
    I("MH-4", "Coordinate between primary care and behavioral health.",
      "Does the plan include a specific step to share information (with consent) between the primary care clinician and the behavioral health clinician?",
      "Names consent/release and a communication action between the two clinicians or a collaborative-care manager.",
      "No coordination step.",
      "efficiency", ["S10", "S21"], "VA/DoD MDD Rec 3; CMS CCM communication with outside providers about psychosocial needs."),
])

CATS["DEPRESSION"] = dict(kind="clinical", label="major depression", items=[
    I("DEP-1", "Track depression with the PHQ-9 at stated intervals.",
      "Does the plan include a specific step to give the PHQ-9 now and repeat it at a stated interval to guide treatment?",
      "Names PHQ-9 (or another validated depression measure) AND a repeat interval.",
      "No measure, or measure without repeat.",
      "safety", ["S10", "S06"], "VA/DoD MDD Rec 2; USPSTF depression (validated instruments).", eval_only=True),
    I("DEP-2", "Connect the patient to evidence-based depression treatment.",
      "Does the plan include a specific step to start or confirm psychotherapy or an evaluation by a prescriber for antidepressant treatment?",
      "Names psychotherapy referral or prescriber evaluation with a scheduling action.",
      "Only self-care tips (sleep, exercise) without connection to treatment.",
      "safety", ["S10"], "VA/DoD MDD Rec 6 (psychotherapy or pharmacotherapy)."),
    I("DEP-3", "Use a collaborative-care follow-up with the primary care team.",
      "Does the plan include a specific step for a care manager to follow up on depression treatment and report to the primary care clinician?",
      "Names care-manager check-ins AND reporting to the PCP or psychiatric consultant.",
      "No structured follow-up of depression treatment.",
      "efficiency", ["S10"], "VA/DoD MDD Rec 3 (collaborative/integrated care, strong for)."),
    I("DEP-4", "Ask about suicidal thoughts and give a crisis plan.",
      "Does the plan include a specific step to ask about thoughts of self-harm and give 988 and same-day escalation if present?",
      "Names suicide-risk inquiry AND a crisis resource/escalation.",
      "No suicide-risk step.",
      "safety", ["S10", "S16"], "VA/DoD MDD algorithm (suicide risk assessment); 988 Lifeline. USPSTF I statement applies to screening of asymptomatic adults, not to patients with diagnosed depression."),
])

CATS["ANXIETY"] = dict(kind="clinical", label="anxiety disorder", items=[
    I("ANX-1", "Track anxiety with the GAD-7 at stated intervals.",
      "Does the plan include a specific step to give the GAD-7 (or GAD-2) now and repeat it at a stated interval?",
      "Names the instrument AND a repeat interval.",
      "No instrument, or no repeat.",
      "safety", ["S07"], "USPSTF anxiety 2023 (GAD-2/GAD-7 most studied)."),
    I("ANX-2", "Connect the patient to evidence-based anxiety treatment.",
      "Does the plan include a specific step to refer to psychotherapy such as CBT or to a prescriber evaluation?",
      "Names CBT/psychotherapy referral or prescriber evaluation with a scheduling action.",
      "Only relaxation tips without connection to treatment.",
      "safety", ["S07"], "USPSTF anxiety 2023 lists CBT and other psychotherapies and pharmacotherapy as treatments. No verified US anxiety-treatment guideline was located; physician reviewer to confirm."),
    I("ANX-3", "Screen for co-occurring depression and substance use.",
      "Does the plan include a specific step to screen for depression (e.g., PHQ-9) and alcohol or substance use?",
      "Names both a depression screen and an alcohol/substance screen.",
      "Neither, or only one.",
      "safety", ["S06", "S09"], "USPSTF depression 2023; USPSTF unhealthy alcohol use 2018.", eval_only=True),
])

CATS["ASTHMA_COPD"] = dict(kind="clinical", label="asthma or COPD", items=[
    I("AC-1", "Provide a written action plan.",
      "Does the plan include a specific step to give or review a written asthma or COPD action plan (what to do when symptoms worsen)?",
      "Names a written action plan with at least one worsening step.",
      "No action plan.",
      "safety", ["S04", "S03"], "GINA: give all patients a written asthma action plan; GOLD 2026: written action plan for self-management."),
    I("AC-2", "Check inhaler technique and adherence.",
      "Does the plan include a specific step for a pharmacist or clinician to watch the patient use the inhaler and correct technique?",
      "Names an observed technique check (teach-back or demonstration) by a named role.",
      "Only 'use your inhaler as prescribed'.",
      "safety", ["S04", "S03"], "GINA: check inhaler technique at every opportunity; GOLD: inhaler technique and adherence assessed regularly."),
    I("AC-3", "Schedule respiratory review, including after an exacerbation.",
      "Does the plan include a specific step to schedule a review at a stated interval and within 1 week after any asthma exacerbation (within 1 month after a COPD hospital discharge)?",
      "States a review interval AND the post-exacerbation timeframe.",
      "No respiratory-specific review interval.",
      "efficiency", ["S04", "S03"], "GINA: review 1-3 months after start, then every 3-12 months; within 1 week after exacerbation. GOLD: early follow-up within 1 month after discharge."),
    I("AC-4", "Review vaccinations and non-drug care.",
      "Does the plan include a specific step to check influenza, pneumococcal, COVID-19, and RSV vaccination status and, for COPD, pulmonary rehabilitation and tobacco cessation?",
      "Names vaccine review AND (for COPD) rehab or tobacco step; for asthma, vaccine review is sufficient.",
      "No vaccine review.",
      "safety", ["S03", "S04"], "GOLD 2026 vaccination, pulmonary rehabilitation, smoking cessation.", eval_only=True),
])

CATS["HEART_FAILURE"] = dict(kind="clinical", label="congestive heart failure", items=[
    I("HF-1", "Teach daily weights and symptom monitoring with action steps.",
      "Does the plan include a specific step to weigh daily at the same time, watch for swelling or breathlessness, and call the care team at a stated weight-gain threshold?",
      "Names daily weights AND a call-in threshold or symptom trigger set by the care team.",
      "No self-monitoring, or monitoring without an action.",
      "safety", ["S05"], "2022 HF guideline 7.1.1 Rec 2 (self-care education and support, COR 1); synopsis: monitor signs and symptoms and know what to do."),
    I("HF-2", "Give self-care steps for sodium and medications.",
      "Does the plan include a specific step on limiting sodium and taking HF medications as prescribed?",
      "Names a sodium step AND a medication-taking step.",
      "Neither, or generic 'healthy diet'.",
      "safety", ["S05"], "2022 HF guideline 7.1.1 synopsis (take medications as prescribed, restrict sodium, stay active, get vaccinations)."),
    I("HF-3", "Link the patient to multidisciplinary HF follow-up.",
      "Does the plan include a specific step for follow-up with the HF clinician or primary care team at a stated interval, and within 7 days after any hospital discharge?",
      "Names HF follow-up with an interval AND the 7-day post-discharge rule.",
      "No HF-specific follow-up.",
      "efficiency", ["S05"], "2022 HF guideline 7.1.1 Rec 1 (multidisciplinary care, COR 1); early follow-up generally within 7 days after discharge (COR 2a)."),
    I("HF-4", "Review respiratory vaccinations.",
      "Does the plan include a specific step to check influenza, pneumococcal, and COVID-19 vaccination status?",
      "Names a vaccine status review or arranges vaccination.",
      "No vaccine step.",
      "safety", ["S05"], "2022 HF guideline 7.1.1 Rec 3 (vaccination against respiratory illness, COR 2a).", eval_only=True),
])

CATS["SUBSTANCE_USE"] = dict(kind="clinical", label="substance use disorder", items=[
    I("SUD-1", "Arrange evaluation for medication treatment where indicated.",
      "Does the plan include a specific step to arrange a prescriber evaluation for medication treatment (e.g., buprenorphine, methadone, or naltrexone for opioid use disorder)?",
      "Names a medication-treatment evaluation or referral with a scheduling action (or states why none applies to the substance).",
      "No treatment evaluation; counseling-only plan for opioid use.",
      "safety", ["S12", "S13", "S11"], "ASAM 2020 OUD; SAMHSA TIP 63; VA/DoD SUD 2021."),
    I("SUD-2", "Connect to psychosocial treatment or recovery support.",
      "Does the plan include a specific step to connect the patient to SUD counseling, a treatment program, or peer recovery support?",
      "Names a service and a referral or scheduling action.",
      "Only 'consider treatment'.",
      "efficiency", ["S11", "S12"], "VA/DoD SUD 2021 Module A; ASAM 2020 Part 7 psychosocial treatment with medications."),
    I("SUD-3", "Arrange infectious disease testing.",
      "Does the plan include a specific step to arrange or confirm HIV and hepatitis B and C testing?",
      "Names HIV and hepatitis testing (or 'if not done').",
      "No testing step.",
      "safety", ["S12"], "ASAM 2020 assessment: tests for TB, hepatitis B and C, and HIV."),
    I("SUD-4", "Make sure the patient has naloxone.",
      "Does the plan include a specific step to obtain naloxone (OTC or prescribed) and teach the patient and household how to use it when opioids or drugs that can contain fentanyl are used?",
      "Names naloxone access AND training of patient or family.",
      "No naloxone step.",
      "safety", ["S12", "S15"], "ASAM 2020 (naloxone to patients with OUD; train patients and family); FDA OTC naloxone 2023.", eval_only=True),
])

CATS["OTHER_MENTAL_BEHAVIORAL|CARE_FOR_MH_BH"] = dict(kind="clinical", label="behavioral health condition", items=[
    I("BH-1", "Confirm an active behavioral health care connection.",
      "Does the plan include a specific step to make or confirm a behavioral health appointment with scheduling help?",
      "Names the service type AND a scheduling or handoff action.",
      "Only 'consider counseling'.",
      "efficiency", ["S10", "S21"], "VA/DoD MDD Rec 3 (integrated care); CMS CCM coordination."),
    I("BH-2", "Coordinate information between primary care and behavioral health.",
      "Does the plan include a specific step to obtain consent and share the care plan between the PCP and the behavioral health clinician?",
      "Names consent/release AND a communication action.",
      "No coordination step.",
      "efficiency", ["S21", "S10"], "CMS CCM care plan shared within and outside the practice; VA/DoD collaborative care.", eval_only=True),
    I("BH-3", "Give a crisis plan with 988.",
      "Does the plan include a specific step that gives crisis instructions including 988 and 911 for immediate danger?",
      "Names 988 or an equivalent 24/7 line AND when to use it.",
      "No crisis resource.",
      "safety", ["S16"], "988 Lifeline."),
])

CATS["MEDICATION_OPTIMIZATION"] = dict(kind="clinical", label="complex medication regimen", items=[
    I("MO-1", "Arrange a pharmacist comprehensive medication review.",
      "Does the plan include a specific step for a pharmacist to do a comprehensive medication review?",
      "Names a pharmacist review with a timeframe or scheduling action.",
      "No pharmacist review.",
      "safety", ["S19", "S21"], "JCPP Pharmacists' Patient Care Process; CMS CCM medication review."),
    I("MO-2", "Check for interactions, duplications, and high-risk drugs.",
      "Does the plan include a specific step to check for drug interactions, duplicate therapies, and (age 65 or older) potentially inappropriate medications on the AGS Beers Criteria?",
      "Names an interaction/duplication check; for patients 65+, also names a Beers or high-risk drug check.",
      "No safety check of the regimen.",
      "safety", ["S18", "S21"], "AGS Beers 2023; CMS CCM 'review medications and any potential interactions'.", eval_only=True),
    I("MO-3", "Give the patient an updated, simple medication list or schedule.",
      "Does the plan include a specific step to give the patient a written, current medication list or schedule?",
      "Names a written list or schedule given to the patient.",
      "No list or schedule.",
      "efficiency", ["S20"], "NPSG.03.06.01 (maintain and communicate accurate medication information)."),
    I("MO-4", "Send regimen-change recommendations to the prescriber.",
      "Does the plan include a specific step for the pharmacist or care manager to send simplification or deprescribing suggestions to the prescriber for decision?",
      "Names prescriber communication of recommendations (care team does not change medications itself).",
      "No prescriber communication, or the plan directs the patient to stop or change a medication without a prescriber.",
      "safety", ["S19"], "JCPP collaborate/implement step."),
])

CATS["SMOKING_CESSATION"] = dict(kind="clinical", label="tobacco use", items=[
    I("SM-1", "Ask about tobacco use and advise quitting.",
      "Does the plan include a specific step to document current tobacco use and give clear advice to quit?",
      "Names tobacco-use assessment AND quit advice (readiness assessment acceptable).",
      "No tobacco step.",
      "safety", ["S08"], "USPSTF 2021 A: ask all adults, advise to stop."),
    I("SM-2", "Connect to behavioral cessation support.",
      "Does the plan include a specific step to connect the patient to counseling, e.g., 1-800-QUIT-NOW or a cessation program?",
      "Names the quitline or a counseling program AND a referral or call action.",
      "Only 'try to quit'.",
      "efficiency", ["S08", "S17"], "USPSTF 2021 behavioral interventions; CDC 1-800-QUIT-NOW."),
    I("SM-3", "Arrange FDA-approved cessation medication.",
      "Does the plan include a specific step to arrange evaluation for nicotine replacement, varenicline, or bupropion with a prescriber or pharmacist?",
      "Names cessation pharmacotherapy evaluation with a responsible role.",
      "No medication step for a nonpregnant adult.",
      "safety", ["S08", "S01"], "USPSTF 2021 A (FDA-approved pharmacotherapy for nonpregnant adults); ADA 5.40 combination counseling and pharmacotherapy.", eval_only=True),
])

CATS["ALCOHOL_USE"] = dict(kind="clinical", label="alcohol use disorder", items=[
    I("AUD-1", "Give brief counseling with a measured drinking goal.",
      "Does the plan include a specific step to measure drinking (e.g., AUDIT-C or drinks per week) and set a reduction or abstinence goal?",
      "Names a measure AND a goal.",
      "Only 'cut back on alcohol'.",
      "safety", ["S09"], "USPSTF 2018 B (brief behavioral counseling)."),
    I("AUD-2", "Arrange evaluation for alcohol use disorder medication.",
      "Does the plan include a specific step to arrange a prescriber evaluation for AUD medication (e.g., naltrexone or acamprosate)?",
      "Names AUD pharmacotherapy evaluation with a scheduling action.",
      "No medication evaluation.",
      "safety", ["S14", "S11"], "APA 2018 AUD guideline; VA/DoD SUD 2021."),
    I("AUD-3", "Connect to AUD treatment or mutual-help support.",
      "Does the plan include a specific step to connect the patient to counseling, a treatment program, or a mutual-help group?",
      "Names a service AND a referral or scheduling action.",
      "Only 'consider support groups'.",
      "efficiency", ["S11"], "VA/DoD SUD 2021 Module A."),
    I("AUD-4", "Warn about withdrawal and arrange supervised withdrawal if needed.",
      "Does the plan include a specific step to warn against stopping heavy daily drinking abruptly without medical advice and to arrange withdrawal assessment?",
      "Names withdrawal risk AND a clinician assessment for withdrawal management.",
      "No withdrawal content, or advice to quit 'cold turkey' without assessment.",
      "safety", ["S11"], "VA/DoD SUD 2021 Module B (stabilization and withdrawal).", eval_only=True),
])

# ---------- social ----------
CLOSED = lambda cid, need, src: I(
    f"{cid}-CL", f"Confirm the {need} referral was completed (closed loop).",
    f"Does the plan include a specific step to check at a stated time whether the patient actually received the {need} service or completed the application?",
    "Names a follow-up check of the referral outcome with a timeframe and a next step if it failed.",
    "Referral only, with no check of the outcome.",
    "equity", src, "NASEM 2019 'assistance'; CMS AHC Model assistance track (connection to community services).", eval_only=True)

CATS["TRANSPORTATION"] = dict(kind="social", label="transportation", items=[
    I("TR-1", "Arrange Medicaid non-emergency medical transportation.",
      "Does the plan include a specific step to book Medicaid NEMT (through the health plan or state broker) for medical visits?",
      "Names NEMT or the plan's ride benefit AND a booking action or lead time.",
      "Tells the patient to 'find a ride', or names only paid options.",
      "equity", ["S24", "S23"], "42 CFR 431.53; CMCS bulletin 12 Jul 2021; AHC core domain."),
    I("TR-2", "Reduce trips the patient must make.",
      "Does the plan include a specific step to cut travel needs, such as telehealth visits, home visits, pharmacy delivery or mail-order, or combining appointments?",
      "Names at least one travel-reducing adjustment.",
      "No adjustment to the plan for transportation limits.",
      "equity", ["S22"], "NASEM 'adjustment'."),
    CLOSED("TR", "transportation", ["S22", "S23"]),
])

CATS["FOOD_INSECURITY"] = dict(kind="social", label="food insecurity", items=[
    I("FI-1", "Help the patient apply for SNAP.",
      "Does the plan include a specific step to help the patient apply for SNAP through the state agency (or confirm current benefits)?",
      "Names SNAP application help or benefit check with a responsible role.",
      "No SNAP step, or only 'look into food stamps'.",
      "equity", ["S25"], "USDA FNS SNAP eligibility and application."),
    I("FI-2", "Connect to immediate food resources.",
      "Does the plan include a specific step to connect the patient to a food pantry or food bank (e.g., through 211) for food now?",
      "Names an immediate food source and how to reach it.",
      "Only long-term benefits with nothing for immediate need.",
      "equity", ["S35", "S23"], "211; AHC core domain."),
    I("FI-3", "Adjust the clinical plan to food access.",
      "Does the plan include a specific step to tailor diet advice or flag hypoglycemia risk (e.g., insulin or sulfonylurea with skipped meals) to the prescriber because of limited food?",
      "Names a concrete clinical adjustment tied to food access, or WIC referral if pregnant/postpartum or a child under 5 is in the household.",
      "Generic diet advice that ignores cost or access.",
      "safety", ["S01", "S22", "S26"], "ADA 1.7 (social determinants inform treatment decisions); NASEM 'adjustment'; WIC."),
    CLOSED("FI", "food", ["S22", "S23"]),
])

CATS["FINANCIAL"] = dict(kind="social", label="financial strain", items=[
    I("FIN-1", "Screen for and help enroll in income supports.",
      "Does the plan include a specific step to check eligibility for benefits (e.g., SNAP, LIHEAP) or free tax preparation for credits such as the EITC (VITA)?",
      "Names at least one benefit or VITA referral with a responsible role.",
      "Only 'look for financial help'.",
      "equity", ["S25", "S27", "S36", "S35"], "SNAP; LIHEAP; IRS VITA; 211."),
    I("FIN-2", "Lower out-of-pocket health care costs.",
      "Does the plan include a specific step to remove cost barriers in the plan itself (generic or no-copay options, covered services, cost-free alternatives)?",
      "Names one concrete cost-lowering action, or states that recommended items are no-cost under Medicaid.",
      "Recommends paid items (devices, programs) without addressing cost.",
      "equity", ["S01", "S22"], "ADA 1.7 financial barriers; NASEM 'adjustment'."),
    CLOSED("FIN", "financial assistance", ["S22", "S23"]),
])

CATS["HOUSING_INSECURITY"] = dict(kind="social", label="housing insecurity", items=[
    I("HI-1", "Connect to housing assistance or coordinated entry.",
      "Does the plan include a specific step to connect the patient to the local coordinated entry access point or housing assistance (e.g., through 211)?",
      "Names coordinated entry, a housing agency, or 211 with a referral action.",
      "Only 'look for housing'.",
      "equity", ["S34", "S35", "S23"], "HUD CPD-17-01 coordinated entry; 211; AHC core domain."),
    I("HI-2", "Refer for legal help if eviction is possible.",
      "Does the plan include a specific step to refer to civil legal aid when the patient faces eviction or a housing notice?",
      "Names legal aid referral (conditional on eviction risk acceptable).",
      "No legal step when eviction is mentioned or possible.",
      "equity", ["S28"], "Legal Services Corporation."),
    I("HI-3", "Adjust the care plan for unstable housing.",
      "Does the plan include a specific step to adapt care to unstable housing (medication storage, reliable contact method, flexible appointments)?",
      "Names one concrete adjustment.",
      "No adjustment.",
      "safety", ["S22", "S05"], "NASEM 'adjustment'; HF guideline lists housing stability among social determinants to assess."),
    CLOSED("HI", "housing", ["S22", "S23"]),
])

CATS["EMPLOYMENT"] = dict(kind="social", label="employment", items=[
    I("EMP-1", "Refer to workforce services.",
      "Does the plan include a specific step to refer the patient to an American Job Center or a local job-training program?",
      "Names the job center or program with a referral action.",
      "Only 'look for work'.",
      "equity", ["S32", "S23"], "DOL American Job Centers; AHC supplemental domain."),
    I("EMP-2", "Fit care around work.",
      "Does the plan include a specific step to schedule care around the patient's work hours (evening, weekend, or telehealth options)?",
      "Names a schedule adjustment.",
      "No adjustment.",
      "equity", ["S22"], "NASEM 'adjustment'."),
    CLOSED("EMP", "employment", ["S22", "S23"]),
])

CATS["FOOD_DIET_NUTRITION"] = dict(kind="social", label="diet and nutrition", items=[
    I("FDN-1", "Refer to a registered dietitian or nutrition program.",
      "Does the plan include a specific step to refer the patient to a registered dietitian nutritionist or a nutrition education program?",
      "Names an RDN or program referral.",
      "No referral.",
      "efficiency", ["S01"], "ADA 5.10 (MNT by RDN)."),
    I("FDN-2", "Give concrete diet targets tied to the patient's conditions.",
      "Does the plan include a specific dietary step tied to the patient's conditions (e.g., sodium under 2,300 mg/day, fewer sugar-sweetened drinks, more whole foods)?",
      "Names at least one concrete diet target.",
      "Only 'eat healthy'.",
      "safety", ["S01", "S02", "S30"], "ADA 5.20, 5.25; 2025 BP guideline sodium; Dietary Guidelines 2025-2030."),
    I("FDN-3", "Make diet advice affordable and culturally fitting.",
      "Does the plan include a specific step that adapts food advice to the patient's budget, food access, and food culture?",
      "Names an adaptation (low-cost options, culturally familiar foods, available stores).",
      "Advice that ignores cost or culture.",
      "equity", ["S01", "S22"], "ADA 5.1 (culturally appropriate DSMES), 1.7; NASEM 'adjustment'.", eval_only=True),
])

CATS["SOCIAL_CONNECTION"] = dict(kind="social", label="social connection", items=[
    I("SC-1", "Connect to a community or peer group.",
      "Does the plan include a specific step to connect the patient to a community group, peer support, senior or faith center, or similar activity?",
      "Names a group or activity type AND a connection action.",
      "Only 'spend time with others'.",
      "equity", ["S29", "S23"], "Surgeon General 2023 advisory; AHC supplemental domain (social isolation)."),
    I("SC-2", "Set up regular check-in contact.",
      "Does the plan include a specific step for regular check-ins by a CHW, peer, or care manager?",
      "Names a check-in schedule and role.",
      "No regular contact.",
      "efficiency", ["S29", "S01"], "Surgeon General 2023; ADA 1.8 (support from community health workers)."),
    CLOSED("SC", "social connection", ["S22", "S29"]),
])

CATS["LEGAL"] = dict(kind="social", label="legal", items=[
    I("LEG-1", "Refer to civil legal aid.",
      "Does the plan include a specific step to refer the patient to an LSC-funded legal aid office or a medical-legal partnership?",
      "Names legal aid referral with an action.",
      "Only 'get a lawyer'.",
      "equity", ["S28"], "Legal Services Corporation."),
    I("LEG-2", "Name the legal issue and any deadline.",
      "Does the plan include a specific step to record the legal problem type (e.g., eviction, benefits denial) and any court or notice deadline?",
      "Names the issue type AND checks for a deadline.",
      "No issue or deadline identified.",
      "safety", ["S28", "S22"], "LSC (common civil legal problems); NASEM 'awareness'."),
    CLOSED("LEG", "legal aid", ["S22", "S28"]),
])

CATS["CHILDCARE"] = dict(kind="social", label="childcare", items=[
    I("CC-1", "Help apply for child care assistance.",
      "Does the plan include a specific step to help the patient apply for child care subsidies (CCDF) through the state agency?",
      "Names child care subsidy application help.",
      "Only 'find childcare'.",
      "equity", ["S31", "S23"], "ACF CCDF; AHC supplemental domain (child care)."),
    I("CC-2", "Fit care around child care limits.",
      "Does the plan include a specific step to schedule visits around child care (telehealth, home visits, child-friendly times)?",
      "Names a schedule adjustment.",
      "No adjustment.",
      "equity", ["S22"], "NASEM 'adjustment'."),
    CLOSED("CC", "child care", ["S22", "S23"]),
])

CATS["HOUSING_QUALITY_SAFETY"] = dict(kind="social", label="housing quality and safety", items=[
    I("HQ-1", "Identify the home hazard and refer for repair or inspection.",
      "Does the plan include a specific step to identify the hazard (mold, pests, no heat, lead, carbon monoxide) and refer to a healthy-homes program or local housing inspection?",
      "Names the hazard type AND a referral.",
      "No hazard-specific step.",
      "safety", ["S33", "S23"], "HUD Healthy Homes; AHC housing domain (poor housing quality)."),
    I("HQ-2", "Link home hazards to the clinical plan.",
      "Does the plan include a specific step that reduces exposure tied to the patient's conditions (e.g., asthma triggers such as mold or pests)?",
      "Names a hazard-to-condition step, or states no condition link applies.",
      "Ignores a hazard relevant to a listed condition.",
      "safety", ["S04", "S22"], "GINA risk-factor and environmental considerations; NASEM 'adjustment'."),
    CLOSED("HQ", "home repair or inspection", ["S22", "S23"]),
])

CATS["UTILITIES"] = dict(kind="social", label="utilities", items=[
    I("UT-1", "Help apply for LIHEAP.",
      "Does the plan include a specific step to help the patient apply for LIHEAP energy bill assistance?",
      "Names LIHEAP application help.",
      "Only 'call the utility'.",
      "equity", ["S27", "S23"], "ACF LIHEAP; AHC core domain."),
    I("UT-2", "Address a shutoff or crisis now.",
      "Does the plan include a specific step for an active or near shutoff (LIHEAP crisis assistance, utility payment plan, 211)?",
      "Names a crisis action, or states no shutoff is present.",
      "No crisis step when shutoff is possible.",
      "safety", ["S27", "S35"], "LIHEAP crisis assistance; 211."),
    CLOSED("UT", "utility assistance", ["S22", "S23"]),
])

# ---------- emit ----------
items_total = sum(len(c["items"]) for c in CATS.values())
eval_total = sum(sum(i["eval_only"] for i in c["items"]) for c in CATS.values())

doc = OrderedDict()
doc["title"] = "Patient-specific care-plan requirement map (DRAFT)"
doc["status"] = "DRAFT - requires physician approval before use"
doc["date"] = "2026-10-17"
doc["inputs_assumed"] = "age, sex, race/ethnicity, clinical goal categories, social-need categories; no labs, medications, or notes"
doc["scoring_rule"] = "For each item that applies, an extractor answers the yes/no question from the plan text alone using the met/not_met criteria."
doc["domains"] = {"safety": "clinical safety and guideline-concordant clinical steps",
                  "efficiency": "care process, timing, coordination",
                  "equity": "social needs and feasibility of the plan for the patient"}
doc["counts"] = {"categories": len(CATS), "items": items_total, "eval_only": eval_total,
                 "per_category": {k: {"items": len(v["items"]), "eval_only": sum(i["eval_only"] for i in v["items"])} for k, v in CATS.items()}}
doc["sources"] = [OrderedDict(id=s[0], title=s[1], body=s[2], year=s[3], url_or_doi=s[4], verified=s[5], verification_note=s[6]) for s in SOURCES]
doc["categories"] = OrderedDict()
for k, v in CATS.items():
    doc["categories"][k] = OrderedDict(kind=v["kind"], label=v["label"], aliases=k.split("|") if "|" in k else [k], items=v["items"])

# check every cited source id exists
sids = {s[0] for s in SOURCES}
for k, v in CATS.items():
    for it in v["items"]:
        for s in it["sources"]:
            assert s in sids, (it["id"], s)
ids = [it["id"] for v in CATS.values() for it in v["items"]]
assert len(ids) == len(set(ids)), "duplicate ids"

with open(OUT + "requirement_map_DRAFT.json", "w") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)

# ---------- markdown ----------
L = []
L.append("# Care-plan requirement map (DRAFT)\n")
L.append("> **DRAFT. This map needs review and approval by a physician before anyone uses it to score, train, or tune a model.** It has not been clinically validated. Each requirement is a care-management step that a reviewer can check from the plan text alone. No requirement tells the care team to start, stop, or change a medication. Steps that involve medications send the question to a prescriber or pharmacist.\n")
L.append("Prepared 2026-10-17 for the outpatient AI care-plan study (adult Medicaid care-management patients; care managers, community health workers, pharmacists; non-emergency). Machine-readable copy: `requirement_map_DRAFT.json`.\n")
L.append("## How to use this map\n")
L.append("Each patient context gives age, sex, race/ethnicity, a set of clinical goal categories, and a set of social-need categories. The applicable items for a patient are the universal items, plus the items of each clinical category in the context, plus the items of each social category in the context. Some items carry their own condition inside the MET rule (for example, MO-2 requires a Beers Criteria check only at age 65 or older, and SUD-4 requires naloxone only when opioids or drugs that can contain fentanyl are in use).For each applicable item, an LLM extractor reads the plan and answers the yes/no question. It uses the MET and NOT MET lines as the decision rule. A conditional step (e.g., \"if you take insulin, ...\") counts as MET when the input does not say whether the condition applies.\n")
L.append("The optimizer does not see items marked `eval_only` (shown as **E** in the tables). We held out one item per category. In the clinical categories, each held-out item is a safety step that the other items in its category do not imply (for example, the severe-BP threshold, annual diabetes complication screening, respiratory vaccines, naloxone, alcohol withdrawal safety, and cessation pharmacotherapy). The crisis item (988) is held out in every category where it appears (MH-3, DEP-4, BH-3), and so is the general red-flag item U-2, so the optimizer cannot learn these steps from another category. Some held-out items overlap partly with visible items in other categories: MA-4 with MO-4 (prescriber handoff) and MO-2 with U-3 (medication review). In every social category except FOOD_DIET_NUTRITION, the held-out item is the closed-loop referral check. FOOD_DIET_NUTRITION holds out FDN-3 (affordable, culturally fitting advice) because it has no referral to close. The reviewer should confirm these choices. One consequence is that the optimizer never sees the general red-flag item or any 988 crisis item. It does see condition-specific action steps such as the HF-1 call-in threshold and the AC-1 written action plan.\n")
L.append("Domains: **safety** (clinical safety and guideline-concordant steps), **efficiency** (timing, coordination, care process), **equity** (social needs and whether the patient can actually carry out the plan).\n")

L.append("## Counts\n")
L.append(f"The map has {items_total} items in {len(CATS)} categories, and {eval_total} of them ({eval_total/items_total*100:.1f}%) are eval_only. OTHER_MENTAL_BEHAVIORAL and CARE_FOR_MH_BH share one category because both show the model the label \"behavioral health condition\".\n")
L.append("| Category | Kind | Label (clinical: shown to model; social: working label) | Items | eval_only |")
L.append("|---|---|---|---|---|")
for k, v in CATS.items():
    L.append(f"| {k} | {v['kind']} | {v['label']} | {len(v['items'])} | {sum(i['eval_only'] for i in v['items'])} |")
L.append(f"| **Total** | | | **{items_total}** | **{eval_total}** |\n")

L.append("## Sources\n")
L.append("We checked every source on 2026-10-17 with web search, PubMed, or a direct read of the document. The verification note says how we checked each one. \"yes\" means we confirmed the title, issuing body, year, and URL or DOI. Where we read only a secondary summary of the recommendation text, the note says so.\n")
L.append("| ID | Source | Issuing body | Year | URL / DOI | Verified | Verification note |")
L.append("|---|---|---|---|---|---|---|")
for s in SOURCES:
    L.append(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[5]} | {s[6]} |")
L.append("")

L.append("## Requirements by category\n")
for k, v in CATS.items():
    lab = f"label shown to model: \"{v['label']}\"" if v['kind']=='clinical' else ("applies to every patient" if v['kind']=='universal' else f"working label \"{v['label']}\"; model-facing label not specified")
    L.append(f"### {k} ({v['kind']}; {lab})\n")
    L.append("| ID | E | Requirement | Extraction question | MET | NOT MET | Domain | Source |")
    L.append("|---|---|---|---|---|---|---|---|")
    for it in v["items"]:
        e = "E" if it["eval_only"] else ""
        L.append(f"| {it['id']} | {e} | {it['requirement']} | {it['question']} | {it['met']} | {it['not_met']} | {it['domain']} | {', '.join(it['sources'])}: {it['source_detail']} |")
    L.append("")

L.append("## Open items for the physician reviewer\n")
L.append("1. Confirm or change the eval_only choices. The general red-flag item (U-2) and every 988 crisis item are held out, so the optimizer never sees them.")
L.append("2. ANX-2 cites the 2023 USPSTF anxiety statement, which lists the treatments. We did not find a verified US guideline on anxiety treatment. Choose a treatment source or accept the USPSTF wording.")
L.append("3. HTN-2 (monthly follow-up until BP control) relies on a search summary of the 2025 BP guideline because the full text returned HTTP 403. Check the exact wording and its class of recommendation in the guideline text.")
L.append("4. HF-1 leaves the weight-gain threshold to the care team. We did not find a verified numeric threshold in the 2022 HF guideline text.")
L.append("5. AUD-2 cites the 2018 APA AUD guideline for naltrexone and acamprosate. We confirmed that recommendation through secondary summaries only.")
L.append("6. S17 (CDC quitline page) has no year printed on the page. Its year is marked [UNVERIFIED year — check manually].")
L.append("7. USPSTF posted a draft update of the alcohol recommendation in 2025. Check whether a final version has replaced the 2018 statement before use.")
L.append("8. SNAP rules changed in 2025 (work requirements and non-citizen eligibility). FI-1 asks only for application help and makes no claim about eligibility.")

with open(OUT + "requirement_map_DRAFT.md", "w") as f:
    f.write("\n".join(L) + "\n")

print(items_total, eval_total, len(CATS))
for k, v in CATS.items():
    print(k, len(v["items"]), sum(i["eval_only"] for i in v["items"]))
