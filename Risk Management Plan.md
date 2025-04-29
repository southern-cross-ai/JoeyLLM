# Australian LLM Project – Risk Assessment

## 1  Purpose

This document identifies and assesses the key risks associated with delivering an **Australian‑specific Large Language Model (LLM)** for our client. It supports proactive decision‑making and fulfils TechLauncher sprint reporting requirements.

## 2  Scope

The assessment covers the full lifecycle:

1. Corpus acquisition (Australian‑sourced text, audio, and code)
2. Data cleaning, annotation, and storage
3. Model training, evaluation, and alignment
4. Deployment (API & on‑prem options)
5. Ongoing monitoring, maintenance, and user support

## 3  Methodology

Risks were gathered through workshops with the team, client representatives, and academic supervisors, then classified into seven categories:

- **Compliance & Legal**
- **Ethical & Social**
- **Data Integrity & Quality**
- **Technical**
- **Security & Privacy**
- **Operational**
- **Stakeholder & Reputation**

Each risk is scored on **Likelihood (L)** and **Impact (I)** from 1 (Low) to 5 (Extreme). **Risk Rating = L × I**. Colour‑coded thresholds:

- 1‑6  Low   🟢 – Monitor
- 7‑12 Medium 🟡 – Active controls
- 13‑25 High   🔴 – Immediate action

## 4  Risk Register

|  #  | Risk                                                                                  | Category           |  L  |  I  | Rating | Primary Mitigation / Controls                                                                                         | Owner           |
| --- | ------------------------------------------------------------------------------------- | ------------------ | --- | --- | ------ | --------------------------------------------------------------------------------------------------------------------- | --------------- |
| 1   | **Collection of personal data without consent breaches *****Privacy Act 1988***       | Compliance & Legal | 3   | 5   | 15 🔴  | *Privacy‑by‑Design*; Privacy Impact Assessment (PIA); strip PII; use only public or consented data; OAIC consultation | Data Lead       |
| 2   | **Use of Indigenous Cultural & Intellectual Property (ICIP) without permission**      | Ethical & Social   | 2   | 5   | 10 🟡  | Follow AIATSIS ‘CARE’ principles; seek community approval; documented licences; exclude unclear content               | Ethics Officer  |
| 3   | **Copyright infringement from web‑scraped corpora**                                   | Compliance & Legal | 4   | 4   | 16 🔴  | Restrict to licensed datasets (Trove, AusGovInfo, CC‑BY); provenance log; legal review; TDM exception audit           | Project Manager |
| 4   | **Dataset bias leads to discriminatory outputs (e.g. against First Nations peoples)** | Ethical & Social   | 3   | 4   | 12 🟡  | Bias metrics; representative sampling; debiasing; external fairness audit                                             | ML Lead         |
| 5   | **Sensitive government data leakage during training or inference**                    | Security & Privacy | 2   | 5   | 10 🟡  | IRAP‑aligned environment; at‑rest & in‑transit encryption; role‑based access; logging & SIEM                          | DevOps          |
| 6   | **Model hallucinations produce harmful or misleading advice**                         | Technical          | 3   | 3   | 9 🟡   | RLHF tuned with Australian safety guidelines; output filters; user disclaimers; continuous eval set                   | ML Lead         |
| 7   | **GPU/compute cost overruns exceed budget**                                           | Operational        | 4   | 3   | 12 🟡  | Early budget forecast; cloud cost dashboards; use spot/commit pricing; model size optimisation                        | Finance         |
| 8   | **Vendor lock‑in with single cloud provider**                                         | Operational        | 2   | 3   | 6 🟢   | Containerised pipelines; Infrastructure‑as‑Code; periodic portability tests; multi‑cloud roadmap                      | DevOps          |
| 9   | **Timeline slippage impacts TechLauncher deliverables**                               | Stakeholder        | 3   | 3   | 9 🟡   | Detailed sprint plan with 15% buffer; weekly burndown review; early escalation path                                   | Scrum Master    |
| 10  | **Negative media coverage over AI ethics & safety**                                   | Reputation         | 2   | 4   | 8 🟡   | Transparent documentation; external ethics board sign‑off; comms contingency plan                                     | PM / Comms      |

> **Note:** A complete register with secondary controls and residual risk scores is kept in the team’s GitLab wiki and reviewed each sprint.

### 4.1 JoeyLLM Team Sprint‑Level Risk Snapshot (25 Mar 2025)

| Risk ID | Risk Description | L | I | Rating | Mitigation / Controls | Owner |
|---------|------------------|---|---|--------|-----------------------|-------|
| R1 | Environment setup delays | 3 | 4 | 12 🟡 | Pre‑session system checks; automated setup scripts; CI validator | DevOps |
| R2 | Limited GPT‑2 domain knowledge | 3 | 4 | 12 🟡 | Allocate learning sprint; pair‑programming workshops; expert consultation | ML Lead |
| R3 | HPC resource contention | 2 | 3 | 6 🟢 | Docker Hub sponsorship; queued job scheduler; off‑peak training windows | Infrastructure |
| R4 | Dataset integration complexity | 3 | 3 | 9 🟡 | Early schema review; incremental ETL tests; data contract in GitLab CI | Data Engineer |
| R5 | Internal communication gaps | 2 | 3 | 6 🟢 | Weekly stand‑ups; Slack #joey‑alerts; decision log in wiki | Scrum Master |

> **Note:** The JoeyLLM snapshot uses the same Likelihood/Impact numeric scale as the master risk register for comparability.

## 5  High‑Priority Risks – Action Plan (Model Design Sprint)

| Risk # | Immediate Action                                                                                              | Due          | Responsible       |
| ------ | ------------------------------------------------------------------------------------------------------------- | ------------ | ----------------- |
|  4     | Finalise fairness‑metric specification & representative sampling plan; schedule bias‑mitigation design review | 16 May 2025  | **Data Engineer** |
|  6     | Document safety‑alignment strategy (RLHF dataset, output filter rules); prototype validation pipeline         | 16 May 2025  | **Model Engineer** |
|  7     | Complete model‑sizing & compute‑cost simulation; refine architecture to stay within budget                    | 16 May 2025  | **Model Engineer** |
|  8     | Produce cloud‑agnostic deployment blueprint; PoC containerised training stack to minimise vendor lock‑in      | 16 May 2025  | **Model Engineer** |

*Roles consolidated to match current team structure (Data Engineer & Model Engineer).*  

## 6  Monitoring & Review  Monitoring & Review  Monitoring & Review

- **Weekly:** Risk owner updates status in stand‑up; tutor checks evidence.
- **End‑of‑Sprint:** Retro includes risk review; adjust scores & controls.
- **Quarterly:** External advisor audit (privacy & ethics).

## 7  Appendices

### 7.1 Likelihood & Impact Scales

| Score | Likelihood     | Example               |
| ----- | -------------- | --------------------- |
| 1     | Rare           | <5% chance in project |
| 3     | Possible       | 20–50% chance         |
| 5     | Almost certain | >90% chance           |

| Score | Impact     | Example                                    |
| ----- | ---------- | ------------------------------------------ |
| 1     | Negligible | No schedule slip; no legal issues          |
| 3     | Moderate   | Minor client dissatisfaction; small fine   |
| 5     | Critical   | Project failure; significant legal penalty |

### 7.2 Regulatory & Ethical References

- **Privacy Act 1988 (Cth)** & Australian Privacy Principles (APPs)
- **OAIC Guide to Big Data & Privacy**
- **Australian Government – Safe & Responsible AI in Australia (2024) Consultation**
- **AIATSIS Code of Ethics for Aboriginal and Torres Strait Islander Research (2021)**
- **ACS/OCEG AI Ethics Principles (2022)**

