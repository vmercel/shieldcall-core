# Patent Pathway — Where to File (Official Links)

**Disclaimer:** This is practical filing navigation, **not legal advice**. Software/AI patents face §101 (Alice) scrutiny in the U.S. Use a registered patent attorney for claim drafting, prior-art clearance, and entity status (micro/small/large).

ShieldCall’s patentable *angle* is the **specific technical combination** (telephony residual fingerprinting + streaming discourse path scoring + co-activation fusion + coverage-debt adaptation loop), not “detecting scams with AI.”

---

## A. Fastest U.S. path: Provisional → Nonprovisional (12 months)

### Step 0 — Prepare before you click “file”

1. Written description (how it works + embodiments + diagrams).  
2. Drawings/flowcharts (pipeline, fusion, adaptation loop).  
3. Inventor list + ownership (assignment to company if any).  
4. Decide **micro / small / large entity** (fee level).  
5. **Do not** publicly disclose (blog, demo video, GitHub public details of claims) before filing if you want maximum international options—public disclosure can burn foreign rights.

> Your GitHub repo is currently **private** — keep critical claim language and unpublished experiments private until provisional is filed.

### Step 1 — Create USPTO.gov + Patent Center account

| What | Official link |
|------|----------------|
| **USPTO home** | https://www.uspto.gov/ |
| **Patent Center (file & manage applications)** | https://patentcenter.uspto.gov/ |
| **Patent Center info / help** | https://www.uspto.gov/patents/apply/patent-center |
| **File online overview** | https://www.uspto.gov/patents/apply |
| **Create / manage USPTO.gov account** | https://account.uspto.gov/ |

### Step 2 — Learn provisional requirements

| What | Official link |
|------|----------------|
| **Provisional application basics** | https://www.uspto.gov/patents/basics/apply/provisional-application |
| **Provisional filing guide** | https://www.uspto.gov/patents-getting-started/patent-basics/types-patent-applications/provisional-application-patent |
| **Provisional cover sheet SB/16 (Patent Center version)** | https://www.uspto.gov/sites/default/files/documents/sb0016pc.pdf |
| **All patent forms** | https://www.uspto.gov/patents/apply/forms |
| **USPTO video: drafting provisionals** | https://www.uspto.gov/learning-and-resources/uspto-videos/path-patent-part-ii-drafting-provisional-patent-applications |

### Step 3 — File the provisional **in Patent Center**

1. Go to: **https://patentcenter.uspto.gov/**  
2. Sign in.  
3. **New submission** → **Utility Provisional**.  
4. Upload specification PDF (+ drawings).  
5. Complete cover sheet (inventors, title, correspondence).  
6. Pay provisional filing fee.  
7. Save filing receipt / application number → you may mark materials **“Patent Pending.”**

**Fee schedule (verify live amounts):**  
https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule  

Provisional fee codes are listed under patent application filing fees (large / small / micro columns differ).

**Payments hub:**  
https://www.uspto.gov/learning-and-resources/fees-and-payment  

### Step 4 — Within 12 months: Nonprovisional (examined utility patent)

| What | Official link |
|------|----------------|
| **Apply for a patent (overview)** | https://www.uspto.gov/patents/basics/apply |
| **Nonprovisional utility (file in Patent Center)** | https://patentcenter.uspto.gov/ |
| **Patent process timeline** | https://www.uspto.gov/patents/basics/patent-process-overview |
| **Subject matter eligibility (Alice / MPEP 2106)** | https://www.uspto.gov/web/offices/pac/mpep/s2106.html |
| **Find a registered patent attorney/agent** | https://oedci.uspto.gov/OEDCI/ |

Nonprovisional requires: claims, oath/declaration, fees (filing + search + examination), proper drawings.

---

## B. International: PCT (optional, usually after or with U.S. filing)

Gives ~30/31 months from priority date to enter national phases (country-by-country).

| What | Official link |
|------|----------------|
| **PCT system overview (WIPO)** | https://www.wipo.int/en/web/pct-system |
| **How to file a PCT application** | https://www.wipo.int/en/web/pct-system/filing/index |
| **ePCT portal (file/manage international apps)** | https://pct.wipo.int/ePCT/ |
| **ePCT about / login** | https://pct.wipo.int/ePCT/about-epct.xhtml |
| **Create WIPO account for ePCT** | https://pct.wipo.int/wipoaccounts/en/ePCT/public/register.xhtml |
| **PCT Applicant’s Guide** | https://www.wipo.int/en/web/pct-system/guide |
| **Receiving Offices accepting ePCT** | https://pct.wipo.int/ePCTExternal/pages/EFilingServers.xhtml |

U.S. applicants often file PCT via:

- **USPTO as Receiving Office** through **Patent Center**, or  
- **WIPO International Bureau** via **ePCT**.

---

## C. Prior-art search (do this before/while drafting)

| Tool | Link |
|------|------|
| **USPTO Patent Public Search** | https://ppubs.uspto.gov/pubwebapp/ |
| **Google Patents** | https://patents.google.com/ |
| **Espacenet (EPO)** | https://worldwide.espacenet.com/ |
| **WIPO Patentscope** | https://patentscope.wipo.int/ |
| **Scholar (papers)** | https://scholar.google.com/ |

Suggested queries: see `docs/PRIOR_ART.md`.

---

## D. Trademark (optional, brand name “ShieldCall”)

Separate from patents:

| What | Link |
|------|------|
| **USPTO Trademark Center / TEAS** | https://www.uspto.gov/trademarks |
| **TESS search** | https://tmsearch.uspto.gov/ |

---

## E. Recommended sequence for *this* project

```
Week 0–2
  ├─ Freeze invention disclosure from docs/ARCHITECTURE.md + PRIOR_ART.md + code
  ├─ Attorney prior-art search
  └─ Draft provisional specification (methods, systems, diagrams)

Day F (File day)
  └─ File provisional @ https://patentcenter.uspto.gov/  → Patent Pending

Months 1–11
  ├─ Run ablations / ASVspoof+TCT experiments; add embodiments to notebook
  ├─ Optional: improve code without public claim spoilers
  └─ Draft nonprovisional claims (attorney)

Month 12
  ├─ File U.S. nonprovisional claiming priority to provisional
  └─ Optional same day: PCT via ePCT / Patent Center

Years 1–3+
  └─ Prosecution (office actions), national phase entries, continuations
```

---

## F. Entity status & cost awareness

| Resource | Link |
|----------|------|
| **Micro entity** | https://www.uspto.gov/patents/laws/micro-entity-status |
| **Small entity** | https://www.uspto.gov/web/offices/pac/mpep/s509.html |
| **Current fee schedule** | https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule |

Expect: provisional USPTO fee (low hundreds or less for micro) **plus** attorney drafting (often the dominant cost). Nonprovisional + PCT + national phases are multi-thousand to tens of thousands depending on countries and counsel.

---

## G. What to put in the provisional for ShieldCall

Minimum technical content (map to code):

1. System diagram (dual stream + fusion).  
2. TCT channel operations (bandlimit, µ-law, packet loss/PLC).  
3. STRF residual feature vector and scoring.  
4. SDTG stages, transitions, path score.  
5. CSCF: co-activation window, regimes, trajectory.  
6. PMA + coverage-debt metric + challenge-response enrollment.  
7. CSR conformal band / abstention.  
8. CTE counterfactual generation.  
9. Streaming frame timing (hop/frame).  
10. Alternate embodiments (cloud service, on-device ONNX, contact-center SBC tap).

Use `docs/ARCHITECTURE.md`, `docs/PRIOR_ART.md`, and source under `shieldcall/` as the backbone of the specification.

---

## H. One-click filing entry points (bookmark these)

1. **File U.S. provisional/nonprovisional:** https://patentcenter.uspto.gov/  
2. **USPTO apply hub:** https://www.uspto.gov/patents/apply  
3. **Fees:** https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule  
4. **International PCT:** https://pct.wipo.int/ePCT/  
5. **Find patent attorney:** https://oedci.uspto.gov/OEDCI/  
